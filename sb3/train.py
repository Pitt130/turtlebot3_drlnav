#!/usr/bin/env python3
"""Train or resume an SB3 TD3 agent on TurtleBot3 DRL Gymnasium."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import stable_baselines3
import torch
import torch.nn.functional as F
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    StopTrainingOnMaxEpisodes,
)
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from stable_baselines3.common.utils import polyak_update

from common import build_vec_env, seed_everything, shutdown_ros
from tomas_policy import ConfigurableTD3Policy


class TD3WithSeparateRandomWarmup(TD3):
    """Allow random-action collection to outlast the gradient warm-up.

    SB3 normally uses ``learning_starts`` for both decisions.  The Tomas
    training loop starts gradient updates once one batch is available while
    continuing to select random actions until ``OBSERVE_STEPS`` is reached.
    """

    def __init__(
        self,
        *args,
        random_action_until: int | None = None,
        critic_loss: str = "mse",
        max_grad_norm: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if random_action_until is None:
            random_action_until = self.learning_starts
        self.random_action_until = int(random_action_until)
        self.critic_loss_kind = critic_loss
        self.max_grad_norm = max_grad_norm

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        # Tomas uses pure random actions during observation and only applies
        # OU noise after switching to actor actions.
        effective_noise = (
            None if self.num_timesteps < self.random_action_until else action_noise
        )
        return super()._sample_action(
            self.random_action_until,
            action_noise=effective_noise,
            n_envs=n_envs,
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """Run standard SB3 TD3 updates with a selectable critic loss."""
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )

            with torch.no_grad():
                noise = replay_data.actions.clone().data.normal_(
                    0, self.target_policy_noise
                )
                noise = noise.clamp(
                    -self.target_noise_clip, self.target_noise_clip
                )
                next_actions = (
                    self.actor_target(replay_data.next_observations) + noise
                ).clamp(-1, 1)
                next_q_values = torch.cat(
                    self.critic_target(
                        replay_data.next_observations, next_actions
                    ),
                    dim=1,
                )
                next_q_values, _ = torch.min(
                    next_q_values, dim=1, keepdim=True
                )
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = self.critic(
                replay_data.observations, replay_data.actions
            )
            loss_function = (
                F.smooth_l1_loss
                if self.critic_loss_kind == "smooth_l1"
                else F.mse_loss
            )
            critic_loss = sum(
                loss_function(current_q, target_q_values)
                for current_q in current_q_values
            )
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
            self.critic.optimizer.step()

            if self._n_updates % self.policy_delay == 0:
                actor_loss = -self.critic.q1_forward(
                    replay_data.observations,
                    self.actor(replay_data.observations),
                ).mean()
                actor_losses.append(actor_loss.item())

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.max_grad_norm
                    )
                self.actor.optimizer.step()

                polyak_update(
                    self.critic.parameters(),
                    self.critic_target.parameters(),
                    self.tau,
                )
                polyak_update(
                    self.actor.parameters(),
                    self.actor_target.parameters(),
                    self.tau,
                )
                polyak_update(
                    self.critic_batch_norm_stats,
                    self.critic_batch_norm_stats_target,
                    1.0,
                )
                polyak_update(
                    self.actor_batch_norm_stats,
                    self.actor_batch_norm_stats_target,
                    1.0,
                )

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if actor_losses:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))



class ActionDiagnostics(BaseCallback):
    """Record executed actions and simulation timing to identify spinning."""

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.output = None

    def _on_training_start(self):
        self.output = self.path.open("w", newline="")
        self.writer = csv.writer(self.output)
        self.writer.writerow(("step", "phase", "linear_action", "angular_action",
                              "reward", "goal_distance", "goal_angle", "outcome",
                              "step_sim_dt", "odom_stamp", "scan_stamp"))

    def _on_step(self):
        action = self.locals["actions"][0]
        info = self.locals["infos"][0]
        # Sampling happens before num_timesteps is incremented by VecEnv.step.
        phase = "random" if self.num_timesteps - 1 < self.model.random_action_until else "actor"
        self.writer.writerow((self.num_timesteps, phase, float(action[0]), float(action[1]),
                              float(self.locals["rewards"][0]), info.get("goal_distance"),
                              info.get("goal_angle"), info.get("success"),
                              info.get("step_sim_dt"), info.get("odom_stamp"), info.get("scan_stamp")))
        if self.num_timesteps % 100 == 0:
            self.output.flush()
        return True

    def close(self):
        if self.output is not None:
            self.output.close()

    def _on_training_end(self):
        self.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or resume SB3 TD3 on the TurtleBot3 Gymnasium environment"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-timesteps", type=int, default=5_000)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step-sim-time", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-freq", type=int, default=2_500)
    parser.add_argument(
        "--checkpoint-replay-buffer",
        action="store_true",
        help="Also save the large replay buffer at every periodic checkpoint",
    )
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=3_000)
    parser.add_argument(
        "--random-action-until",
        type=int,
        help=(
            "Use uniformly random actions until this absolute timestep. "
            "Defaults to --learning-starts, matching standard SB3 behavior."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--critic-loss",
        choices=("mse", "smooth_l1"),
        default="mse",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        help="Clip actor and critic gradient norms to this value.",
    )
    parser.add_argument(
        "--optimizer",
        choices=("adam", "adamw"),
        default="adam",
    )
    parser.add_argument(
        "--critic-architecture",
        choices=("sb3", "tomas"),
        default="sb3",
        help="Use the standard SB3 critic or Tomas state/action split critic.",
    )
    parser.add_argument(
        "--weight-init",
        choices=("sb3", "tomas"),
        default="sb3",
        help="Use framework defaults or Tomas Xavier-uniform weights and 0.01 bias.",
    )
    parser.add_argument("--tau", type=float, default=0.003)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--noise", choices=("normal", "ou"), default="normal")
    parser.add_argument("--noise-sigma", type=float, default=0.1)
    parser.add_argument("--noise-dt", type=float, default=1.0,
                        help="OU discretization: 1.0 matches Tomas; 0.01 is SB3 default")
    parser.add_argument(
        "--normalize-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume-model", type=Path)
    parser.add_argument("--resume-vecnormalize", type=Path)
    parser.add_argument("--resume-replay-buffer", type=Path)
    parser.add_argument(
        "--replay-warmup-steps",
        type=int,
        default=3_000,
        help="Fresh transitions collected before updates when resuming without a replay buffer",
    )
    return parser.parse_args()


def make_action_noise(kind: str, sigma: float, dt: float = 1.0):
    mean = np.zeros(2, dtype=np.float32)
    stddev = sigma * np.ones(2, dtype=np.float32)
    if kind == "ou":
        return OrnsteinUhlenbeckActionNoise(mean=mean, sigma=stddev, dt=dt)
    return NormalActionNoise(mean=mean, sigma=stddev)


def validate_args(args: argparse.Namespace) -> None:
    if not np.isfinite(args.noise_dt) or args.noise_dt <= 0:
        raise ValueError("--noise-dt must be positive and finite")
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive")
    if args.checkpoint_freq <= 0:
        raise ValueError("--checkpoint-freq must be positive")
    if args.learning_starts < 0:
        raise ValueError("--learning-starts must be non-negative")
    if args.random_action_until is not None and args.random_action_until < 0:
        raise ValueError("--random-action-until must be non-negative")
    if args.max_grad_norm is not None and args.max_grad_norm <= 0:
        raise ValueError("--max-grad-norm must be positive")
    if args.resume_vecnormalize and not args.resume_model:
        raise ValueError("--resume-vecnormalize requires --resume-model")
    if args.resume_replay_buffer and not args.resume_model:
        raise ValueError("--resume-replay-buffer requires --resume-model")
    if args.resume_model and args.normalize_reward and not args.resume_vecnormalize:
        raise ValueError(
            "Resuming normalized training requires --resume-vecnormalize; "
            "use --no-normalize-reward only for models trained without VecNormalize"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    random_action_until = (
        args.learning_starts
        if args.random_action_until is None
        else args.random_action_until
    )
    output_dir = args.output_dir.resolve()
    model_dir = output_dir / "models"
    log_dir = output_dir / "logs"
    checkpoint_dir = model_dir / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = args.device.startswith("cuda") or args.device == "auto"
    seed_everything(args.seed, use_cuda=use_cuda)

    config = {
        **vars(args),
        "random_action_until": random_action_until,
        "output_dir": str(output_dir),
        "resume_model": str(args.resume_model.resolve()) if args.resume_model else None,
        "resume_vecnormalize": (
            str(args.resume_vecnormalize.resolve()) if args.resume_vecnormalize else None
        ),
        "resume_replay_buffer": (
            str(args.resume_replay_buffer.resolve()) if args.resume_replay_buffer else None
        ),
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    env = build_vec_env(
        log_dir,
        args.seed,
        training=True,
        normalize_reward=args.normalize_reward,
        gamma=args.gamma,
        vecnormalize_path=args.resume_vecnormalize,
        step_sim_time=args.step_sim_time,
    )
    action_noise = make_action_noise(args.noise, args.noise_sigma, args.noise_dt)
    optimizer_class = (
        torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    )
    use_configurable_policy = (
        args.critic_architecture != "sb3" or args.weight_init != "sb3"
    )

    if args.resume_model:
        model = TD3WithSeparateRandomWarmup.load(
            str(args.resume_model), env=env, device=args.device
        )
        model.set_random_seed(args.seed)
        model.action_noise = action_noise
        model.random_action_until = random_action_until
        model.critic_loss_kind = args.critic_loss
        model.max_grad_norm = args.max_grad_norm
        if args.resume_replay_buffer:
            model.load_replay_buffer(str(args.resume_replay_buffer))
        elif args.replay_warmup_steps > 0:
            model.learning_starts = model.num_timesteps + args.replay_warmup_steps
        reset_num_timesteps = False
    else:
        policy = ConfigurableTD3Policy if use_configurable_policy else "MlpPolicy"
        policy_kwargs = {
            "net_arch": [512, 512],
            "optimizer_class": optimizer_class,
        }
        if use_configurable_policy:
            policy_kwargs.update(
                critic_architecture=args.critic_architecture,
                xavier_init=args.weight_init == "tomas",
            )
        model = TD3WithSeparateRandomWarmup(
            policy=policy,
            env=env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            random_action_until=random_action_until,
            batch_size=args.batch_size,
            critic_loss=args.critic_loss,
            max_grad_norm=args.max_grad_norm,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=(1, "step"),
            gradient_steps=1,
            action_noise=action_noise,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=args.device,
            seed=args.seed,
        )
        reset_num_timesteps = True

    diagnostics = ActionDiagnostics(log_dir / "actions.csv")
    callbacks = [
        diagnostics,
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(checkpoint_dir),
            name_prefix="td3_stage4",
            save_replay_buffer=args.checkpoint_replay_buffer,
            save_vecnormalize=True,
            verbose=2,
        )
    ]
    if args.max_episodes is not None:
        callbacks.append(
            StopTrainingOnMaxEpisodes(max_episodes=args.max_episodes, verbose=1)
        )

    try:
        print(f"SB3 version: {stable_baselines3.__version__}")
        print(f"device request: {args.device}, CUDA available: {torch.cuda.is_available()}")
        print(f"seed: {args.seed}, output: {output_dir}")
        print(f"starting timestep: {model.num_timesteps}")
        print(f"gradient updates start at: {model.learning_starts}")
        print(f"uniformly random actions end at: {model.random_action_until}")
        print(f"critic loss: {model.critic_loss_kind}")
        print(f"max gradient norm: {model.max_grad_norm}")
        print(f"exploration noise: {args.noise}, sigma={args.noise_sigma}, dt={args.noise_dt}")
        print(f"optimizer: {args.optimizer}")
        print(f"critic architecture: {args.critic_architecture}")
        print(f"weight initialization: {args.weight_init}")

        try:
            model.learn(
                total_timesteps=args.total_timesteps,
                callback=CallbackList(callbacks),
                reset_num_timesteps=reset_num_timesteps,
            )
        except KeyboardInterrupt:
            print("Training interrupted; saving current model and replay buffer.")

        final_model = model_dir / "final_model"
        final_vecnormalize = model_dir / "vecnormalize.pkl"
        final_replay_buffer = model_dir / "replay_buffer.pkl"
        model.save(str(final_model))
        env.save(str(final_vecnormalize))
        model.save_replay_buffer(str(final_replay_buffer))
        print(f"model saved: {final_model}.zip")
        print(f"VecNormalize saved: {final_vecnormalize}")
        print(f"replay buffer saved: {final_replay_buffer}")
    finally:
        diagnostics.close()
        env.close()
        shutdown_ros()


if __name__ == "__main__":
    main()
