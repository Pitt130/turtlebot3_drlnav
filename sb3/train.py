#!/usr/bin/env python3
"""Train or resume an SB3 TD3 agent on TurtleBot3 DRL Gymnasium."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    StopTrainingOnMaxEpisodes,
)
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

from common import build_vec_env, seed_everything, shutdown_ros


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or resume SB3 TD3 on the TurtleBot3 Gymnasium environment"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-timesteps", type=int, default=5_000)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tau", type=float, default=0.003)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--noise", choices=("normal", "ou"), default="normal")
    parser.add_argument("--noise-sigma", type=float, default=0.1)
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


def make_action_noise(kind: str, sigma: float):
    mean = np.zeros(2, dtype=np.float32)
    stddev = sigma * np.ones(2, dtype=np.float32)
    if kind == "ou":
        return OrnsteinUhlenbeckActionNoise(mean=mean, sigma=stddev)
    return NormalActionNoise(mean=mean, sigma=stddev)


def validate_args(args: argparse.Namespace) -> None:
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive")
    if args.checkpoint_freq <= 0:
        raise ValueError("--checkpoint-freq must be positive")
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
    )
    action_noise = make_action_noise(args.noise, args.noise_sigma)

    if args.resume_model:
        model = TD3.load(str(args.resume_model), env=env, device=args.device)
        model.set_random_seed(args.seed)
        model.action_noise = action_noise
        if args.resume_replay_buffer:
            model.load_replay_buffer(str(args.resume_replay_buffer))
        elif args.replay_warmup_steps > 0:
            model.learning_starts = model.num_timesteps + args.replay_warmup_steps
        reset_num_timesteps = False
    else:
        model = TD3(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=(1, "step"),
            gradient_steps=1,
            action_noise=action_noise,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            policy_kwargs={"net_arch": [512, 512]},
            verbose=1,
            device=args.device,
            seed=args.seed,
        )
        reset_num_timesteps = True

    callbacks = [
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

        model.learn(
            total_timesteps=args.total_timesteps,
            callback=CallbackList(callbacks),
            reset_num_timesteps=reset_num_timesteps,
        )

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
        env.close()
        shutdown_ros()


if __name__ == "__main__":
    main()
