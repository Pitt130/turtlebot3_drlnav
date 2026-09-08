#!/usr/bin/env python3
"""Evaluate an SB3 TD3 model and report navigation and spinning metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import TD3

from common import build_vec_env, seed_everything, shutdown_ros
from turtlebot3_drl.common.settings import (
    ENABLE_BACKWARD,
    SPEED_ANGULAR_MAX,
    SPEED_LINEAR_MAX,
)


OUTCOME_NAMES = {
    0: "unknown",
    1: "success",
    2: "collision_wall",
    3: "collision_obstacle",
    4: "timeout",
    5: "tumble",
}
FIELDS = (
    "episode",
    "outcome_code",
    "outcome",
    "reward",
    "steps",
    "goal_distance",
    "mean_linear_speed",
    "mean_abs_angular_speed",
    "stationary_fraction",
    "turning_fraction",
    "spin_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an SB3 TurtleBot3 TD3 model")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vecnormalize", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step-sim-time", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stationary-linear-threshold", type=float, default=0.02)
    parser.add_argument("--spin-angular-threshold", type=float, default=0.5)
    return parser.parse_args()


def physical_action(action: np.ndarray) -> tuple[float, float]:
    if ENABLE_BACKWARD:
        linear = float(action[0] * SPEED_LINEAR_MAX)
    else:
        linear = float((action[0] + 1.0) * 0.5 * SPEED_LINEAR_MAX)
    angular = float(action[1] * SPEED_ANGULAR_MAX)
    return linear, angular


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError("--episodes and --max-steps must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.json"
    seed_everything(args.seed, use_cuda=args.device.startswith("cuda"))

    env = build_vec_env(
        output_dir / "monitor",
        args.seed,
        training=False,
        normalize_reward=False,
        gamma=0.99,
        vecnormalize_path=args.vecnormalize,
        step_sim_time=args.step_sim_time,
    )
    model = TD3.load(str(args.model), env=env, device=args.device)
    model.action_noise = None

    outcomes = Counter()
    records = []
    try:
        with result_path.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            observation = env.reset()
            for episode in range(1, args.episodes + 1):
                total_reward = 0.0
                linear_speeds = []
                angular_speeds = []
                stationary_steps = 0
                turning_steps = 0
                spin_steps = 0
                info = {}

                for step in range(1, args.max_steps + 1):
                    action, _ = model.predict(observation, deterministic=True)
                    action_vector = np.asarray(action[0], dtype=np.float32)
                    linear, angular = physical_action(action_vector)
                    linear_speeds.append(linear)
                    angular_speeds.append(abs(angular))
                    stationary = abs(linear) < args.stationary_linear_threshold
                    turning = abs(angular) > args.spin_angular_threshold
                    stationary_steps += int(stationary)
                    turning_steps += int(turning)
                    spin_steps += int(stationary and turning)

                    observation, reward, done, infos = env.step(action)
                    total_reward += float(reward[0])
                    info = infos[0]
                    if bool(done[0]):
                        outcome_code = int(info.get("success", 0))
                        outcome = OUTCOME_NAMES.get(outcome_code, f"unknown_{outcome_code}")
                        break
                else:
                    outcome_code = -1
                    outcome = "max_steps"
                    observation = env.reset()

                outcomes[outcome] += 1
                record = {
                    "episode": episode,
                    "outcome_code": outcome_code,
                    "outcome": outcome,
                    "reward": total_reward,
                    "steps": step,
                    "goal_distance": float(info.get("goal_distance", np.nan)),
                    "mean_linear_speed": float(np.mean(linear_speeds)),
                    "mean_abs_angular_speed": float(np.mean(angular_speeds)),
                    "stationary_fraction": stationary_steps / step,
                    "turning_fraction": turning_steps / step,
                    "spin_fraction": spin_steps / step,
                }
                records.append(record)
                writer.writerow(record)
                output.flush()
                print(
                    f"episode={episode:03d} outcome={outcome} reward={total_reward:.2f} "
                    f"steps={step} turn={100.0 * record['turning_fraction']:.1f}% "
                    f"spin={100.0 * record['spin_fraction']:.1f}%",
                    flush=True,
                )

        summary = {
            "episodes": len(records),
            "seed": args.seed,
            "model": str(args.model.resolve()),
            "outcomes": dict(outcomes),
            "success_rate_percent": 100.0 * outcomes["success"] / len(records),
            "mean_reward": float(np.mean([r["reward"] for r in records])),
            "mean_episode_length": float(np.mean([r["steps"] for r in records])),
            "mean_linear_speed": float(np.mean([r["mean_linear_speed"] for r in records])),
            "mean_abs_angular_speed": float(
                np.mean([r["mean_abs_angular_speed"] for r in records])
            ),
            "mean_stationary_fraction": float(
                np.mean([r["stationary_fraction"] for r in records])
            ),
            "mean_turning_fraction": float(
                np.mean([r["turning_fraction"] for r in records])
            ),
            "mean_spin_fraction": float(np.mean([r["spin_fraction"] for r in records])),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        env.close()
        shutdown_ros()


if __name__ == "__main__":
    main()
