#!/usr/bin/env python3
"""Minimal Gymnasium smoke test for the TurtleBot3 DRL environment."""

import argparse
import importlib
import sys


def _shape(observation):
    return getattr(observation, "shape", None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module",
        default="drl_environment_gymnasium",
        help="Python module containing make_env() or the environment class.",
    )
    parser.add_argument("--factory", default="make_env", help="Environment factory function name.")
    parser.add_argument("--steps", type=int, default=10, help="Number of environment steps to run.")
    args = parser.parse_args()

    module = importlib.import_module(args.module)
    make_env = getattr(module, args.factory)
    env = make_env()

    try:
        observation, info = env.reset()
        print(f"reset: observation_shape={_shape(observation)}, info={info}")

        for step in range(args.steps):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            print(
                f"step={step + 1}: observation_shape={_shape(observation)}, "
                f"reward={reward}, terminated={terminated}, truncated={truncated}, info={info}"
            )

            if terminated or truncated:
                observation, info = env.reset()
                print(f"reset: observation_shape={_shape(observation)}, info={info}")
    finally:
        env.close()
        if "rclpy" in sys.modules:
            rclpy = sys.modules["rclpy"]
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
