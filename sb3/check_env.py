#!/usr/bin/env python3
"""Run the Stable-Baselines3 checker against the ROS Gymnasium environment."""

import argparse

import rclpy
from stable_baselines3.common.env_checker import check_env

from turtlebot3_drl.drl_environment.drl_environment_gymnasium import make_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = make_env()
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    try:
        env.reset(seed=args.seed)
        check_env(env, warn=True, skip_render_check=True)
        print("SB3 check_env passed")
    finally:
        env.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
