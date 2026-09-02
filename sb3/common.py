"""Shared environment and reproducibility helpers for SB3 experiments."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import rclpy
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from turtlebot3_drl.drl_environment.drl_environment_gymnasium import make_env


def seed_everything(seed: int, use_cuda: bool = True) -> None:
    """Seed Python, NumPy, PyTorch, SB3, and CUDA where available."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_random_seed(seed, using_cuda=use_cuda and torch.cuda.is_available())


def build_vec_env(
    log_dir: Path,
    seed: int,
    *,
    training: bool,
    normalize_reward: bool,
    gamma: float,
    vecnormalize_path: Path | None = None,
) -> VecNormalize:
    """Create the single ROS environment used by SB3 train and evaluation."""
    log_dir.mkdir(parents=True, exist_ok=True)

    def make_monitored_env():
        env = make_env()
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return Monitor(
            env,
            filename=str(log_dir / "monitor.csv"),
            info_keywords=("success",),
        )

    base_env = DummyVecEnv([make_monitored_env])
    base_env.seed(seed)

    if vecnormalize_path is not None:
        env = VecNormalize.load(str(vecnormalize_path), base_env)
    else:
        env = VecNormalize(
            base_env,
            training=training,
            norm_obs=False,
            norm_reward=normalize_reward,
            clip_reward=10.0,
            gamma=gamma,
        )

    env.training = training
    env.norm_reward = normalize_reward
    env.seed(seed)
    return env


def shutdown_ros() -> None:
    """Shut down rclpy after the Gymnasium environment is closed."""
    if rclpy.ok():
        rclpy.shutdown()
