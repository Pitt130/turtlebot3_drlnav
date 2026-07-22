#!/usr/bin/env python3

import importlib
from gymnasium.utils.env_checker import check_env

MODULE = "turtlebot3_drl.drl_environment.drl_environment_gymnasium"

module = importlib.import_module(MODULE)
env = module.make_env()

print("Checking Gymnasium environment...")
check_env(env, skip_render_check=True)

if hasattr(env, "close"):
    env.close()

print("GYMNASIUM CHECK_ENV PASSED")
