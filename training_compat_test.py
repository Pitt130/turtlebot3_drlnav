#!/usr/bin/env python3

import importlib
import numpy as np

MODULE = "turtlebot3_drl.drl_environment.drl_environment_gymnasium"

module = importlib.import_module(MODULE)
env = module.make_env()

print("========== Training Compatibility Test ==========")
print("action_space:", env.action_space)
print("observation_space:", env.observation_space)

num_steps = 1000
episode = 1
episode_reward = 0.0
episode_steps = 0

obs, info = env.reset()

print(f"initial obs shape={obs.shape}, dtype={obs.dtype}, valid={env.observation_space.contains(obs)}")
print("initial info:", info)

for global_step in range(1, num_steps + 1):
    action = env.action_space.sample()

    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    assert env.action_space.contains(action), "action not in action_space"
    assert env.observation_space.contains(next_obs), "next_obs not in observation_space"
    assert isinstance(terminated, bool), f"terminated type error: {type(terminated)}"
    assert isinstance(truncated, bool), f"truncated type error: {type(truncated)}"
    assert isinstance(info, dict), f"info type error: {type(info)}"
    assert np.isscalar(reward), f"reward type error: {type(reward)}"

    episode_reward += float(reward)
    episode_steps += 1

    if global_step % 50 == 0 or done:
        print(
            f"global_step={global_step}, "
            f"episode={episode}, "
            f"episode_steps={episode_steps}, "
            f"reward={float(reward):.3f}, "
            f"episode_reward={episode_reward:.3f}, "
            f"terminated={terminated}, "
            f"truncated={truncated}, "
            f"success={info.get('success')}, "
            f"goal_distance={info.get('goal_distance')}, "
            f"obstacle_distance={info.get('obstacle_distance')}, "
            f"time_sec={info.get('time_sec')}"
        )

    obs = next_obs

    if done:
        print(
            f"Episode {episode} ended: "
            f"steps={episode_steps}, "
            f"reward={episode_reward:.3f}, "
            f"terminated={terminated}, "
            f"truncated={truncated}, "
            f"info={info}"
        )

        obs, info = env.reset()
        print(f"reset after episode {episode}: info={info}")

        episode += 1
        episode_reward = 0.0
        episode_steps = 0

if hasattr(env, "close"):
    env.close()

print("TRAINING COMPATIBILITY TEST PASSED")
