"""Offline check of WaypointEnv on a scene grid with a fake env that follows waypoints perfectly."""
import glob
import math
import sys

import gymnasium
import numpy as np

sys.path.insert(0, "isaac")
from grscene import Occupancy, StartGoalSampler  # noqa: E402
from waypoint_env import WaypointEnv  # noqa: E402

scene = sys.argv[1]
occ = Occupancy.load(sorted(glob.glob(scene + "/occupancy_v4_*.npz"))[0])
sampler = StartGoalSampler(occ, seed=3)


class Fake(gymnasium.Env):
    observation_space = gymnasium.spaces.Box(-1, 1, (44,), np.float32)
    action_space = gymnasium.spaces.Box(-1, 1, (2,), np.float32)

    def reset(self, **kwargs):
        self.robot_x, self.robot_y, self.robot_heading = sampler.sample_start()
        self.goal_x, self.goal_y = sampler.sample_goal((self.robot_x, self.robot_y))
        self.obstacle_distance, self.n = 1.0, 0
        return np.zeros(44, np.float32), {}

    def step(self, action):
        wx, wy, _ = W.planner.waypoint(self.robot_x, self.robot_y)
        d = math.hypot(wx - self.robot_x, wy - self.robot_y)
        self.robot_heading = math.atan2(wy - self.robot_y, wx - self.robot_x)
        step = min(0.022, d)
        self.robot_x += step * math.cos(self.robot_heading)
        self.robot_y += step * math.sin(self.robot_heading)
        self.n += 1
        done = math.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y) < 0.2
        return (np.zeros(44, np.float32), -999.0, done, self.n >= 500,
                {"success": 1 if done else (4 if self.n >= 500 else 0), "step_sim_dt": 0.1})


W = WaypointEnv(Fake(), scene, lookahead=0.75, max_goal_distance=5.94)
for ep in range(4):
    obs, info = W.reset()
    total, n = 0.0, 0
    while True:
        obs, r, term, trunc, step_info = W.step(np.array([1.0, 0.0], np.float32))
        total += r
        n += 1
        if n == 1:
            print(f"   first step: reward {r:+.2f}, waypoint {obs[40] * 5.94:.2f} m at {obs[41] * 180:+.0f} deg")
        if term or trunc:
            break
    print(f"ep {ep}: path {info['path_length']:.2f} m, {n} steps, outcome {step_info['success']}, "
          f"return {total:.0f}, path_blocked {step_info['path_blocked']}")
