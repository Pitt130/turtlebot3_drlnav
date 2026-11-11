"""Train the local controller of hierarchical navigation: the policy sees grid-path waypoints.

WaypointEnv wraps the Gymnasium environment exactly as check_grscene.py --waypoints drives it at
evaluation time: the two goal entries of the observation point at a waypoint `lookahead` metres
down the occupancy-grid path, with the below-lidar margin and stall recovery of WaypointPlanner.
Success, collision, timeout and tumble are still judged by the environment against the real goal.

With path_reward the step reward follows reward B, but measured along the path: progress is the
decrease of the grid path length to the goal (straight-line distance would punish every detour
around a wall) and the heading term uses the waypoint direction. With slow_dist the full-speed target
of the speed term drops linearly inside slow_dist of the nearest obstacle, to slow_min_frac of
v_max at the collision distance, and going faster than that target is penalized too. With scan_stack > 1
older scans go in front of the observation (waypoints.ScanStack), so moving obstacles can be told apart.
"""
import glob
import math
from pathlib import Path
import sys

import gymnasium
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grscene import Occupancy  # noqa: E402
from waypoints import ScanStack, WaypointPlanner, make_planner  # noqa: E402

SUCCESS, COLLISION_WALL, COLLISION_OBSTACLE, TIMEOUT, TUMBLE = 1, 2, 3, 4, 5
TERMINAL = {SUCCESS: 2500.0, COLLISION_WALL: -2000.0, COLLISION_OBSTACLE: -2000.0, TIMEOUT: -500.0, TUMBLE: -1000.0}


class WaypointEnv(gymnasium.Wrapper):
    def __init__(self, env, scene_dir, lookahead=0.75, inflate=0.18, low_inflate=0.30, stall_recovery=True,
                 path_reward=True, speed_linear_max=0.22, speed_angular_max=2.0, max_goal_distance=None,
                 obstacle_penalty_dist=0.15, obstacle_penalty=10.0, progress_gain=50.0, plan_map="occupancy_v4",
                 plan_cost=0.0, plan_cost_dist=0.25, plan_fallback_map=None,
                 slow_dist=0.0, slow_min_frac=0.3, collision_dist=0.13, scan_stack=1, scan_stack_skip=2):
        super().__init__(env)
        self.planner = make_planner(scene_dir, plan_map, plan_fallback_map, inflate=inflate, lookahead=lookahead,
                                    low_inflate=low_inflate, cost_weight=plan_cost, cost_dist=plan_cost_dist)
        self.stall_recovery, self.path_reward = stall_recovery, path_reward
        self.v_max, self.w_max = speed_linear_max, speed_angular_max
        if max_goal_distance is None:
            from turtlebot3_drl.drl_environment.drl_environment_gymnasium import MAX_GOAL_DISTANCE
            max_goal_distance = MAX_GOAL_DISTANCE
        self.max_goal_distance = max_goal_distance
        self.obstacle_penalty_dist, self.progress_gain = obstacle_penalty_dist, progress_gain
        self.obstacle_penalty = obstacle_penalty
        self.slow_dist, self.slow_min_frac, self.collision_dist = slow_dist, slow_min_frac, collision_dist
        self.prev_len = None
        low, high = self.env.observation_space.low, self.env.observation_space.high
        self.stack = ScanStack(scan_stack, scan_stack_skip, scan_size=len(low) - 4)
        if scan_stack > 1:
            n = len(low) - 4
            self.observation_space = gymnasium.spaces.Box(
                low=np.concatenate([low[:n]] * (scan_stack - 1) + [low]),
                high=np.concatenate([high[:n]] * (scan_stack - 1) + [high]), dtype=np.float32)

    def _robot(self):
        e = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        return e

    def _waypoint_obs(self, obs):
        e = self._robot()
        if self.stall_recovery:
            wx, wy, _, _ = self.planner.step(e.robot_x, e.robot_y)
        else:
            wx, wy, _ = self.planner.waypoint(e.robot_x, e.robot_y)
        return self.planner.observation_with(obs, e.robot_x, e.robot_y, e.robot_heading, wx, wy, self.max_goal_distance)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        e = self._robot()
        self.planner.set_goal(e.goal_x, e.goal_y)
        self.planner.reset_progress()
        self.prev_len = self.planner.path_length(e.robot_x, e.robot_y)
        info = dict(info, path_length=self.prev_len)
        return self.stack.reset(self._waypoint_obs(obs)), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        e = self._robot()
        obs = self._waypoint_obs(obs)
        info = dict(info, path_blocked=self.planner.stalls, env_reward=reward)
        if self.path_reward:
            length = self.planner.path_length(e.robot_x, e.robot_y)
            progress = 0.0
            if math.isfinite(length) and self.prev_len is not None and math.isfinite(self.prev_len):
                progress = float(np.clip(self.prev_len - length, -0.2, 0.2))   # nearest-cell jumps are not progress
            self.prev_len = length
            a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
            v = (a[0] + 1) / 2 * self.v_max
            w = a[1] * self.w_max
            angle = float(obs[-3]) * math.pi                    # waypoint bearing, as the policy sees it
            v_target = self.v_max
            if self.slow_dist > self.collision_dist and e.obstacle_distance < self.slow_dist:
                frac = (e.obstacle_distance - self.collision_dist) / (self.slow_dist - self.collision_dist)
                v_target = self.v_max * float(np.clip(frac, self.slow_min_frac, 1.0))
            reward = (-abs(angle) - w * w - ((v_target - v) * 10) ** 2 + self.progress_gain * progress - 1.0
                      + (-self.obstacle_penalty if e.obstacle_distance < self.obstacle_penalty_dist else 0.0))
            if terminated or truncated:
                reward += TERMINAL.get(int(info.get("success", 0)), 0.0)
        return self.stack.push(obs), float(reward), terminated, truncated, info
