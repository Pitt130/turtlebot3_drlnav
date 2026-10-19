"""ROS installation required; no simulator, robot commands or reset calls."""

import math
import os
from pathlib import Path
import types
import unittest

import numpy as np
import rclpy

from turtlebot3_drl.common import utilities as util
from turtlebot3_drl.drl_environment.drl_environment_gymnasium import DRLGymnasiumEnvironment, NUM_SCAN_SAMPLES


class RosMigrationTests(unittest.TestCase):
    def test_isaac_constructs_without_gazebo_asset_path(self):
        self.assertEqual(os.environ.get("DRLNAV_SIM_BACKEND"), "isaac",
                         "Run this suite with DRLNAV_SIM_BACKEND=isaac")
        self.assertNotIn("DRLNAV_BASE_PATH", os.environ)
        rclpy.init()
        env = DRLGymnasiumEnvironment()
        try:
            self.assertFalse(env.reset_rewinds_clock)
            self.assertEqual(env.observation_space.shape, (44,))
            self.assertEqual(env.action_space.shape, (2,))
            self.assertEqual(util.get_simulation_speed(1), 1.0)
        finally:
            # Avoid even a zero-velocity publication to any external robot.
            env._executor.remove_node(env)
            env.destroy_node()
            rclpy.shutdown()

    def scan_message(self, ranges):
        stamp = types.SimpleNamespace(sec=1, nanosec=0)
        return types.SimpleNamespace(ranges=ranges, angle_min=-math.pi, angle_increment=math.pi / 180,
                                     range_min=0.12, range_max=3.5, header=types.SimpleNamespace(stamp=stamp))

    def test_dense_scan_callback_normalizes_full_circle(self):
        ranges = np.full(360, 3.5)
        ranges[180] = 0.15
        env = types.SimpleNamespace(_awaiting_rewind=False)
        DRLGymnasiumEnvironment.scan_callback(env, self.scan_message(ranges))
        self.assertEqual(len(env.scan_ranges), NUM_SCAN_SAMPLES)
        self.assertAlmostEqual(env.obstacle_distance, 0.15)
        self.assertAlmostEqual(env.scan_ranges[0], 0.15 / 3.5)
        self.assertTrue(np.isfinite(env.scan_ranges).all())

    def test_collision_distance_uses_every_ray(self):
        # 14 degrees lies between policy directions 1 (9.2) and 2 (18.5).
        ranges = np.full(360, 2.0)
        ranges[180 + 14] = 0.125
        env = types.SimpleNamespace(_awaiting_rewind=False)
        DRLGymnasiumEnvironment.scan_callback(env, self.scan_message(ranges))
        self.assertAlmostEqual(env.obstacle_distance, 0.125)
        np.testing.assert_allclose(env.scan_ranges, np.full(NUM_SCAN_SAMPLES, 2.0 / 3.5), rtol=1e-6)

    def test_scan_callback_uses_message_range_limits(self):
        # A driver reporting missing returns as 0.0 must not look like a collision.
        ranges = np.full(360, 2.0)
        ranges[:180] = 0.0
        env = types.SimpleNamespace(_awaiting_rewind=False)
        DRLGymnasiumEnvironment.scan_callback(env, self.scan_message(ranges))
        self.assertAlmostEqual(env.obstacle_distance, 2.0)

    def test_shared_arena_bounds_reach_environment_settings(self):
        from turtlebot3_drl.common.settings import ARENA_LENGTH, ARENA_WIDTH
        from turtlebot3_drl.common.arena_config import load_arena_config
        arena = load_arena_config(os.environ["DRLNAV_ARENA_CONFIG"])
        self.assertEqual((ARENA_LENGTH, ARENA_WIDTH), (arena.length, arena.width))


if __name__ == "__main__":
    unittest.main(verbosity=2)
