"""Offline contracts: configuration, geometry, full-circle LiDAR and control."""

import json
import math
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from turtlebot3_drl.common import simulation_config as config
from turtlebot3_drl.common.arena_config import load_arena_config
from turtlebot3_drl.common.lidar_adapter import policy_scan
from tools.export_arena import export_arena


class ConfigurationTests(unittest.TestCase):
    def test_isaac_without_gazebo_files(self):
        with patch.dict(os.environ, {"DRLNAV_SIM_BACKEND": "isaac"}, clear=True):
            self.assertEqual(config.scan_count(), 40)
            self.assertEqual(config.simulation_speed(9), 1.0)

    def test_explicit_parameters(self):
        values = {"DRLNAV_SCAN_SAMPLES": "40", "DRLNAV_SIM_SPEED": "2.5", "DRLNAV_STAGE": "9"}
        with patch.dict(os.environ, values, clear=True):
            self.assertEqual(config.scan_count(), 40)
            self.assertEqual(config.simulation_speed(9), 2.5)
            self.assertEqual(config.current_stage("/nonexistent"), 9)

    def test_stage_file_and_unknown_stage(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / "stage"
            # No silent default: it would mislabel sessions, weights and logs.
            with self.assertRaises(FileNotFoundError):
                config.current_stage(path)
            path.write_text("4\n")
            self.assertEqual(config.current_stage(path), 4)

    def test_gazebo_launch_stage_conflict_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage"
            path.write_text("9\n")
            with patch.dict(os.environ, {"DRLNAV_STAGE": "9"}, clear=True):
                self.assertEqual(config.current_stage(path), 9)
            with patch.dict(os.environ, {"DRLNAV_STAGE": "1"}, clear=True):
                with self.assertRaises(ValueError):
                    config.current_stage(path)
            with patch.dict(os.environ, {"DRLNAV_STAGE": "1", "DRLNAV_SIM_BACKEND": "isaac"}, clear=True):
                self.assertEqual(config.current_stage(path), 1)

    def test_invalid_configuration(self):
        for key, value, function in [
            ("DRLNAV_SCAN_SAMPLES", "0", config.scan_count),
            ("DRLNAV_SIM_SPEED", "nan", lambda: config.simulation_speed(1)),
            ("DRLNAV_STAGE", "-1", config.current_stage),
            ("DRLNAV_SIM_BACKEND", "unknown", config.simulation_backend),
        ]:
            with self.subTest(key=key), patch.dict(os.environ, {key: value}, clear=True):
                with self.assertRaises(ValueError):
                    function()

    def test_gazebo_keeps_model_scan_count(self):
        root = Path(__file__).resolve().parents[1]
        with patch.dict(os.environ, {"DRLNAV_BASE_PATH": str(root)}, clear=True):
            self.assertEqual(config.scan_count(), 40)


class LidarTests(unittest.TestCase):
    def test_legacy_angles_and_values_are_unchanged(self):
        ranges = np.linspace(0.2, 3.0, 40)
        result = policy_scan(ranges, 0.0, 6.28 / 39, 40, 3.5)
        np.testing.assert_array_equal(result, ranges.astype(np.float32))

    def test_dense_scan_keeps_obstacle_beyond_first_40_rays(self):
        ranges = np.full(360, 3.5)
        ranges[182:188] = 0.1  # around policy direction 20 at 184.5 degrees
        result = policy_scan(ranges, 0.0, math.pi / 180, 40, 3.5)
        self.assertEqual(result.shape, (40,))
        self.assertAlmostEqual(float(result.min()), 0.1)
        self.assertLess(float(result[20]), 0.2)
        self.assertEqual(result.dtype, np.float32)

    def test_dense_scan_matches_gazebo_point_sampling(self):
        # Each policy input is the single ray nearest its legacy direction.
        legacy = np.linspace(0.0, 6.28, 40)
        for angle_min, increment in [(-math.pi, 2 * math.pi / 1440), (math.pi, -2 * math.pi / 1440)]:
            with self.subTest(angle_min=angle_min, increment=increment):
                ranges = np.linspace(0.2, 3.4, 1440)
                result = policy_scan(ranges, angle_min, increment, 40, 3.5)
                nearest = np.rint(np.mod((legacy - angle_min) / increment, 1440)).astype(int) % 1440
                np.testing.assert_allclose(result, ranges[nearest], rtol=1e-6)

    def test_obstacle_between_directions_is_not_widened(self):
        # Gazebo's 40 rays miss this; sector minima made it look 4.5 degrees wide.
        ranges = np.full(360, 3.5)
        ranges[14] = 0.3  # between policy directions 1 (9.2 deg) and 2 (18.5 deg)
        result = policy_scan(ranges, 0.0, math.pi / 180, 40, 3.5)
        np.testing.assert_array_equal(result, np.full(40, 3.5, dtype=np.float32))

    def test_shifted_origin_preserves_forward_direction(self):
        ranges = np.full(360, 3.5)
        ranges[180] = 0.2  # angle zero in a scan starting at -pi
        result = policy_scan(ranges, -math.pi, math.pi / 180, 40, 3.5)
        self.assertAlmostEqual(float(result[0]), 0.2)
        self.assertEqual(result[-1], result[0])

    def test_clockwise_scan_preserves_forward_direction(self):
        ranges = np.full(360, 3.5)
        ranges[180] = 0.2
        result = policy_scan(ranges, math.pi, -math.pi / 180, 40, 3.5)
        self.assertAlmostEqual(float(result[0]), 0.2)

    def test_nonfinite_ranges_without_limits(self):
        ranges = np.full(40, 2.0)
        ranges[1:4] = [np.nan, np.inf, -np.inf]
        result = policy_scan(ranges, 0.0, 6.28 / 39, 40, 3.5)
        expected = np.full(40, 2.0, dtype=np.float32)
        expected[1:4] = [3.5, 3.5, 0.0]  # REP-117: -inf is closer than range_min
        np.testing.assert_array_equal(result, expected)

    def test_rep117_range_semantics(self):
        ranges = np.full(40, 2.0)
        ranges[1:8] = [np.nan, np.inf, -np.inf, 0.0, 5.0, 0.12, 0.05]
        result = policy_scan(ranges, 0.0, 6.28 / 39, 40, 3.5, range_min=0.12, range_max=3.5)
        expected = np.full(40, 2.0)
        # A 0.0 "no return" and other out-of-range values are discarded.
        expected[1:8] = [3.5, 3.5, 0.0, 3.5, 3.5, 0.12, 3.5]
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_invalid_range_limits_are_rejected(self):
        for low, high in [(0.12, 0.0), (float("nan"), 3.5), (-0.1, 3.5), (0.12, float("nan"))]:
            with self.subTest(low=low, high=high), self.assertRaises(ValueError):
                policy_scan(np.full(40, 2.0), 0.0, 6.28 / 39, 40, 3.5, range_min=low, range_max=high)

    def test_partial_scan_is_rejected(self):
        with self.assertRaises(ValueError):
            policy_scan(np.ones(180), 0.0, math.pi / 180, 40, 3.5)

    def test_sparse_scan_is_rejected(self):
        with self.assertRaises(ValueError):
            policy_scan(np.ones(10), 0.0, 2 * math.pi / 10, 40, 3.5)


class ArenaTests(unittest.TestCase):
    def write_arena(self, path):
        path.write_text(json.dumps({"schema_version": 1, "arena_length": 4.2, "arena_width": 4.2,
                                   "obstacles": [{"center": [0, 0], "size": [2, 0.2], "yaw": math.pi / 2}]}))

    def test_rotated_box_goal_exclusion_and_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arena.json"
            self.write_arena(path)
            arena = load_arena_config(path)
            self.assertFalse(arena.goal_is_valid(0.0, 0.8))
            self.assertTrue(arena.goal_is_valid(0.8, 0.0))
            self.assertFalse(arena.goal_is_valid(3.0, 0.0))
            self.assertFalse(arena.goal_is_valid(float("nan"), 0.0))

    def test_nonfinite_size_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arena.json"
            self.write_arena(path)
            data = json.loads(path.read_text())
            data["obstacles"][0]["size"][0] = float("inf")
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_arena_config(path)

    def test_export_composes_model_link_and_collision_poses(self):
        sdf = '''<sdf><model name="walls"><pose>1 2 0 0 0 1.5707963267948966</pose>
                 <link name="wall"><pose>1 0 0 0 0 0</pose>
                 <collision name="box"><pose>0.5 0 0 0 0 0</pose>
                 <geometry><box><size>1 0.2 0.5</size></box></geometry>
                 </collision></link></model></sdf>'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.sdf"
            path.write_text(sdf)
            data = export_arena(path, 4.2, 4.2)
            box = data["obstacles"][0]
            np.testing.assert_allclose(box["center"], [1, 3.5], atol=1e-8)
            self.assertAlmostEqual(box["yaw"], math.pi / 2)

    def test_export_rejects_unsupported_collision(self):
        sdf = '''<sdf><model name="walls"><link name="wall"><collision name="ball">
                 <geometry><sphere><radius>1</radius></sphere></geometry>
                 </collision></link></model></sdf>'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.sdf"
            path.write_text(sdf)
            with self.assertRaises(ValueError):
                export_arena(path, 4.2, 4.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
