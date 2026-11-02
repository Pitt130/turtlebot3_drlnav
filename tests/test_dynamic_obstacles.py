"""Offline contracts for moving obstacles: routes, yielding, scan stacking and weight carry-over."""

import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "isaac"))
sys.path.insert(0, str(ROOT / "sb3"))

from dynamic_obstacles import Crowd  # noqa: E402
from grscene import FREE, OCCUPIED, Occupancy  # noqa: E402
from waypoints import ScanStack  # noqa: E402


def two_rooms():
    """6 x 4 m, 5 cm cells, a wall at x = 3 m with a 1 m door in the middle."""
    grid = np.full((80, 120), FREE, dtype=np.uint8)
    grid[[0, -1], :] = OCCUPIED
    grid[:, [0, -1]] = OCCUPIED
    grid[:30, 60] = OCCUPIED
    grid[50:, 60] = OCCUPIED
    return Occupancy((0.0, 0.0), 0.05, grid)


class CrowdTests(unittest.TestCase):
    def setUp(self):
        self.occ = two_rooms()

    def on_free(self, x, y):
        r, c = self.occ.to_cell(x, y)
        return self.occ.grid[r, c] == FREE

    def test_walks_on_free_cells_at_its_speed(self):
        crowd = Crowd(self.occ, 3, speed=(0.3, 0.3), seed=1)
        crowd.reset(1, robot_xy=(-10.0, -10.0))
        self.assertEqual(crowd.active, 3)
        travelled = np.zeros(3)
        for _ in range(1500):                          # 30 s, several legs, through the door too
            before = crowd.pos[:crowd.active].copy()
            crowd.advance(0.02, robot_xy=(-10.0, -10.0))
            moved = np.hypot(*(crowd.pos[:crowd.active] - before).T)
            self.assertTrue(np.all(moved <= 0.3 * 0.02 + 1e-9))
            travelled += moved
            for x, y in crowd.pos[:crowd.active]:
                self.assertTrue(self.on_free(x, y), (x, y))
        self.assertTrue(np.all(travelled > 8.5), travelled)     # 9 m at 0.3 m/s, nothing in the way

    def test_starts_clear_of_robot_and_each_other(self):
        crowd = Crowd(self.occ, 4, radius=0.12, start_clearance=1.0, seed=0)
        for episode in range(1, 20):
            crowd.reset(episode, robot_xy=(1.5, 2.0))
            pos = crowd.pos[:crowd.active]
            self.assertTrue(np.all(np.hypot(pos[:, 0] - 1.5, pos[:, 1] - 2.0) >= 1.0))
            for i in range(len(pos)):
                for j in range(i):
                    self.assertGreaterEqual(math.dist(pos[i], pos[j]), 0.44)

    def test_episode_depends_only_on_seed_and_episode(self):
        a, b = Crowd(self.occ, 3, min_count=0, seed=7), Crowd(self.occ, 3, min_count=0, seed=7)
        a.reset(5, (1.0, 1.0))
        for _ in range(200):
            a.advance(0.02, (1.0, 1.0))
        a.reset(9, (1.0, 1.0))
        b.reset(9, (1.0, 1.0))
        self.assertEqual(a.active, b.active)
        np.testing.assert_allclose(a.pos[:a.active], b.pos[:b.active])
        counts = set()
        for episode in range(40):
            a.reset(episode, (1.0, 1.0))
            counts.add(a.active)
        self.assertEqual(counts, {0, 1, 2, 3})

    def test_waits_for_the_robot_then_gives_way(self):
        crowd = Crowd(self.occ, 1, speed=(0.5, 0.5), yield_dist=0.45, give_way=2.0, seed=0)
        crowd.reset(1, (-10.0, -10.0))
        crowd.pos[0] = (1.0, 2.0)
        crowd.routes[0] = [(5.0, 2.0)]
        robot = (1.6, 2.0)                             # standing on the route
        for _ in range(20):
            crowd.advance(0.02, robot)
        self.assertLess(math.dist(crowd.pos[0], robot), 0.46)
        self.assertGreaterEqual(math.dist(crowd.pos[0], robot), 0.45 - 1e-9)
        self.assertTrue(np.allclose(crowd.vel[0], 0.0))
        waiting_at, left = crowd.pos[0].copy(), 0.0
        for _ in range(400):                           # 8 s: a new leg after 2 s of waiting
            crowd.advance(0.02, robot)
            self.assertGreaterEqual(math.dist(crowd.pos[0], robot), 0.45 - 1e-9)
            left = max(left, math.dist(crowd.pos[0], waiting_at))
        self.assertGreater(left, 0.5)

    def test_a_non_yielding_obstacle_walks_on(self):
        crowd = Crowd(self.occ, 1, speed=(0.5, 0.5), yield_prob=0.0, seed=0)
        crowd.reset(1, (-10.0, -10.0))
        self.assertFalse(crowd.yields[0])
        crowd.pos[0] = (1.0, 2.0)
        crowd.routes[0] = [(5.0, 2.0)]
        robot = (1.6, 2.0)
        for _ in range(40):                            # 0.8 s at 0.5 m/s: through the robot's position
            crowd.advance(0.02, robot)
        np.testing.assert_allclose(crowd.pos[0], (1.4, 2.0))

    def test_yield_draw_keeps_starts_and_mixes(self):
        always, mixed = Crowd(self.occ, 3, seed=4), Crowd(self.occ, 3, yield_prob=0.5, seed=4)
        drawn = []
        for episode in range(30):
            always.reset(episode, (1.0, 1.0))
            mixed.reset(episode, (1.0, 1.0))
            np.testing.assert_allclose(always.pos, mixed.pos)
            self.assertTrue(always.yields.all())
            drawn.extend(mixed.yields)
        self.assertTrue(0.3 < np.mean(drawn) < 0.7, np.mean(drawn))

    def test_a_pedestrian_steers_around_a_standing_robot(self):
        crowd = Crowd(self.occ, 1, speed=(0.4, 0.4), yield_prob=0.0, avoid=0.5, seed=0)
        crowd.reset(1, (-10.0, -10.0))
        crowd.pos[0] = (1.5, 0.3)                     # across the 3 x 4 m left room
        crowd.routes[0] = [(1.5, 3.7)]
        robot, closest, passed = (1.5, 2.0), math.inf, False
        for _ in range(500):                           # 10 s: past the robot without stopping
            crowd.advance(0.02, robot)
            closest = min(closest, math.dist(crowd.pos[0], robot))
            passed |= crowd.pos[0][1] > 2.5
            self.assertTrue(self.on_free(*crowd.pos[0]))
        self.assertTrue(passed)
        self.assertGreater(closest, 0.45)

    def test_a_pedestrian_dodges_a_robot_driving_at_it(self):
        crowd = Crowd(self.occ, 1, speed=(0.3, 0.3), yield_prob=0.0, avoid=0.5, seed=0)
        crowd.reset(1, (-10.0, -10.0))
        crowd.pos[0] = (1.5, 0.3)
        crowd.routes[0] = [(1.5, 3.7)]
        closest = math.inf
        for t in range(300):                           # robot at 0.2 m/s head-on along the same line
            robot = (1.5, 3.0 - 0.2 * 0.02 * t)
            crowd.advance(0.02, robot)
            closest = min(closest, math.dist(crowd.pos[0], robot))
        self.assertGreater(closest, 0.4)

    def test_moves_away_from_a_close_robot(self):
        crowd = Crowd(self.occ, 1, speed=(0.4, 0.4), seed=0)
        crowd.reset(1, (-10.0, -10.0))
        crowd.pos[0] = (2.0, 2.0)
        crowd.routes[0] = [(1.0, 2.0)]
        crowd.advance(0.1, (2.3, 2.0))                 # 0.3 m behind it, but it walks away
        np.testing.assert_allclose(crowd.pos[0], (1.96, 2.0))


class ScanStackTests(unittest.TestCase):
    def obs(self, value):
        return np.concatenate([np.full(40, value), [0.5, 0.25, 0.1, -0.2]]).astype(np.float32)

    def test_layout_keeps_current_observation_last(self):
        stack = ScanStack(3, 2, 40)
        out = stack.reset(self.obs(0.0))
        self.assertEqual(out.shape, (124,))
        np.testing.assert_array_equal(out[:80], 0.0)
        for t in range(1, 6):
            out = stack.push(self.obs(float(t)))
        # t = 5: scans of t-4 and t-2, then the full observation of t.
        np.testing.assert_array_equal(out[:40], 1.0)
        np.testing.assert_array_equal(out[40:80], 3.0)
        np.testing.assert_array_equal(out[80:], self.obs(5.0))
        self.assertAlmostEqual(float(out[-3]), 0.25)

    def test_single_frame_is_unchanged(self):
        stack = ScanStack(1, 2, 40)
        np.testing.assert_array_equal(stack.reset(self.obs(1.0)), self.obs(1.0))
        np.testing.assert_array_equal(stack.push(self.obs(2.0)), self.obs(2.0))


class InitFromTests(unittest.TestCase):
    def test_padded_policy_acts_like_the_original(self):
        try:
            import gymnasium
            from stable_baselines3 import TD3
            with patch.dict(os.environ, {"DRLNAV_SIM_BACKEND": "isaac"}):   # train imports the environment
                from train import copy_padded_weights
            from tomas_policy import ConfigurableTD3Policy
        except ImportError as error:                     # needs the SB3 venv and a sourced ROS workspace
            self.skipTest(str(error))

        class Dummy(gymnasium.Env):
            def __init__(self, size):
                self.observation_space = gymnasium.spaces.Box(-1.0, 1.0, (size,), np.float32)
                self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (2,), np.float32)

        kwargs = dict(policy_kwargs={"net_arch": [64, 64], "critic_architecture": "tomas", "xavier_init": True},
                      device="cpu", seed=0)
        old = TD3(ConfigurableTD3Policy, Dummy(44), **kwargs)
        new = TD3(ConfigurableTD3Policy, Dummy(124), **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            old.save(f"{directory}/old.zip")
            copy_padded_weights(new, Path(f"{directory}/old.zip"))
        rng = np.random.default_rng(0)
        for _ in range(5):
            current = rng.uniform(-1, 1, 44).astype(np.float32)
            stacked = np.concatenate([rng.uniform(-1, 1, 80).astype(np.float32), current])
            np.testing.assert_allclose(new.predict(stacked, deterministic=True)[0],
                                       old.predict(current, deterministic=True)[0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
