"""goal_poses.next_goal reproduces DRLGazebo.generate_goal_pose exactly."""

import os
import random
import unittest
from unittest.mock import patch

from turtlebot3_drl.common import goal_poses


def legacy_generate_goal_pose(stage, goal_x, goal_y, rng):
    """Verbatim loop of DRLGazebo.generate_goal_pose before the refactor (5bc0933)."""
    prev_x, prev_y = goal_x, goal_y
    tries = 0
    while ((abs(prev_x - goal_x) + abs(prev_y - goal_y)) < 2):
        if stage == 11:
            goal_pose_list = [[0.0, 0.0], [0.0, 6.5], [5.0, 5.5], [-2.5, -6.0], [3.0, -4.0], [6.0, -1.0]]
            index = rng.randrange(0, len(goal_pose_list))
            goal_x = float(goal_pose_list[index][0])
            goal_y = float(goal_pose_list[index][1])
        elif stage == 8 or stage == 9 or stage == 12:
            goal_pose_list = [[2.0, 2.0], [2.0, 1.5], [2.0, -0.5], [2.0, -1.0], [2.0, -2.0], [1.3, 1.0],
                              [1.0, 0.3], [1.0, -2.0], [0.3, -1.0], [0.0, 2.0], [0.0, -1.0], [-1.0, 1.0],
                              [-1.0, -1.2], [-2.0, 1.0], [-2.2, 0.0], [-2.0, -2.2], [-2.4, 2.4]]
            index = rng.randrange(0, len(goal_pose_list))
            goal_x = float(goal_pose_list[index][0])
            goal_y = float(goal_pose_list[index][1])
        elif stage not in [4, 5, 7]:
            goal_x = rng.randrange(-15, 16) / 10.0
            goal_y = rng.randrange(-15, 16) / 10.0
        else:
            goal_pose_list = [[1.0, 0.0], [2.0, -1.5], [0.0, -2.0], [2.0, 2.0], [0.8, 2.0],
                              [-1.9, 1.9], [-1.9, 0.2], [-1.9, -0.5], [-2.0, -2.0], [-0.5, -1.0],
                              [1.5, -1.0], [-0.5, 1.0], [-1.0, -2.0], [1.8, -0.2], [1.0, -1.9]]
            index = rng.randrange(0, len(goal_pose_list))
            goal_x = float(goal_pose_list[index][0])
            goal_y = float(goal_pose_list[index][1])
        tries += 1
        if tries > 100:
            break
    return goal_x, goal_y


class GoalPoseTests(unittest.TestCase):
    def test_same_sequence_as_drl_gazebo(self):
        for stage in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12):
            for seed in range(5):
                with self.subTest(stage=stage, seed=seed):
                    old_rng, new_rng = random.Random(seed), random.Random(seed)
                    old = new = goal_poses.INITIAL_GOAL
                    for _ in range(200):
                        old = legacy_generate_goal_pose(stage, *old, old_rng)
                        new = goal_poses.next_goal(stage, new, new_rng)
                        self.assertEqual(old, new)

    def test_goals_stay_two_metres_apart(self):
        rng, goal = random.Random(3), goal_poses.INITIAL_GOAL
        for _ in range(500):
            previous, goal = goal, goal_poses.next_goal(4, goal, rng)
            self.assertGreaterEqual(abs(goal[0] - previous[0]) + abs(goal[1] - previous[1]), 2)
            self.assertIn(list(goal), goal_poses.GOALS_STAGE_4_5_7)

    def test_goal_seed(self):
        with patch.dict(os.environ, {"DRLNAV_GOAL_SEED": "7"}):
            a = [goal_poses.next_goal(9, (0.5, 0.0), goal_poses.goal_rng()) for _ in range(3)]
            rng = goal_poses.goal_rng()
            b = [goal_poses.next_goal(9, (0.5, 0.0), rng) for _ in range(1)]
            self.assertEqual(a[0], b[0])
        with patch.dict(os.environ, {}, clear=True):
            self.assertIs(goal_poses.goal_rng(), random)


if __name__ == "__main__":
    unittest.main(verbosity=2)
