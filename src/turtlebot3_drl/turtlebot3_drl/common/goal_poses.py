"""Stage goal selection shared by the Gazebo goal node and the Isaac task controller.

Mirrors DRLGazebo.generate_goal_pose: draw goals until one is at least 2 m
(Manhattan) from the previous goal, giving up after 100 tries. The random
calls are made in the original order, so a seeded generator reproduces the
Gazebo sequence.
"""
import os
import random

GOALS_STAGE_11 = [[0.0, 0.0], [0.0, 6.5], [5.0, 5.5], [-2.5, -6.0], [3.0, -4.0], [6.0, -1.0]]
GOALS_STAGE_8_9_12 = [[2.0, 2.0], [2.0, 1.5], [2.0, -0.5], [2.0, -1.0], [2.0, -2.0], [1.3, 1.0],
                      [1.0, 0.3], [1.0, -2.0], [0.3, -1.0], [0.0, 2.0], [0.0, -1.0], [-1.0, 1.0],
                      [-1.0, -1.2], [-2.0, 1.0], [-2.2, 0.0], [-2.0, -2.2], [-2.4, 2.4]]
GOALS_STAGE_4_5_7 = [[1.0, 0.0], [2.0, -1.5], [0.0, -2.0], [2.0, 2.0], [0.8, 2.0],
                     [-1.9, 1.9], [-1.9, 0.2], [-1.9, -0.5], [-2.0, -2.0], [-0.5, -1.0],
                     [1.5, -1.0], [-0.5, 1.0], [-1.0, -2.0], [1.8, -0.2], [1.0, -1.9]]
INITIAL_GOAL = (0.5, 0.0)


def goal_rng():
    """DRLNAV_GOAL_SEED pins the sequence; unset keeps the global random module."""
    seed = os.environ.get("DRLNAV_GOAL_SEED")
    return random.Random(int(seed)) if seed not in (None, "") else random


def next_goal(stage, previous, rng=random):
    prev_x, prev_y = previous
    goal_x, goal_y = previous
    tries = 0
    while (abs(prev_x - goal_x) + abs(prev_y - goal_y)) < 2:
        if stage == 11:
            goal_x, goal_y = (float(v) for v in GOALS_STAGE_11[rng.randrange(0, len(GOALS_STAGE_11))])
        elif stage in (8, 9, 12):
            goal_x, goal_y = (float(v) for v in GOALS_STAGE_8_9_12[rng.randrange(0, len(GOALS_STAGE_8_9_12))])
        elif stage not in (4, 5, 7):
            goal_x = rng.randrange(-15, 16) / 10.0
            goal_y = rng.randrange(-15, 16) / 10.0
        else:
            goal_x, goal_y = (float(v) for v in GOALS_STAGE_4_5_7[rng.randrange(0, len(GOALS_STAGE_4_5_7))])
        tries += 1
        if tries > 100:
            print("ERROR: distance between goals is small!")
            break
    return goal_x, goal_y
