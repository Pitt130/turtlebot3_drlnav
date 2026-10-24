"""Reward B (indoor) keeps A's shaping but prices doorways, stalling and tumbling differently."""
import importlib
import os
import unittest
from unittest.mock import patch


def load_reward(name):
    with patch.dict(os.environ, {"DRLNAV_REWARD": name}):
        from turtlebot3_drl.common import settings
        importlib.reload(settings)
        from turtlebot3_drl.drl_environment import reward
        importlib.reload(reward)
        return reward, settings


class IndoorRewardTests(unittest.TestCase):
    def test_default_is_a(self):
        with patch.dict(os.environ, {}, clear=True):
            from turtlebot3_drl.common import settings
            importlib.reload(settings)
            self.assertEqual(settings.REWARD_FUNCTION, "A")

    def test_doorway_is_not_a_crash(self):
        rw, st = load_reward("B")
        rw.reward_initalize(3.0)
        # 0.18 m from both door posts: A charges -20 per step, B nothing.
        self.assertGreater(rw.get_reward(st.UNKNOWN, 0.22, 0.0, 3.0, 0.0, 0.18), -2)
        self.assertLess(rw.get_reward(st.UNKNOWN, 0.22, 0.0, 3.0, 0.0, 0.14), -10)
        rw_a, _ = load_reward("A")
        rw_a.reward_initalize(3.0)
        self.assertLess(rw_a.get_reward(st.UNKNOWN, 0.22, 0.0, 3.0, 0.0, 0.18), -20)

    def test_progress_pays_and_stalling_costs(self):
        rw, st = load_reward("B")
        rw.reward_initalize(3.0)
        towards = rw.get_reward(st.UNKNOWN, 0.22, 0.0, 2.978, 0.0, 1.0)      # one full-speed step closer
        rw.reward_initalize(3.0)
        still = rw.get_reward(st.UNKNOWN, 0.0, 0.0, 3.0, 0.0, 1.0)
        self.assertGreater(towards, still + 4)
        self.assertLess(rw.get_reward(st.TIMEOUT, 0.0, 0.0, 3.0, 0.0, 1.0), -500)
        self.assertLess(rw.get_reward(st.TUMBLE, 0.0, 0.0, 3.0, 0.0, 1.0), -1000)
        self.assertGreater(rw.get_reward(st.SUCCESS, 0.22, 0.0, 0.1, 0.0, 1.0), 2400)
        self.assertLess(rw.get_reward(st.COLLISION_WALL, 0.22, 0.0, 2.0, 0.0, 0.1), -2000)

    def test_progress_resets_each_episode(self):
        rw, st = load_reward("B")
        rw.reward_initalize(2.0)
        rw.get_reward(st.UNKNOWN, 0.22, 0.0, 1.0, 0.0, 1.0)                 # big jump inside an episode
        rw.reward_initalize(4.0)                                           # new episode, farther goal
        r = rw.get_reward(st.UNKNOWN, 0.22, 0.0, 4.0, 0.0, 1.0)             # no progress yet
        self.assertAlmostEqual(r, -1.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
