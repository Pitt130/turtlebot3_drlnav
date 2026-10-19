#!/usr/bin/env python3
"""Dependency-light tests for the integrated Gymnasium agent.

The repository's full semantic suite needs ROS 2, Gymnasium, and PyTorch.
These tests execute the real agent definitions from source while replacing
those unavailable integrations with small deterministic fakes.
"""

import argparse
import ast
import os
import pickle
from pathlib import Path
import time
from types import SimpleNamespace
import unittest

import numpy as np


AGENT_PATH = Path(
    os.environ.get(
        "DRL_AGENT_PATH",
        Path(__file__).with_name("drl_agent_gymnasium.py"),
    )
)


class FakeReplayBuffer:
    def __init__(self, size):
        self.buffer = []
        self.size = size

    def add_sample(self, *transition):
        self.buffer.append(transition)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)

    def get_length(self):
        return len(self.buffer)


class FakeModel:
    buffer_size = 100
    batch_size = 1
    state_size = 2
    stack_depth = 3
    frame_skip = 1
    stacking_enabled = False
    networks = []

    def __init__(self, device=None, sim_speed=1):
        self.random_calls = 0
        self.policy_calls = 0
        self.train_calls = 0

    def get_action_random(self):
        self.random_calls += 1
        return [0.0, 0.0]

    def get_action(self, state, training, step):
        self.policy_calls += 1
        return [0.25, -0.25]

    def _train(self, replay_buffer):
        self.train_calls += 1
        return 2.0, 4.0


class FakeActionSpace:
    low = np.array([-1.0, -1.0], dtype=np.float32)
    high = np.array([1.0, 1.0], dtype=np.float32)
    dtype = np.float32


class EndlessEnv:
    action_space = FakeActionSpace()

    def __init__(self):
        self.steps = 0
        self.closed = False

    def reset(self):
        self.steps = 0
        return np.zeros(2, dtype=np.float32), {"success": 0}

    def step(self, action):
        self.steps += 1
        observation = np.full(2, self.steps, dtype=np.float32)
        return observation, 1.0, False, False, {"success": 0}

    def close(self):
        self.closed = True


def load_agent_definitions():
    source = AGENT_PATH.read_text()
    tree = ast.parse(source, filename=str(AGENT_PATH))
    wanted = {
        "_as_float32_observation",
        "_simulation_stage",
        "_nonnegative_int",
        "_positive_int",
        "_extract_outcome",
        "DrlAgentGymnasium",
    }
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted
    ]
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace = {
        "argparse": argparse,
        "os": os,
        "pickle": pickle,
        "time": time,
        "np": np,
        "COLLISION_OBSTACLE": 3,
        "COLLISION_WALL": 2,
        "ENABLE_STACKING": False,
        "GRAPH_DRAW_INTERVAL": 10,
        "MODEL_STORE_INTERVAL": 100,
        "OBSERVE_STEPS": 3000,
        "RESULTS_NUM": 6,
        "SUCCESS": 1,
        "TIMEOUT": 4,
        "TUMBLE": 5,
        "UNKNOWN": 0,
        "ReplayBuffer": FakeReplayBuffer,
        "StorageManager": object,
        "DDPG": FakeModel,
        "TD3": FakeModel,
        "util": SimpleNamespace(check_gpu=lambda: "cpu", stage=4),
    }
    exec(compile(module, str(AGENT_PATH), "exec"), namespace)
    return source, namespace


SOURCE, DEFINITIONS = load_agent_definitions()
DrlAgentGymnasium = DEFINITIONS["DrlAgentGymnasium"]
extract_outcome = DEFINITIONS["_extract_outcome"]


class IntegrationTests(unittest.TestCase):
    def make_agent(self, observe_steps=0):
        agent = DrlAgentGymnasium.__new__(DrlAgentGymnasium)
        agent.env = EndlessEnv()
        agent.model = FakeModel()
        agent.training = True
        agent.total_steps = 0
        agent.observe_steps = observe_steps
        agent.episode = 0
        agent.replay_buffer = FakeReplayBuffer(100)
        agent.graph = None
        agent.logger = None
        agent.sm = None
        agent.last_saved_episode = None
        agent.loaded_graphdata = [0, [], [], [], []]
        return agent

    def test_no_seed_feature_was_added(self):
        self.assertNotIn("--seed", SOURCE)
        self.assertNotIn("manual_seed", SOURCE)
        self.assertNotIn("env.reset(seed=", SOURCE)

    def test_uses_tomasvr_gpu_check(self):
        self.assertNotIn("def _check_gpu", SOURCE)
        self.assertIn("util.check_gpu()", SOURCE)

    def test_outcome_prefers_categorical_outcome(self):
        self.assertEqual(
            extract_outcome({"success": False, "outcome": "collision_wall"}),
            2,
        )
        self.assertEqual(extract_outcome({"success": True}), 1)
        self.assertEqual(extract_outcome({"success": 4}), 4)
        self.assertEqual(extract_outcome({"success": float("nan")}), 0)

    def test_warmup_trains_but_is_not_recorded(self):
        """drl_agent.py optimizes during warm-up; the rollout is just not logged."""
        agent = self.make_agent(observe_steps=3)
        result = agent.run_episode(max_steps_per_episode=10)

        # Warm-up does not cut the rollout short and does not skip optimization.
        self.assertEqual(result["step"], 10)
        self.assertEqual(agent.total_steps, 10)
        self.assertEqual(agent.model.random_calls, 3)
        self.assertEqual(agent.model.policy_calls, 7)
        self.assertGreater(agent.model.train_calls, 0)
        self.assertEqual(result["train_updates"], agent.model.train_calls)

        # total_steps already passed observe_steps, so this one is recorded.
        agent.finish_episode(**result)
        self.assertEqual(agent.episode, 1)
        self.assertEqual(len(agent.loaded_graphdata[1]), 1)

    def test_warmup_rollout_below_the_threshold_is_not_recorded(self):
        agent = self.make_agent(observe_steps=50)
        result = agent.run_episode(max_steps_per_episode=5)
        agent.finish_episode(**result)
        self.assertEqual(agent.episode, 0)

    def test_stage_defaults_to_the_running_simulation(self):
        agent = DrlAgentGymnasium(training=True, env=EndlessEnv(), observe_steps=0)
        self.assertEqual(agent.stage, 4)

    def test_only_terminal_transitions_stop_bootstrapping(self):
        class TimeoutEnv(EndlessEnv):
            def step(self, action):
                obs, r, _, _, info = super().step(action)
                return obs, r, False, self.steps >= 3, info

        class TerminalEnv(EndlessEnv):
            def step(self, action):
                obs, r, _, _, info = super().step(action)
                return obs, r, self.steps >= 3, False, info

        agent = self.make_agent()
        agent.env = TimeoutEnv()
        agent.run_episode()
        self.assertEqual(list(agent.replay_buffer.buffer[-1][4]), [False])

        agent_terminal = self.make_agent()
        agent_terminal.env = TerminalEnv()
        agent_terminal.run_episode()
        self.assertEqual(list(agent_terminal.replay_buffer.buffer[-1][4]), [True])

        agent2 = self.make_agent()
        agent2.run_episode(max_steps_per_episode=3)
        self.assertEqual(list(agent2.replay_buffer.buffer[-1][4]), [False])

    def test_non_finite_action_does_not_reach_the_environment(self):
        class NanModel(FakeModel):
            def get_action(self, state, training, step):
                return [float("nan"), 0.0]

        agent = self.make_agent()
        agent.model = NanModel()
        sent = agent._format_env_action(np.array([np.nan, 0.0], dtype=np.float32))
        self.assertTrue(np.isfinite(sent).all())

    def test_frame_stack_keeps_reset_observation(self):
        agent = self.make_agent()
        agent.model.state_size = 1
        agent.model.stack_depth = 3
        agent.model.frame_skip = 1
        agent.model.stacking_enabled = True

        state, frame_buffer = agent._prepare_reset_state([10.0])
        next_state, _ = agent._prepare_next_state([20.0], frame_buffer)

        np.testing.assert_array_equal(state, [0.0, 0.0, 10.0])
        np.testing.assert_array_equal(next_state, [0.0, 10.0, 20.0])

    def test_graph_history_is_trimmed_and_padded_to_loaded_episode(self):
        agent = self.make_agent()
        agent.episode = 3
        normalized = agent._normalize_graph_history(
            [50, [1, 2, 3, 4], [10.0], [], [0.5, 0.4]]
        )

        self.assertEqual(normalized[0], 50)
        self.assertEqual(normalized[1], [1, 2, 3])
        self.assertEqual(normalized[2], [10.0, 0.0, 0.0])
        self.assertEqual(normalized[3], [0.0, 0.0, 0.0])
        self.assertEqual(normalized[4], [0.5, 0.4, 0.0])

    def test_invalid_episode_limit_is_rejected_before_env_step(self):
        agent = self.make_agent()
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            agent.run_episode(max_steps_per_episode=0)
        self.assertEqual(agent.env.steps, 0)

    def test_testing_requires_a_saved_session(self):
        env = EndlessEnv()
        with self.assertRaisesRegex(ValueError, "Testing requires load_session"):
            DrlAgentGymnasium(training=False, env=env)
        self.assertFalse(env.closed)

    def test_constructor_closes_environment_when_storage_setup_fails(self):
        env = EndlessEnv()
        previous = os.environ.pop("DRLNAV_BASE_PATH", None)
        try:
            with self.assertRaisesRegex(RuntimeError, "DRLNAV_BASE_PATH"):
                DrlAgentGymnasium(training=True, env=env, enable_storage=True)
        finally:
            if previous is not None:
                os.environ["DRLNAV_BASE_PATH"] = previous
        self.assertTrue(env.closed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
