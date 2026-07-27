#!/usr/bin/env python3
"""Focused semantic tests for the TurtleBot3 Gymnasium migration."""

import math
import types
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from turtlebot3_drl.common.replaybuffer import ReplayBuffer
from turtlebot3_drl.common.settings import UNKNOWN
from turtlebot3_drl.drl_agent.ddpg import DDPG
from turtlebot3_drl.drl_agent.drl_agent_gymnasium import DrlAgentGymnasium
from turtlebot3_drl.drl_gazebo.drl_gazebo import DRLGazebo, WORLD_RESET_SERVICE
from turtlebot3_drl.drl_environment import reward as rw
from turtlebot3_drl.drl_environment.drl_environment import (
    DRLEnvironment,
    MAX_GOAL_DISTANCE,
    NUM_SCAN_SAMPLES,
)
from turtlebot3_drl.drl_environment.drl_environment_gymnasium import (
    DRLGymnasiumEnvironment,
)


class _ActionSpace:
    low = np.array([-1.0, -1.0], dtype=np.float32)
    high = np.array([1.0, 1.0], dtype=np.float32)
    dtype = np.float32


class _AgentModel:
    batch_size = 999
    state_size = NUM_SCAN_SAMPLES + 4
    stack_depth = 3
    frame_skip = 4

    def get_action(self, state, training, step):
        return [0.0, 0.0]

    def get_action_random(self):
        return [0.0, 0.0]


class _OneStepEnv:
    action_space = _ActionSpace()

    def __init__(self, terminated=False, truncated=False):
        self.terminated = terminated
        self.truncated = truncated

    def reset(self):
        return np.zeros(NUM_SCAN_SAMPLES + 4, dtype=np.float32), {"success": UNKNOWN}

    def step(self, action):
        outcome = 4 if self.truncated else 1 if self.terminated else UNKNOWN
        return (
            np.ones(NUM_SCAN_SAMPLES + 4, dtype=np.float32),
            1.0,
            self.terminated,
            self.truncated,
            {"success": outcome},
        )


def _run_agent_transition(terminated=False, truncated=False, step_limit=None):
    agent = DrlAgentGymnasium.__new__(DrlAgentGymnasium)
    agent.env = _OneStepEnv(terminated=terminated, truncated=truncated)
    agent.model = _AgentModel()
    agent.training = True
    agent.total_steps = 0
    agent.observe_steps = 0
    agent.replay_buffer = ReplayBuffer(10)
    result = agent.run_episode(max_steps_per_episode=step_limit)
    transition = agent.replay_buffer.buffer[-1]
    return result, bool(transition[4][0])


class ReplayMaskTests(unittest.TestCase):
    def test_terminated_transition_disables_bootstrap(self):
        _, replay_done = _run_agent_transition(terminated=True)
        self.assertTrue(replay_done)

    def test_truncated_transition_keeps_bootstrap(self):
        _, replay_done = _run_agent_transition(truncated=True)
        self.assertFalse(replay_done)

    def test_agent_step_limit_is_a_truncation_not_terminal(self):
        result, replay_done = _run_agent_transition(step_limit=1)
        self.assertFalse(result["env_done"])
        self.assertFalse(replay_done)


class _ZeroActor(nn.Module):
    def forward(self, state):
        return torch.zeros((state.shape[0], 2), dtype=state.dtype, device=state.device)


class _ConstantCritic(nn.Module):
    def forward(self, state, action):
        return torch.full((state.shape[0], 1), 10.0, dtype=state.dtype, device=state.device)


class TDTargetTests(unittest.TestCase):
    def test_ddpg_target_bootstraps_only_when_mask_is_zero(self):
        model = DDPG(torch.device("cpu"), 1)
        model.actor_target = _ZeroActor()
        model.critic_target = _ConstantCritic()
        captured_targets = []

        def capture_loss(prediction, target):
            if not captured_targets:
                captured_targets.append(target.detach().clone())
            return F.smooth_l1_loss(prediction, target)

        model.loss_function = capture_loss
        state = torch.zeros((2, model.state_size), dtype=torch.float32)
        action = torch.zeros((2, model.action_size), dtype=torch.float32)
        reward = torch.ones((2, 1), dtype=torch.float32)
        next_state = torch.ones_like(state)
        terminal_mask = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

        model.train(state, action, reward, next_state, terminal_mask)

        expected = torch.tensor([[10.9], [1.0]], dtype=torch.float32)
        self.assertTrue(torch.allclose(captured_targets[0], expected, atol=1e-5))


class StateParityTests(unittest.TestCase):
    def _base_state(self):
        return {
            "scan_ranges": np.linspace(0.0, 1.0, NUM_SCAN_SAMPLES).tolist(),
            "goal_distance": MAX_GOAL_DISTANCE * 0.5,
            "goal_angle": math.pi / 4,
            "local_step": 0,
        }

    def test_state_dimension_order_matches_legacy_get_state(self):
        legacy = types.SimpleNamespace(**self._base_state())
        gym_env = types.SimpleNamespace(**self._base_state())
        gym_env.observation_space = spaces.Box(
            low=np.array([0.0] * NUM_SCAN_SAMPLES + [0.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.ones(NUM_SCAN_SAMPLES + 4, dtype=np.float32),
            dtype=np.float32,
        )
        previous_action = (0.25, -0.5)

        legacy_state = np.asarray(
            DRLEnvironment.get_state(legacy, *previous_action), dtype=np.float32
        )
        gym_state = DRLGymnasiumEnvironment.get_state(gym_env, *previous_action)

        self.assertEqual(gym_state.shape, (NUM_SCAN_SAMPLES + 4,))
        self.assertTrue(np.allclose(gym_state, legacy_state))

    def test_step_embeds_legacy_previous_action(self):
        captured = []

        class _Publisher:
            def publish(self, message):
                pass

        env = types.SimpleNamespace(
            action_space=spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            previous_action=np.array([0.25, -0.5], dtype=np.float32),
            cmd_vel_pub=_Publisher(),
            succeed=UNKNOWN,
            done=False,
            total_distance=0.0,
            local_step=1,
            goal_distance=1.0,
            goal_angle=0.0,
            obstacle_distance=3.5,
        )
        env._spin_once = lambda: None
        env._get_info = lambda: {"success": UNKNOWN}

        def capture_state(linear, angular):
            captured.append((float(linear), float(angular)))
            return np.zeros(NUM_SCAN_SAMPLES + 4, dtype=np.float32)

        env.get_state = capture_state
        rw.reward_initalize(1.0)
        DRLGymnasiumEnvironment.step(env, np.array([0.75, 0.5], dtype=np.float32))

        self.assertEqual(captured[0], (0.25, -0.5))


class WorldResetTests(unittest.TestCase):
    def test_goal_manager_uses_reset_world_to_preserve_animation_time(self):
        self.assertEqual(WORLD_RESET_SERVICE, "reset_world")

    def test_reset_world_calls_the_configured_service(self):
        class _Client:
            def __init__(self):
                self.wait_count = 0
                self.requests = []

            def wait_for_service(self, timeout_sec):
                self.wait_count += 1
                return True

            def call_async(self, request):
                self.requests.append(request)

        client = _Client()
        goal_manager = types.SimpleNamespace(reset_world_client=client)
        DRLGazebo.reset_world(goal_manager)

        self.assertEqual(client.wait_count, 1)
        self.assertEqual(len(client.requests), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
