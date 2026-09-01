#!/usr/bin/env python3
#
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Gymnasium training path for TurtleBot3 DRL.

import argparse
import copy
import os
import pickle
import sys
import time

import numpy as np

from ..common import utilities as util
from ..common.settings import (
    ENABLE_STACKING,
    GRAPH_DRAW_INTERVAL,
    MODEL_STORE_INTERVAL,
    OBSERVE_STEPS,
    UNKNOWN,
)
from ..common.replaybuffer import ReplayBuffer
from ..common.storagemanager import StorageManager
from .ddpg import DDPG
from .td3 import TD3


def _as_float32_observation(observation):
    return np.asarray(observation, dtype=np.float32)


def _extract_outcome(info):
    if not isinstance(info, dict):
        return UNKNOWN

    # The Gymnasium environment reports success/outcome through info instead
    # of the old ROS DrlStep service response.
    outcome = info.get("success", info.get("outcome", UNKNOWN))
    if isinstance(outcome, np.generic):
        outcome = outcome.item()
    if outcome is None:
        return UNKNOWN
    return outcome


class DrlAgentGymnasium:
    """Train or evaluate DDPG/TD3 against the Gymnasium TurtleBot3 environment.

    This class intentionally does not inherit from rclpy.node.Node and does not
    call util.init_episode() or util.step(). The original ROS service based
    training path remains in drl_agent.py.
    """

    def __init__(
        self,
        training=True,
        algorithm="td3",
        load_session="",
        load_episode=0,
        env=None,
        observe_steps=OBSERVE_STEPS,
        enable_storage=False,
        enable_logger=False,
        enable_graph=False,
        stage=1,
        sim_speed=1,
    ):
        self.algorithm = algorithm.lower()
        self.training = bool(int(training)) if isinstance(training, str) else bool(training)
        self.load_session = load_session
        self.episode = int(load_episode)

        if self.algorithm not in ("ddpg", "td3"):
            raise ValueError("Gymnasium agent supports continuous-control algorithms: ddpg, td3")
        if not self.training and not self.load_session:
            raise ValueError("Testing requires load_session to specify which model to evaluate")

        self.env = env if env is not None else self._make_env()
        # Random replay-buffer warm-up is a training-only phase and must not
        # delay result logging during evaluation.
        self.observe_steps = int(observe_steps) if self.training else 0
        self.total_steps = 0
        self.stage = stage

        self.device = util.check_gpu()
        if self.algorithm == "ddpg":
            self.model = DDPG(self.device, sim_speed)
        else:
            self.model = TD3(self.device, sim_speed)

        self.replay_buffer = ReplayBuffer(self.model.buffer_size)
        self.storage_enabled = bool(enable_storage or self.load_session)
        self.logger_enabled = bool(enable_logger)
        self.graph_enabled = bool(enable_graph)
        self.sm = None
        self.logger = None
        self.graph = None
        self.last_saved_episode = None
        self.loaded_graphdata = None

        if self.storage_enabled:
            self._init_storage()
        if self.graph_enabled:
            from ..common.graph import Graph

            self.graph = Graph()
            if self.sm:
                self.graph.session_dir = self.sm.session_dir
            self._restore_graph_history()
        if self.logger_enabled:
            self._init_logger()

    def _make_env(self):
        from turtlebot3_drl.drl_environment.drl_environment_gymnasium import make_env

        return make_env()

    def _init_storage(self):
        if not os.getenv("DRLNAV_BASE_PATH"):
            raise RuntimeError("DRLNAV_BASE_PATH is required when Gymnasium agent storage is enabled")

        self.sm = StorageManager(
            self.algorithm,
            self.load_session,
            self.episode,
            self.device,
            self.stage,
        )
        if self.load_session:
            del self.model
            self.model = self.sm.load_model()
            self.model.device = self.device
            self.sm.load_weights(self.model.networks)
            if self.training:
                buffer_path = os.path.join(
                    self.load_session,
                    "stage" + str(self.sm.stage) + "_latest_buffer.pkl",
                )
                self.replay_buffer.buffer = self.sm.load_replay_buffer(self.model.buffer_size, buffer_path)
            self.last_saved_episode = self.episode
            self.loaded_graphdata = self._load_graph_history()
            if self.loaded_graphdata is not None:
                self.total_steps = self.loaded_graphdata[0]
                print(f"resumed session at global step: {self.total_steps}")
            elif self.training:
                # A loaded model has already completed its original warm-up.
                # Missing optional graph history must not make it act randomly
                # for OBSERVE_STEPS again.
                self.total_steps = self.observe_steps
                print(
                    "resumed session has no usable global step count; "
                    f"assuming observation warm-up is complete at {self.total_steps} steps"
                )
        else:
            self.sm.new_session_dir(self.stage)
            self.sm.store_model(self.model)

    def _init_logger(self):
        if self.sm is None:
            raise RuntimeError("Logger requires storage to be enabled")

        from ..common.logger import Logger

        self.logger = Logger(
            self.training,
            self.sm.machine_dir,
            self.sm.session_dir,
            self.sm.session,
            self.model.get_model_parameters(),
            self.model.get_model_configuration(),
            str(self.stage),
            self.algorithm,
            self.episode,
        )

    def _load_graph_history(self):
        try:
            graphdata = self.sm.load_graphdata()
        except (OSError, EOFError, pickle.PickleError, TypeError, ValueError) as error:
            print(f"could not load graph data of resumed session: {error}")
            return None

        if not isinstance(graphdata, (list, tuple)) or len(graphdata) != 5:
            print("resumed session contains no usable graph data")
            return None

        try:
            total_steps = int(graphdata[0])
            if total_steps < 0:
                raise ValueError("global step count cannot be negative")

            columns = []
            for column in graphdata[1:]:
                if isinstance(column, (str, bytes)):
                    raise TypeError("graph history columns must be sequences")
                columns.append(list(column))
        except (TypeError, ValueError) as error:
            print(f"resumed session contains no usable graph data: {error}")
            return None

        return [total_steps] + columns

    def _restore_graph_history(self):
        if not self.load_session or self.episode <= 0 or self.graph is None:
            return

        graphdata = self.loaded_graphdata or [self.total_steps, [], [], [], []]
        columns = []
        placeholders = (UNKNOWN, 0.0, 0.0, 0.0)
        for source, placeholder in zip(graphdata[1:], placeholders):
            column = list(source[:self.episode])
            if len(column) < self.episode:
                column.extend([placeholder] * (self.episode - len(column)))
            columns.append(column)

        self.graph.set_graphdata([self.total_steps] + columns, self.episode)

    def _prepare_reset_state(self, observation):
        state = _as_float32_observation(observation)
        if not ENABLE_STACKING:
            return state, None

        frame_buffer = [0.0] * (self.model.state_size * self.model.stack_depth * self.model.frame_skip)
        stacked_state = [0.0] * (self.model.state_size * (self.model.stack_depth - 1)) + list(state)
        return np.asarray(stacked_state, dtype=np.float32), frame_buffer

    def _prepare_next_state(self, observation, frame_buffer):
        next_state = _as_float32_observation(observation)
        if not ENABLE_STACKING:
            return next_state, frame_buffer

        frame_buffer = frame_buffer[self.model.state_size:] + list(next_state)
        stacked_next_state = []
        for depth in range(self.model.stack_depth):
            start = (
                self.model.state_size * (self.model.frame_skip - 1)
                + (self.model.state_size * self.model.frame_skip * depth)
            )
            stacked_next_state += frame_buffer[start:start + self.model.state_size]
        return np.asarray(stacked_next_state, dtype=np.float32), frame_buffer

    def _get_action(self, state, step):
        if self.training and self.total_steps < self.observe_steps:
            action = self.model.get_action_random()
        else:
            action = self.model.get_action(state, self.training, step)
        return np.asarray(action, dtype=np.float32)

    def _format_env_action(self, action):
        if hasattr(self.env, "action_space"):
            low = getattr(self.env.action_space, "low", None)
            high = getattr(self.env.action_space, "high", None)
            dtype = getattr(self.env.action_space, "dtype", np.float32)
            if low is not None and high is not None:
                action = np.clip(action, low, high)
            return np.asarray(action, dtype=dtype)
        return np.asarray(action, dtype=np.float32)

    def run(self, max_episodes=None, max_total_steps=None, max_steps_per_episode=None):
        episodes_run = 0
        while max_episodes is None or episodes_run < max_episodes:
            if max_total_steps is not None and self.total_steps >= max_total_steps:
                break

            episode_step_limit = max_steps_per_episode
            if max_total_steps is not None:
                remaining_steps = max_total_steps - self.total_steps
                episode_step_limit = (
                    remaining_steps
                    if episode_step_limit is None
                    else min(episode_step_limit, remaining_steps)
                )

            result = self.run_episode(max_steps_per_episode=episode_step_limit)
            episode_before = self.episode
            self.finish_episode(**result)
            # Warm-up rollouts are not formal episodes and should not consume
            # the caller's max_episodes budget.
            if self.episode != episode_before:
                episodes_run += 1

            if max_total_steps is not None and self.total_steps >= max_total_steps:
                break

    def run_episode(self, max_steps_per_episode=None):
        # Gymnasium reset replaces the old:
        # state = util.init_episode(self)
        observation, info = self.env.reset()
        state, frame_buffer = self._prepare_reset_state(observation)

        episode_done = False
        step = 0
        reward_sum = 0.0
        loss_critic = 0.0
        loss_actor = 0.0
        outcome = _extract_outcome(info)
        distance_traveled = 0.0
        env_done = False
        episode_start = time.perf_counter()

        while not episode_done:
            action = self._get_action(state, step)
            action_current = self._format_env_action(action)

            # Gymnasium step replaces the old service call:
            # next_state, reward, episode_done, outcome, distance_traveled = util.step(...)
            observation, reward, terminated, truncated, info = self.env.step(action_current)
            terminated = bool(terminated)
            truncated = bool(truncated)
            env_done = terminated or truncated
            if (
                not env_done
                and max_steps_per_episode is not None
                and step + 1 >= max_steps_per_episode
            ):
                truncated = True
            episode_done = terminated or truncated
            next_state, frame_buffer = self._prepare_next_state(observation, frame_buffer)
            reward = float(reward)
            outcome = _extract_outcome(info)
            distance_traveled = float(info.get("distance_traveled", 0.0)) if isinstance(info, dict) else 0.0
            reward_sum += reward

            if self.training:
                self.replay_buffer.add_sample(
                    state,
                    action_current,
                    [reward],
                    next_state,
                    # Only a true MDP terminal disables bootstrapping. Gymnasium
                    # truncations and this agent's local step limit end the
                    # rollout, but the value target must still use next_state.
                    [terminated],
                )
                if self.replay_buffer.get_length() >= self.model.batch_size:
                    loss_c, loss_a = self.model._train(self.replay_buffer)
                    loss_critic += float(loss_c)
                    loss_actor += float(loss_a)

            state = copy.deepcopy(next_state)
            step += 1
            self.total_steps += 1

        duration = time.perf_counter() - episode_start
        return {
            "step": step,
            "eps_duration": duration,
            "outcome": outcome,
            "dist_traveled": distance_traveled,
            "reward_sum": reward_sum,
            "loss_critic": loss_critic,
            "loss_actor": loss_actor,
            "done": episode_done,
            "env_done": env_done,
        }

    def finish_episode(
        self,
        step,
        eps_duration,
        outcome,
        dist_traveled,
        reward_sum,
        loss_critic,
        loss_actor,
        done=True,
        env_done=True,
    ):
        if self.total_steps < self.observe_steps:
            print(
                f"Observe phase: {self.total_steps}/{self.observe_steps} steps, "
                f"episode reward: {reward_sum:.3f}, replay buffer: {self.replay_buffer.get_length()}"
            )
            return

        self.episode += 1
        print(
            f"Epi: {self.episode:<5}R: {reward_sum:<8.3f}"
            f"outcome: {outcome!s:<13}steps: {step:<6}"
            f"steps_total: {self.total_steps:<7}time: {eps_duration:<6.2f}"
        )

        if not self.training:
            if self.logger:
                self.logger.update_test_results(step, outcome, dist_traveled, eps_duration, 0)
            return

        if self.graph:
            self.graph.update_data(step, self.total_steps, outcome, reward_sum, loss_critic, loss_actor)
        if self.logger:
            avg_critic = loss_critic / step if step else 0.0
            avg_actor = loss_actor / step if step else 0.0
            self.logger.file_log.write(
                f"{self.episode}, {reward_sum}, {outcome}, {eps_duration}, {step}, "
                f"{self.total_steps}, {self.replay_buffer.get_length()}, {avg_critic}, {avg_actor}\n"
            )
            self.logger.file_log.flush()

        if self.sm and ((self.episode % MODEL_STORE_INTERVAL == 0) or (self.episode == 1)):
            self._save_checkpoint()
        if self.graph and ((self.episode % GRAPH_DRAW_INTERVAL == 0) or (self.episode == 1)):
            self.graph.draw_plots(self.episode)

    def _save_checkpoint(self):
        # Preserve the global step count even when plots are disabled so a
        # resumed model does not repeat random replay-buffer warm-up.
        graphdata = self.graph.graphdata if self.graph else [self.total_steps, [], [], [], []]
        self.sm.save_session(self.episode, self.model.networks, graphdata, self.replay_buffer.buffer)
        self.last_saved_episode = self.episode
        if self.logger and self.graph:
            self.logger.update_comparison_file(
                self.episode,
                self.graph.get_success_count(),
                self.graph.get_reward_average(),
            )

    def close(self):
        try:
            if (
                self.training
                and self.sm
                and self.episode > 0
                and self.episode != self.last_saved_episode
            ):
                print(f"saving final checkpoint at episode {self.episode}")
                self._save_checkpoint()
        finally:
            if self.logger:
                self.logger.close()
            if hasattr(self.env, "close"):
                self.env.close()


def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Train TurtleBot3 DRL with the Gymnasium environment")
    parser.add_argument("--algorithm", choices=["ddpg", "td3"], default="td3")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None, help="Maximum total environment steps")
    parser.add_argument("--steps-per-episode", type=int, default=None)
    parser.add_argument("--observe-steps", type=int, default=OBSERVE_STEPS)
    parser.add_argument("--load-session", default="")
    parser.add_argument("--load-episode", type=int, default=0)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--storage", action="store_true")
    parser.add_argument("--logger", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--stage", type=int, default=1)
    parsed = parser.parse_args(args)

    agent = DrlAgentGymnasium(
        training=not parsed.test,
        algorithm=parsed.algorithm,
        load_session=parsed.load_session,
        load_episode=parsed.load_episode,
        observe_steps=parsed.observe_steps,
        enable_storage=parsed.storage,
        enable_logger=parsed.logger,
        enable_graph=parsed.graph,
        stage=parsed.stage,
    )
    try:
        agent.run(
            max_episodes=parsed.episodes,
            max_total_steps=parsed.steps,
            max_steps_per_episode=parsed.steps_per_episode,
        )
    finally:
        agent.close()


if __name__ == "__main__":
    main()
