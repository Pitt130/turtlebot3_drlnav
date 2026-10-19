"""The Gazebo goal manager replies only after its Gazebo requests finish.

Fake delete_entity, spawn_entity and reset_simulation services stand in for Gazebo.
Run on an isolated ROS_DOMAIN_ID; no simulator is started.
"""

import os
import threading
import time
import unittest
from unittest.mock import patch

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from geometry_msgs.msg import Pose
from std_srvs.srv import Empty
from turtlebot3_msgs.srv import RingGoal

from turtlebot3_drl.drl_gazebo import drl_gazebo

DELAY = 0.3  # seconds each fake Gazebo request takes


class FakeGazebo(Node):
    def __init__(self):
        super().__init__("fake_gazebo")
        self.events = []
        self.create_service(DeleteEntity, "delete_entity", self.slow("delete_entity"))
        self.create_service(SpawnEntity, "spawn_entity", self.slow("spawn_entity"))
        self.create_service(Empty, "reset_simulation", self.slow("reset_simulation"))
        self.create_subscription(Pose, "goal_pose", lambda _: self.events.append("goal_pose"), 10)

    def slow(self, name):
        def callback(request, response):
            time.sleep(DELAY)
            self.events.append(name)
            return response
        return callback


class GoalManagerSyncTests(unittest.TestCase):
    def setUp(self):
        self.assertEqual(os.environ.get("ROS_DOMAIN_ID"), "231", "Run on an isolated ROS_DOMAIN_ID")
        rclpy.init()
        self.gazebo = FakeGazebo()
        with patch.object(drl_gazebo, "current_stage", return_value=4):
            self.manager = drl_gazebo.DRLGazebo()
        self.caller = rclpy.create_node("task_caller")
        self.executor = MultiThreadedExecutor()
        for node in (self.gazebo, self.manager, self.caller):
            self.executor.add_node(node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()
        self.wait_until(lambda: "goal_pose" in self.gazebo.events, 10.0)

    def tearDown(self):
        self.executor.shutdown()
        self.thread.join(timeout=5.0)
        for node in (self.gazebo, self.manager, self.caller):
            node.destroy_node()
        rclpy.shutdown()

    def wait_until(self, condition, timeout):
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(condition())

    def call(self, service):
        client = self.caller.create_client(RingGoal, service)
        self.assertTrue(client.wait_for_service(timeout_sec=5.0))
        self.gazebo.events.clear()
        sent = time.monotonic()
        future = client.call_async(RingGoal.Request())
        self.wait_until(future.done, 15.0)
        elapsed = time.monotonic() - sent
        before_reply = list(self.gazebo.events)
        # The goal is published before the reply but may be delivered after it.
        self.wait_until(lambda: "goal_pose" in self.gazebo.events, 2.0)
        return elapsed, before_reply, list(self.gazebo.events)

    def test_startup_waits_for_each_request(self):
        self.assertEqual(self.gazebo.events[:4], ["delete_entity", "reset_simulation", "spawn_entity", "goal_pose"])

    def test_task_fail_replies_after_reset_and_spawn(self):
        elapsed, before_reply, events = self.call("task_fail")
        self.assertEqual(events, ["delete_entity", "reset_simulation", "spawn_entity", "goal_pose"])
        self.assertEqual(before_reply[:3], ["delete_entity", "reset_simulation", "spawn_entity"])
        self.assertGreaterEqual(elapsed, 3 * DELAY)

    def test_task_succeed_replies_after_spawn(self):
        elapsed, before_reply, events = self.call("task_succeed")
        self.assertEqual(events, ["delete_entity", "spawn_entity", "goal_pose"])
        self.assertEqual(before_reply[:2], ["delete_entity", "spawn_entity"])
        self.assertGreaterEqual(elapsed, 2 * DELAY)


if __name__ == "__main__":
    unittest.main(verbosity=2)
