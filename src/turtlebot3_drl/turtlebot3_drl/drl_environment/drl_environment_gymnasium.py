import copy
import math
import re
import sys
import time

import gymnasium as gym
import numpy
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec

from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from turtlebot3_msgs.srv import RingGoal
from std_srvs.srv import Empty

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)

from . import reward as rw
from ..common import utilities as util
from ..common.settings import (
    ARENA_LENGTH,
    ARENA_WIDTH,
    COLLISION_OBSTACLE,
    COLLISION_WALL,
    ENABLE_BACKWARD,
    ENABLE_DYNAMIC_GOALS,
    ENABLE_MOTOR_NOISE,
    EPISODE_TIMEOUT_SECONDS,
    LIDAR_DISTANCE_CAP,
    MAX_NUMBER_OBSTACLES,
    OBSTACLE_RADIUS,
    SPEED_ANGULAR_MAX,
    SPEED_LINEAR_MAX,
    SUCCESS,
    THRESHOLD_COLLISION,
    THREHSOLD_GOAL,
    TIMEOUT,
    TOPIC_ODOM,
    TOPIC_SCAN,
    TOPIC_VELO,
    TUMBLE,
    UNKNOWN,
)


# Automatically retrieved from Gazebo model configuration (40 by default).
# Can be set manually if needed.
NUM_SCAN_SAMPLES = util.get_scan_count()
LINEAR = 0
ANGULAR = 1
MAX_GOAL_DISTANCE = math.sqrt(ARENA_LENGTH**2 + ARENA_WIDTH**2)
SERVICE_WAIT_TIMEOUT_SEC = 3.0


class DRLGymnasiumEnvironment(Node, gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, spin_timeout_sec=0.1, reset_timeout_sec=10.0, step_sim_time=0.1):
        Node.__init__(self, "drl_gymnasium_environment")
        gym.Env.__init__(self)
        self.spec = EnvSpec(
            id="TurtleBot3DRL-v0",
            entry_point=(
                "turtlebot3_drl.drl_environment."
                "drl_environment_gymnasium:make_env"
            ),
            nondeterministic=True,
        )

        with open("/tmp/drlnav_current_stage.txt", "r") as f:
            self.stage = int(f.read())
        print(f"running on stage: {self.stage}")

        if not math.isfinite(step_sim_time) or step_sim_time <= 0:
            raise ValueError("step_sim_time must be positive and finite")
        self.step_sim_time = float(step_sim_time)
        self.odom_stamp = -math.inf
        self.scan_stamp = -math.inf
        self.last_step_sim_dt = 0.0
        self._clock_received = False
        self._clock_epoch = 0
        self._awaiting_rewind = False
        self._pending_outcome = None
        self._needs_reset = True
        self.spin_timeout_sec = spin_timeout_sec
        self.reset_timeout_sec = reset_timeout_sec
        self.episode_timeout = EPISODE_TIMEOUT_SECONDS

        self.scan_topic = TOPIC_SCAN
        self.velo_topic = TOPIC_VELO
        self.odom_topic = TOPIC_ODOM
        self.goal_topic = "goal_pose"

        self.goal_x, self.goal_y = 0.0, 0.0
        self.robot_x, self.robot_y = 0.0, 0.0
        self.robot_x_prev, self.robot_y_prev = 0.0, 0.0
        self.robot_heading = 0.0
        self.total_distance = 0.0
        self.robot_tilt = 0.0

        self.done = False
        self.succeed = UNKNOWN
        self.episode_deadline = math.inf
        self.reset_deadline = False
        self.clock_msgs_skipped = 0

        self.obstacle_distances = [math.inf] * MAX_NUMBER_OBSTACLES

        self.new_goal = False
        self.goal_angle = 0.0
        self.goal_distance = MAX_GOAL_DISTANCE
        self.initial_distance_to_goal = MAX_GOAL_DISTANCE

        self.scan_ranges = [LIDAR_DISTANCE_CAP] * NUM_SCAN_SAMPLES
        self.obstacle_distance = LIDAR_DISTANCE_CAP

        self.difficulty_radius = 1
        self.local_step = 0
        self.time_sec = 0
        self.previous_action = numpy.zeros(2, dtype=numpy.float32)

        obs_low = numpy.array([0.0] * NUM_SCAN_SAMPLES + [0.0, -1.0, -1.0, -1.0], dtype=numpy.float32)
        obs_high = numpy.array([1.0] * NUM_SCAN_SAMPLES + [1.0, 1.0, 1.0, 1.0], dtype=numpy.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=numpy.float32)
        self.action_space = spaces.Box(
            low=numpy.array([-1.0, -1.0], dtype=numpy.float32),
            high=numpy.array([1.0, 1.0], dtype=numpy.float32),
            dtype=numpy.float32,
        )

        qos = QoSProfile(depth=10)
        sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        qos_clock = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.cmd_vel_pub = self.create_publisher(Twist, self.velo_topic, qos)
        self.goal_pose_sub = self.create_subscription(Pose, self.goal_topic, self.goal_pose_callback, qos)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, sensor_qos)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, qos_profile=sensor_qos)
        self.clock_sub = self.create_subscription(Clock, "/clock", self.clock_callback, qos_profile=qos_clock)
        self.obstacle_odom_sub = self.create_subscription(Odometry, "obstacle/odom", self.obstacle_odom_callback, sensor_qos)
        self.task_succeed_client = self.create_client(RingGoal, "task_succeed")
        self.task_fail_client = self.create_client(RingGoal, "task_fail")
        self.pause_physics_client = self.create_client(Empty, '/pause_physics')
        self.unpause_physics_client = self.create_client(Empty, '/unpause_physics')

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self.closed = False
        self.reset_request_failed = False
        self._goal_reset_future = None

    def goal_pose_callback(self, msg):
        self.goal_x = msg.position.x
        self.goal_y = msg.position.y
        self.new_goal = True
        print(f"new goal! x: {self.goal_x} y: {self.goal_y}")

    def obstacle_odom_callback(self, msg):
        if self._awaiting_rewind:
            return
        match = re.search(r"obstacle(\d+)", msg.child_frame_id)
        if match is None:
            self.get_logger().error(f"received odom was not from an obstacle: {msg.child_frame_id}")
            return

        obstacle_id = int(match.group(1)) - 1
        if obstacle_id < 0 or obstacle_id >= MAX_NUMBER_OBSTACLES:
            self.get_logger().error(f"obstacle id out of range: {msg.child_frame_id}")
            return

        robot_pos = msg.pose.pose.position
        diff_x = self.robot_x - robot_pos.x
        diff_y = self.robot_y - robot_pos.y
        self.obstacle_distances[obstacle_id] = math.sqrt(diff_y**2 + diff_x**2)

    def odom_callback(self, msg):
        if self._awaiting_rewind:
            return
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        roll, pitch, yaw = util.euler_from_quaternion(msg.pose.pose.orientation)
        if not math.isfinite(roll) or not math.isfinite(pitch) or not math.isfinite(yaw):
            self.get_logger().error("received non-finite odometry orientation")
            return

        self.robot_heading = yaw
        self.robot_tilt = max(abs(roll), abs(pitch))

        if self.local_step % 32 == 0:
            self.total_distance += math.sqrt(
                (self.robot_x_prev - self.robot_x) ** 2 +
                (self.robot_y_prev - self.robot_y) ** 2
            )
            self.robot_x_prev = self.robot_x
            self.robot_y_prev = self.robot_y

        diff_y = self.goal_y - self.robot_y
        diff_x = self.goal_x - self.robot_x
        distance_to_goal = math.sqrt(diff_x**2 + diff_y**2)
        heading_to_goal = math.atan2(diff_y, diff_x)
        goal_angle = heading_to_goal - self.robot_heading

        while goal_angle > math.pi:
            goal_angle -= 2 * math.pi
        while goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        self.goal_distance = distance_to_goal
        self.goal_angle = goal_angle
        self.odom_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def scan_callback(self, msg):
        if self._awaiting_rewind:
            return
        if len(msg.ranges) != NUM_SCAN_SAMPLES:
            print(f"more or less scans than expected! check model.sdf, got: {len(msg.ranges)}, expected: {NUM_SCAN_SAMPLES}")

        min_scan_range = LIDAR_DISTANCE_CAP
        for i in range(NUM_SCAN_SAMPLES):
            if i < len(msg.ranges):
                scan_range = float(msg.ranges[i])
            else:
                scan_range = LIDAR_DISTANCE_CAP

            if not math.isfinite(scan_range):
                scan_range = LIDAR_DISTANCE_CAP

            scan_range = float(numpy.clip(scan_range, 0, LIDAR_DISTANCE_CAP))
            self.scan_ranges[i] = float(numpy.clip(scan_range / LIDAR_DISTANCE_CAP, 0, 1))
            if scan_range < min_scan_range:
                min_scan_range = scan_range
        self.obstacle_distance = min_scan_range
        self.scan_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def clock_callback(self, msg):
        stamp = msg.clock.sec + msg.clock.nanosec * 1e-9
        if self._clock_received and stamp < self.time_sec - 1e-9:
            self._clock_epoch += 1
            self.odom_stamp = self.scan_stamp = -math.inf
            self.obstacle_distances = [math.inf] * MAX_NUMBER_OBSTACLES
            self._awaiting_rewind = False
        self.time_sec = stamp
        self._clock_received = True
        if not self.reset_deadline:
            return

        self.clock_msgs_skipped += 1
        if self.clock_msgs_skipped <= 10:
            return

        episode_time = self.episode_timeout
        if ENABLE_DYNAMIC_GOALS:
            episode_time = numpy.clip(episode_time * self.difficulty_radius, 10, 50)
        self.episode_deadline = self.time_sec + episode_time
        self.reset_deadline = False
        self.clock_msgs_skipped = 0

    def stop_reset_robot(self, success):
        # Preserve the terminal observation; request world changes only in reset().
        self.cmd_vel_pub.publish(Twist())
        self.done = True
        self._needs_reset = True
        self._pending_outcome = bool(success)

    def _request_outcome_reset(self, success):
        self.reset_request_failed = False
        self.cmd_vel_pub.publish(Twist())
        self.episode_deadline = math.inf
        self.done = True

        req = RingGoal.Request()
        req.robot_pose_x = self.robot_x
        req.robot_pose_y = self.robot_y
        req.radius = numpy.clip(self.difficulty_radius, 0.5, 4)

        if success:
            self.difficulty_radius *= 1.01
            client = self.task_succeed_client
            service_name = "task_succeed"
        else:
            self.difficulty_radius *= 0.99
            client = self.task_fail_client
            service_name = "task_fail"

        if not client.wait_for_service(timeout_sec=SERVICE_WAIT_TIMEOUT_SEC):
            self.reset_request_failed = True
            self.get_logger().error(f"{service_name} service not available after {SERVICE_WAIT_TIMEOUT_SEC:.1f}s; skipping reset request")
            return
        self.new_goal = False
        self._goal_reset_future = client.call_async(req)

    def _request_reset_goal(self):
        if not self.task_fail_client.wait_for_service(timeout_sec=SERVICE_WAIT_TIMEOUT_SEC):
            self.reset_request_failed = True
            self.get_logger().error(
                "task_fail service not available; cannot request a reset goal"
            )
            return

        req = RingGoal.Request()
        req.robot_pose_x = self.robot_x
        req.robot_pose_y = self.robot_y
        req.radius = float(numpy.clip(self.difficulty_radius, 0.5, 4))
        self.new_goal = False
        self._goal_reset_future = self.task_fail_client.call_async(req)

    def get_state(self, action_linear_previous, action_angular_previous, check_terminal=True):
        state = copy.deepcopy(self.scan_ranges)
        state.append(float(numpy.clip((self.goal_distance / MAX_GOAL_DISTANCE), 0, 1)))
        state.append(float(self.goal_angle) / math.pi)
        state.append(float(action_linear_previous))
        state.append(float(action_angular_previous))
        self.local_step += 1

        if not check_terminal:
            state = numpy.asarray(state, dtype=numpy.float32)
            state = numpy.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
            return numpy.clip(state, self.observation_space.low, self.observation_space.high).astype(numpy.float32)

        if self.goal_distance < THREHSOLD_GOAL:
            self.succeed = SUCCESS
        elif self.obstacle_distance < THRESHOLD_COLLISION:
            dynamic_collision = False
            for obstacle_distance in self.obstacle_distances:
                if obstacle_distance < (THRESHOLD_COLLISION + OBSTACLE_RADIUS + 0.05):
                    dynamic_collision = True
            if dynamic_collision:
                self.succeed = COLLISION_OBSTACLE
            else:
                self.succeed = COLLISION_WALL
        elif self.time_sec >= self.episode_deadline:
            self.succeed = TIMEOUT
        elif self.robot_tilt > 0.06:
            self.succeed = TUMBLE

        if self.succeed != UNKNOWN:
            self.stop_reset_robot(self.succeed == SUCCESS)
        state = numpy.asarray(state, dtype=numpy.float32)
        state = numpy.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
        return numpy.clip(state, self.observation_space.low, self.observation_space.high).astype(numpy.float32)

    def _set_physics(self, paused):
        client = self.pause_physics_client if paused else self.unpause_physics_client
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"Gazebo {client.srv_name} service unavailable")
        future = client.call_async(Empty.Request())
        self._spin_until(future.done, 5.0)
        if not future.done() or future.result() is None:
            raise RuntimeError(f"Gazebo {client.srv_name} service timed out or failed")

    def _observation_ready(self, target):
        # A pre-reset header can be numerically larger than all new timestamps.
        # Bound inter-topic scheduling skew to one configured simulation step.
        return (not self._awaiting_rewind
                and self.time_sec >= target - 1e-9
                and target - 1e-9 <= self.odom_stamp <= self.time_sec + self.step_sim_time + 1e-9
                and target - 1e-9 <= self.scan_stamp <= self.time_sec + self.step_sim_time + 1e-9)

    def _timing_details(self, target):
        return (f"target={target:.9f}, clock={self.time_sec:.9f}, "
                f"odom={self.odom_stamp:.9f}, scan={self.scan_stamp:.9f}, "
                f"epoch={self._clock_epoch}, awaiting_rewind={self._awaiting_rewind}")

    def _wait_for_observation(self, target):
        epoch = self._clock_epoch
        self._spin_until(
            lambda: self._clock_epoch != epoch or self._observation_ready(target),
            self.reset_timeout_sec,
        )
        if self._clock_epoch != epoch:
            raise RuntimeError("Simulation clock reset during observation wait; "
                               + self._timing_details(target))
        if not self._observation_ready(target):
            raise RuntimeError("Timed out waiting for fresh /clock, /odom and /scan; "
                               + self._timing_details(target))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        self._needs_reset = True
        self.cmd_vel_pub.publish(Twist())
        try:
            self._set_physics(False)
            # Original task_fail acknowledges before reset_simulation completes.
            # Observe a nonzero clock first so its rewind can be detected even
            # on the first reset immediately after starting Gazebo.
            self._spin_until(lambda: self._clock_received and self.time_sec >= 0.2,
                             self.reset_timeout_sec)
            if not self._clock_received or self.time_sec < 0.2:
                raise RuntimeError("No pre-reset simulation clock; "
                                   + self._timing_details(0.2))
            epoch_before = self._clock_epoch
            expect_rewind = self._pending_outcome is not True
            self._awaiting_rewind = expect_rewind
            self._goal_reset_future = None
            if self._pending_outcome is not None:
                self._request_outcome_reset(self._pending_outcome)
            else:
                self._request_reset_goal()
            self._spin_until(
                lambda: self.new_goal and self._goal_reset_future is not None
                and self._goal_reset_future.done(), self.reset_timeout_sec,
            )
            if (not self.new_goal or self._goal_reset_future is None
                    or not self._goal_reset_future.done()
                    or self._goal_reset_future.result() is None):
                raise RuntimeError("Goal reset failed; refusing a stale initial observation")
            if expect_rewind:
                self._spin_until(lambda: self._clock_epoch > epoch_before,
                                 self.reset_timeout_sec)
                if self._clock_epoch <= epoch_before:
                    raise RuntimeError("Goal service returned but reset_simulation "
                                       "clock rewind was not observed; "
                                       + self._timing_details(self.time_sec))
            self._goal_reset_future = None
            self.odom_stamp = self.scan_stamp = -math.inf
            self._wait_for_observation(self.time_sec + self.step_sim_time)
            self._pending_outcome = None
        finally:
            self._set_physics(True)

        self.previous_action = numpy.zeros(2, dtype=numpy.float32)
        self.succeed = UNKNOWN
        self.done = False
        self.reset_deadline = False
        self.clock_msgs_skipped = 0
        self.total_distance = 0.0
        self.robot_x_prev, self.robot_y_prev = self.robot_x, self.robot_y
        self.local_step = 0
        episode_time = self.episode_timeout
        if ENABLE_DYNAMIC_GOALS:
            episode_time = float(numpy.clip(episode_time * self.difficulty_radius, 10, 50))
        self.episode_deadline = self.time_sec + episode_time
        self.initial_distance_to_goal = self.goal_distance
        rw.reward_initalize(self.initial_distance_to_goal)
        observation = self.get_state(0.0, 0.0, check_terminal=False)
        self._needs_reset = False
        info = self._get_info()
        info["reset_timeout"] = False
        return observation, info

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("Call reset() before step() or after an episode ends")
        action = numpy.asarray(action, dtype=numpy.float32)
        if action.shape != (2,) or not numpy.isfinite(action).all():
            raise ValueError("action must contain two finite values")
        action = numpy.clip(action, self.action_space.low, self.action_space.high)

        noisy_action = action.copy()
        if ENABLE_MOTOR_NOISE:
            noisy_action[LINEAR] += numpy.clip(self.np_random.normal(0, 0.05), -0.1, 0.1)
            noisy_action[ANGULAR] += numpy.clip(self.np_random.normal(0, 0.05), -0.1, 0.1)
            noisy_action = numpy.clip(noisy_action, self.action_space.low, self.action_space.high)

        if ENABLE_BACKWARD:
            action_linear = noisy_action[LINEAR] * SPEED_LINEAR_MAX
        else:
            action_linear = (noisy_action[LINEAR] + 1) / 2 * SPEED_LINEAR_MAX
        action_angular = noisy_action[ANGULAR] * SPEED_ANGULAR_MAX

        twist = Twist()
        twist.linear.x = float(action_linear)
        twist.angular.z = float(action_angular)
        self.cmd_vel_pub.publish(twist)

        start = self.time_sec
        step_epoch = getattr(self, "_clock_epoch", 0)
        try:
            self._set_physics(False)
            self._wait_for_observation(start + self.step_sim_time)
        except Exception:
            self._needs_reset = True
            self.cmd_vel_pub.publish(Twist())
            raise
        finally:
            self._set_physics(True)
        if getattr(self, "_clock_epoch", 0) != step_epoch:
            self._needs_reset = True
            self.cmd_vel_pub.publish(Twist())
            raise RuntimeError("Simulation clock reset before step completed; "
                               + self._timing_details(start + self.step_sim_time))
        self.last_step_sim_dt = self.time_sec - start

        # In s_(t+1), the previous action is the action just executed (a_t).
        observation = self.get_state(noisy_action[LINEAR], noisy_action[ANGULAR])
        reward = rw.get_reward(
            self.succeed,
            action_linear,
            action_angular,
            self.goal_distance,
            self.goal_angle,
            self.obstacle_distance,
        )

        terminated = self.done and self.succeed != TIMEOUT
        truncated = self.done and self.succeed == TIMEOUT
        info = self._get_info()
        info["distance_traveled"] = self.total_distance if self.done else 0.0

        if self.done:
            self._finish_episode_bookkeeping()

        self.previous_action = noisy_action.astype(numpy.float32)

        if self.local_step % 200 == 0:
            print(f"Rtot: {reward:<8.2f}GD: {self.goal_distance:<8.2f}GA: {math.degrees(self.goal_angle):.1f}\t", end="")
            print(f"MinD: {self.obstacle_distance:<8.2f}Alin: {action[LINEAR]:<7.1f}Aturn: {action[ANGULAR]:<7.1f}")

        return observation, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        if self.closed:
            return

        if rclpy.ok():
            self.cmd_vel_pub.publish(Twist())
        self._executor.remove_node(self)
        self.destroy_node()
        self.closed = True

    def _spin_once(self):
        self._executor.spin_once(timeout_sec=self.spin_timeout_sec)

    def _spin_until(self, condition, timeout_sec):
        start_time = time.monotonic()
        while not condition():
            self._spin_once()
            if time.monotonic() - start_time >= timeout_sec:
                break

    def _finish_episode_bookkeeping(self):
        self.succeed = UNKNOWN
        self.total_distance = 0.0
        self.local_step = 0
        self.done = False
        self.reset_deadline = True

    def _get_info(self):
        return {
            "success": self.succeed,
            "is_success": self.succeed == SUCCESS,
            "goal_distance": self.goal_distance,
            "goal_angle": self.goal_angle,
            "obstacle_distance": self.obstacle_distance,
            "robot_pose": (self.robot_x, self.robot_y),
            "reset_request_failed": self.reset_request_failed,
            "time_sec": self.time_sec,
            "step_sim_dt": self.last_step_sim_dt,
            "odom_stamp": self.odom_stamp,
            "scan_stamp": self.scan_stamp,
            "clock_epoch": self._clock_epoch,
        }


def make_env(args=None, **kwargs):
    if not rclpy.ok():
        rclpy.init(args=args)
    return DRLGymnasiumEnvironment(**kwargs)


def main(args=sys.argv[1:]):
    if not rclpy.ok():
        rclpy.init(args=args)
    env = DRLGymnasiumEnvironment()
    try:
        observation, info = env.reset()
        print(f"Gymnasium environment ready. observation_shape={observation.shape}, info={info}")
        while rclpy.ok():
            env._spin_once()
    finally:
        env.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
