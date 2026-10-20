"""Drive the real Gymnasium environment (Isaac backend) against the Isaac runtime.

Start isaac/turtlebot3_isaac_sim.py --scene test_walls first. The task services
are stubbed here (publish a fixed goal, then reply) until the Isaac task
controller exists; the robot is not reset. Full forward drives into the wall
0.5 m ahead, which must end the episode as a wall collision.

    DRLNAV_SIM_BACKEND=isaac DRLNAV_STAGE=4 .venv/bin/python isaac/check_gym_env.py
"""
import math
import threading

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from turtlebot3_msgs.srv import RingGoal

from turtlebot3_drl.common.settings import COLLISION_WALL
from turtlebot3_drl.drl_environment.drl_environment_gymnasium import make_env

GOAL = (2.0, 0.0)


class StubTasks(Node):
    def __init__(self):
        super().__init__("stub_tasks")
        self.goal_pub = self.create_publisher(Pose, "goal_pose", 10)
        self.create_service(RingGoal, "task_fail", self.on_task)
        self.create_service(RingGoal, "task_succeed", self.on_task)

    def on_task(self, request, response):
        goal = Pose()
        goal.position.x, goal.position.y = GOAL
        self.goal_pub.publish(goal)
        return response


def main():
    rclpy.init()
    stub = StubTasks()
    executor = SingleThreadedExecutor()
    executor.add_node(stub)
    threading.Thread(target=executor.spin, daemon=True).start()

    env = make_env()
    try:
        obs, info = env.reset()
        print(f"reset: obs shape {obs.shape}, dtype {obs.dtype}, sim time {info['time_sec']:.3f} s, "
              f"goal distance {info['goal_distance']:.3f} m, scan[0] {obs[0]:.3f} (front, normalized)", flush=True)
        dts, steps, times = [], 0, [info["time_sec"]]
        while True:
            obs, reward, terminated, truncated, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
            steps += 1
            dts.append(info["step_sim_dt"])
            times.append(info["time_sec"])
            if steps % 5 == 0 or terminated or truncated:
                print(f"step {steps:2d}: t={info['time_sec']:.3f}s dt={info['step_sim_dt']:.3f} "
                      f"x={info['robot_pose'][0]:.3f} obstacle={info['obstacle_distance']:.3f} m "
                      f"scan[0]={obs[0]:.3f} goal_dist={info['goal_distance']:.3f} reward={reward:.2f}", flush=True)
            if terminated or truncated or steps >= 80:
                break
        print(f"RESULT: {steps} steps, terminated={terminated}, truncated={truncated}, "
              f"outcome={info['success']} (COLLISION_WALL={COLLISION_WALL}), "
              f"step_sim_dt mean {np.mean(dts):.4f} min {min(dts):.4f} max {max(dts):.4f}; "
              f"control period (obs to obs) mean {np.mean(np.diff(times)):.4f} max {np.max(np.diff(times)):.4f}",
              flush=True)
        ok = (terminated and info["success"] == COLLISION_WALL and obs.shape == (44,)
              and all(abs(dt - 0.1) < 0.011 for dt in dts))
        print("GYM CHECK PASS" if ok else "GYM CHECK FAIL", flush=True)
    finally:
        env.close()
        executor.shutdown()
        stub.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
