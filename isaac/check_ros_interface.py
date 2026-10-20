"""Check the Isaac runtime's ROS 2 interface from outside, as the env sees it.

Run with the system ROS 2 Python while isaac/turtlebot3_isaac_sim.py is up:
    --scene test_walls   timing, topics and LaserScan against the test walls
    --scene empty        timing, topics and the open-loop drive vs Gazebo
"""
import argparse
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

# Same protocol in Gazebo (empty world, lab): moved / turned.
GAZEBO = {(0.2, 0.0, 3.0): (0.574, 0.0), (0.0, 1.0, 3.0): (0.004, 167.7), (0.1, 0.5, 4.0): (0.329, 110.8)}


def stamp(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


class Check(Node):
    def __init__(self):
        super().__init__("check_isaac_ros")
        self.now = None
        self.odom = self.scan = None
        self.counts = {"clock": 0, "odom": 0, "scan": 0}
        self.create_subscription(Clock, "/clock", self.on_clock, qos_profile_sensor_data)
        self.create_subscription(Odometry, "odom", self.on_odom, qos_profile_sensor_data)
        self.create_subscription(LaserScan, "scan", self.on_scan, qos_profile_sensor_data)
        self.pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.pause_cli = self.create_client(Empty, "/pause_physics")
        self.unpause_cli = self.create_client(Empty, "/unpause_physics")

    def on_clock(self, m):
        self.now = m.clock.sec + m.clock.nanosec * 1e-9
        self.counts["clock"] += 1

    def on_odom(self, m):
        self.odom = m
        self.counts["odom"] += 1

    def on_scan(self, m):
        self.scan = m
        self.counts["scan"] += 1

    def call(self, client):
        assert client.wait_for_service(timeout_sec=10.0), client.srv_name
        f = client.call_async(Empty.Request())
        while not f.done():
            rclpy.spin_once(self, timeout_sec=0.05)

    def spin_wall(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.01)

    def pose(self):
        p, q = self.odom.pose.pose.position, self.odom.pose.pose.orientation
        return p.x, p.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def hold(self, v, w, seconds):
        t_end, next_cmd = self.now + seconds, self.now
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = float(v), float(w)
        while self.now < t_end:
            if self.now >= next_cmd:
                self.pub.publish(cmd)
                next_cmd += 0.02
            rclpy.spin_once(self, timeout_sec=0.01)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["test_walls", "empty"], required=True)
    scene = parser.parse_args().scene
    rclpy.init()
    n = Check()
    ok = True

    def report(label, passed, detail):
        nonlocal ok
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] {label}: {detail}", flush=True)

    # 1. Starts paused: no clock progress, then unpause makes time advance in real time.
    n.call(n.pause_cli)
    n.spin_wall(1.0)
    t0, c0 = n.now, n.counts["clock"]
    n.spin_wall(1.0)
    report("starts paused", n.now is None, f"{n.counts['clock']} clock messages before the first unpause")
    n.call(n.unpause_cli)
    n.spin_wall(0.5)
    w0, s0, counts0 = time.monotonic(), n.now, dict(n.counts)
    n.spin_wall(3.0)
    sim, wall = n.now - s0, time.monotonic() - w0
    rates = {k: (n.counts[k] - counts0[k]) / sim for k in counts0}
    report("real-time factor", 0.9 < sim / wall < 1.1, f"{sim / wall:.3f} ({sim:.2f} s sim in {wall:.2f} s wall)")
    report("clock rate", 9 <= rates["clock"] <= 11, f"{rates['clock']:.1f} msgs per sim second (Gazebo: 10)")
    report("odom / scan rate", all(45 < rates[k] < 55 for k in ("odom", "scan")),
           f"odom {rates['odom']:.1f} Hz, scan {rates['scan']:.1f} Hz (sim time)")
    lag = n.now - stamp(n.scan.header)
    report("scan stamp is sim time", abs(lag) < 0.1, f"clock - scan stamp = {lag:.3f} s")

    # 2. LaserScan metadata and the known walls (robot at origin facing +x).
    s = n.scan
    meta = (len(s.ranges) == 40 and abs(s.angle_min) < 1e-6 and abs(s.angle_max - 6.28) < 1e-5
            and abs(s.angle_increment - 6.28 / 39) < 1e-6 and abs(s.range_min - 0.12) < 1e-6
            and abs(s.range_max - 3.5) < 1e-6)
    report("LaserScan layout", meta, f"{len(s.ranges)} rays, {s.angle_min:.3f}..{s.angle_max:.3f}, "
           f"inc {s.angle_increment:.5f}, range {s.range_min}..{s.range_max}")
    if scene == "test_walls":
        report("front / left wall ranges", abs(s.ranges[0] - 0.558) < 0.04 and abs(s.ranges[10] - 0.6255) < 0.04,
               f"front {s.ranges[0]:.3f} (0.558), left {s.ranges[10]:.3f} (0.626), "
               f"misses {sum(math.isinf(r) for r in s.ranges)}")
    else:
        report("empty scene has no returns", all(math.isinf(r) for r in s.ranges),
               f"{sum(math.isinf(r) for r in s.ranges)}/40 rays without a return")

    # 3. Pause stops time; unpause continues from the same time.
    n.call(n.pause_cli)
    n.spin_wall(0.3)
    p0 = n.now
    n.spin_wall(1.0)
    n.call(n.unpause_cli)
    n.spin_wall(0.5)
    report("pause/unpause", n.now - p0 < 0.6, f"paused at {p0:.3f}, 0.5 s wall after unpause at {n.now:.3f}")

    # 4. Same open-loop protocol as Gazebo.
    for (v, w, seconds), (gz_d, gz_turn) in (GAZEBO.items() if scene == "empty" else ()):
        n.hold(0.0, 0.0, 0.5)
        x0, y0, a0 = n.pose()
        n.hold(v, w, seconds)
        x1, y1, a1 = n.pose()
        d = math.hypot(x1 - x0, y1 - y0)
        turn = math.degrees(math.remainder(a1 - a0, 2 * math.pi))
        close = abs(d - gz_d) < 0.015 and abs(turn - gz_turn) < 3.0
        report(f"cmd v={v} w={w} {seconds:.0f}s vs Gazebo", close,
               f"Isaac {d:.3f} m / {turn:.1f} deg, Gazebo {gz_d:.3f} m / {gz_turn:.1f} deg")
    n.hold(0.0, 0.0, 0.3)
    n.call(n.pause_cli)
    print("ALL PASS" if ok else "SOME CHECKS FAILED", flush=True)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
