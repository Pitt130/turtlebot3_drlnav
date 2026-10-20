"""Check a running Isaac stage from ROS 2 against the Gazebo stage geometry.

Start turtlebot3_isaac_sim.py --stage N first, then (system ROS 2 Python):
    python3 isaac/check_stage.py --stage N [--reference isaac/reference/gazebo_stageN_obstacles.csv]

- /obstacle/odom: frame ids, rate and positions against the Gazebo animation
  (and against a recorded Gazebo trajectory when --reference is given).
- /scan: every ray against analytic ranges to the stage's walls and to the
  obstacles at the scan's timestamp.
"""
import argparse
import bisect
import csv
import math
from pathlib import Path
import re
import sys
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_scene import load_stage  # noqa: E402

SCAN_OFFSET = -0.032  # base_scan relative to base_footprint, along +x


def ray_box(ox, oy, dx, dy, box):
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    rx, ry = ox - box.center[0], oy - box.center[1]
    lx, ly = c * rx + s * ry, -s * rx + c * ry          # origin in box frame
    ldx, ldy = c * dx + s * dy, -s * dx + c * dy
    t0, t1 = -math.inf, math.inf
    for o, d, h in ((lx, ldx, box.size[0] / 2), (ly, ldy, box.size[1] / 2)):
        if abs(d) < 1e-12:
            if abs(o) > h:
                return math.inf
            continue
        a, b = sorted(((-h - o) / d, (h - o) / d))
        t0, t1 = max(t0, a), min(t1, b)
    return t0 if t0 <= t1 and t0 > 0 else math.inf


def ray_circle(ox, oy, dx, dy, cx, cy, r):
    fx, fy = ox - cx, oy - cy
    b = fx * dx + fy * dy
    disc = b * b - (fx * fx + fy * fy - r * r)
    if disc < 0:
        return math.inf
    t = -b - math.sqrt(disc)
    return t if t > 0 else math.inf


class Probe(Node):
    def __init__(self):
        super().__init__("check_stage")
        self.now = None
        self.odom = None
        self.scans = []
        self.obstacle_msgs = []
        self.create_subscription(Clock, "/clock", lambda m: setattr(self, "now", m.clock.sec + m.clock.nanosec * 1e-9),
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, "odom", lambda m: setattr(self, "odom", m), qos_profile_sensor_data)
        self.create_subscription(LaserScan, "scan", lambda m: self.scans.append((m, self.odom)), qos_profile_sensor_data)
        self.create_subscription(Odometry, "/obstacle/odom", self.obstacle_msgs.append, 50)
        self.unpause = self.create_client(Empty, "/unpause_physics")
        self.pause = self.create_client(Empty, "/pause_physics")

    def call(self, client):
        assert client.wait_for_service(timeout_sec=10.0)
        f = client.call_async(Empty.Request())
        while not f.done():
            rclpy.spin_once(self, timeout_sec=0.05)


def stamp(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--reference")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    args = parser.parse_args()
    stage = load_stage(args.repo, args.stage)
    by_frame = {o.frame_id: o for o in stage.obstacles}
    ok = True

    def report(label, passed, detail):
        nonlocal ok
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] {label}: {detail}", flush=True)

    rclpy.init()
    n = Probe()
    n.call(n.unpause)
    end = time.monotonic() + args.seconds + 1.0
    while time.monotonic() < end:
        rclpy.spin_once(n, timeout_sec=0.01)
    n.call(n.pause)

    # Obstacles: names the env can parse, rate, and positions.
    frames = sorted({m.child_frame_id for m in n.obstacle_msgs})
    ids = sorted(int(re.search(r"obstacle(\d+)", f).group(1)) for f in frames)
    report("obstacle frame ids", frames == sorted(by_frame),
           f"{frames} -> env obstacle ids {ids}")
    span = max(stamp(m.header) for m in n.obstacle_msgs) - min(stamp(m.header) for m in n.obstacle_msgs)
    rate = len(n.obstacle_msgs) / max(len(frames), 1) / span
    report("obstacle odom rate", 45 < rate < 55, f"{rate:.1f} Hz per obstacle (Gazebo p3d: 50)")
    err = max(math.hypot(m.pose.pose.position.x - by_frame[m.child_frame_id].pose(stamp(m.header))[0][0],
                         m.pose.pose.position.y - by_frame[m.child_frame_id].pose(stamp(m.header))[0][1])
              for m in n.obstacle_msgs)
    report("obstacle odom vs animation", err < 1e-3, f"max {err * 1000:.2f} mm over {len(n.obstacle_msgs)} msgs")
    if args.reference:
        ref = {}
        for row in csv.DictReader(open(args.reference)):
            ref.setdefault(row["child"], []).append((float(row["t"]), float(row["x"]), float(row["y"])))
        worst = 0.0
        for m in n.obstacle_msgs:
            rows = ref[m.child_frame_id]
            t = stamp(m.header)
            k = bisect.bisect_left(rows, (t,))
            near = min(rows[max(0, k - 1):k + 1], key=lambda r: abs(r[0] - t))
            if abs(near[0] - t) < 0.011:
                worst = max(worst, math.hypot(m.pose.pose.position.x - near[1], m.pose.pose.position.y - near[2]))
        report("obstacle odom vs recorded Gazebo", worst < 0.02, f"max {worst * 1000:.2f} mm")

    # Scans: every ray of every scan against walls and obstacles at the scan time.
    errors, mismatches, rays = [], 0, 0
    for scan, odom in n.scans:
        if odom is None or abs(stamp(odom.header) - stamp(scan.header)) > 1e-6:
            continue
        p, q = odom.pose.pose.position, odom.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        ox, oy = p.x + SCAN_OFFSET * math.cos(yaw), p.y + SCAN_OFFSET * math.sin(yaw)
        t = stamp(scan.header)
        circles = [(o.pose(t)[0][0], o.pose(t)[0][1], o.radius) for o in stage.obstacles]
        for i, measured in enumerate(scan.ranges):
            a = yaw + scan.angle_min + i * scan.angle_increment
            dx, dy = math.cos(a), math.sin(a)
            hits = [ray_box(ox, oy, dx, dy, b) for b in stage.boxes] + \
                   [ray_circle(ox, oy, dx, dy, cx, cy, r) for cx, cy, r in circles]
            expected = min([h for h in hits if h >= scan.range_min] + [math.inf])
            expected = expected if expected < scan.range_max else math.inf
            rays += 1
            if math.isinf(expected) != math.isinf(measured):
                mismatches += 1
            elif math.isfinite(expected):
                errors.append(abs(measured - expected))
    if errors:
        within = sum(e < 0.04 for e in errors) / len(errors)
        report("scan vs stage geometry", within > 0.999 and mismatches <= rays * 0.002,
               f"{len(n.scans)} scans, {rays} rays: {100 * within:.2f}% of hits within 4 cm (noise 1 cm), "
               f"mean |err| {1000 * sum(errors) / len(errors):.1f} mm, hit/miss mismatches {mismatches}")
    else:
        report("scan vs stage geometry", False, "no scans with a matching odom stamp")
    print("STAGE CHECK PASS" if ok else "STAGE CHECK FAIL", flush=True)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
