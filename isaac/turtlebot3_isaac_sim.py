"""Isaac Sim runtime for the TurtleBot3 DRL environment, Gazebo-compatible ROS 2 side.

Publishes /clock at 10 Hz and /odom, /scan at 50 Hz, the rates Gazebo uses, drives the
wheels from /cmd_vel, and serves /pause_physics and /unpause_physics. It starts
paused, like the Gazebo launch with pause:=true. Run after sourcing Jazzy:

    source /opt/ros/jazzy/setup.bash
    OMNI_KIT_ACCEPT_EULA=YES ~/isaacsim/.venv/bin/python isaac/turtlebot3_isaac_sim.py \
        --usd ~/isaac_assets/turtlebot3_burger_usd/turtlebot3_burger_isaac/turtlebot3_burger_isaac.usda
"""
import argparse
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

PHYSICS_HZ = 200.0
SENSOR_HZ = 50.0                 # Gazebo diff drive and LDS update_rate
# gazebo_ros_init publishes /clock at 10 Hz. The env reads time from /clock, so
# its 0.1 s steps start on this grid and absorb the pause-call latency; with a
# faster clock every step ran ~4.5% long (pause round trip at real time).
CLOCK_HZ = 10.0
WHEEL_ACCEL = 1.0                # m/s^2 at the wheel rim, gazebo_ros_diff_drive max_wheel_acceleration
BASE_LINK_HEIGHT = 0.010         # base_footprint -> base_link fixed joint
BASE_SCAN = "/World/burger/Geometry/base_footprint/base_link/base_scan"
TEST_WALLS = {"wall_x": ((0.6, 0.0), (0.15, 1.0)), "wall_y": ((0.0, 0.7), (1.0, 0.15))}


class DiffDrive:
    """gazebo_ros_diff_drive: rim speeds ramp toward (v -/+ w L/2) at WHEEL_ACCEL."""

    def __init__(self, radius, separation, accel=WHEEL_ACCEL):
        self.radius, self.separation, self.accel = radius, separation, accel
        self.target = [0.0, 0.0]
        self.rim = [0.0, 0.0]

    def command(self, v, w):
        half = self.separation / 2
        self.target = [v - w * half, v + w * half]

    def stop(self):
        self.target = [0.0, 0.0]
        self.rim = [0.0, 0.0]

    def step(self, dt):
        limit = self.accel * dt if self.accel > 0 else math.inf
        for i in (0, 1):
            self.rim[i] += max(-limit, min(limit, self.target[i] - self.rim[i]))
        return [s / self.radius for s in self.rim]   # wheel joint velocities, rad/s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", required=True)
    parser.add_argument("--scene", choices=["empty", "test_walls"], default="empty")
    parser.add_argument("--physics-hz", type=float, default=PHYSICS_HZ)
    parser.add_argument("--sensor-hz", type=float, default=SENSOR_HZ)
    parser.add_argument("--clock-hz", type=float, default=CLOCK_HZ)
    parser.add_argument("--rtf", type=float, default=1.0, help="real-time factor cap; 0 = as fast as possible")
    parser.add_argument("--start-unpaused", action="store_true")
    # With the 1 m/s^2 rim ramp these match Gazebo within 1% (check_ros_interface.py).
    parser.add_argument("--wheel-friction", type=float, default=1.0)
    parser.add_argument("--caster-friction", type=float, default=0.0)
    args = parser.parse_args()

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})

    import numpy as np
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import LaserScan
    from std_srvs.srv import Empty

    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    from build_turtlebot3 import (WHEEL_JOINTS, WHEEL_RADIUS, WHEEL_SEPARATION, apply_contact_materials,
                                  use_sphere_wheel_collisions)
    from turtlebot3_lidar import RANGE_MAX, RANGE_MIN, TurtleBot3Lidar

    world = World(physics_dt=1 / args.physics_hz, rendering_dt=1 / args.physics_hz)
    world.scene.add_default_ground_plane()
    prim = add_reference_to_stage(usd_path=args.usd, prim_path="/World/burger")
    prim.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
    use_sphere_wheel_collisions(world.stage, "/World/burger")
    apply_contact_materials(world.stage, "/World/burger", args.wheel_friction, args.caster_friction)
    if args.scene == "test_walls":
        for name, ((cx, cy), (sx, sy)) in TEST_WALLS.items():
            world.scene.add(FixedCuboid(prim_path=f"/World/{name}", name=name, position=np.array([cx, cy, 0.25]),
                                        scale=np.array([sx, sy, 0.5])))
    robot = world.scene.add(SingleArticulation(prim_path="/World/burger/Geometry/base_footprint/base_link",
                                               name="burger"))
    lidar = TurtleBot3Lidar(BASE_SCAN)
    world.reset()
    wheels = [robot.get_dof_index(n) for n in WHEEL_JOINTS]
    drive = DiffDrive(WHEEL_RADIUS, WHEEL_SEPARATION)

    rclpy.init()
    node = Node("isaac_turtlebot3")
    state = {"playing": args.start_unpaused, "sim_time": 0.0, "steps": 0}
    clock_pub = node.create_publisher(Clock, "/clock", QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
    odom_pub = node.create_publisher(Odometry, "odom", qos_profile_sensor_data)
    scan_pub = node.create_publisher(LaserScan, "scan", qos_profile_sensor_data)
    node.create_subscription(Twist, "cmd_vel", lambda m: drive.command(m.linear.x, m.angular.z), 10)

    def set_playing(playing):
        def callback(request, response):
            state["playing"] = playing
            return response
        return callback

    node.create_service(Empty, "/pause_physics", set_playing(False))
    node.create_service(Empty, "/unpause_physics", set_playing(True))

    def stamp(msg):
        t = state["sim_time"]
        msg.header.stamp.sec = int(t)
        msg.header.stamp.nanosec = int(round((t - int(t)) * 1e9)) % 1000000000

    def publish_clock():
        msg = Clock()
        t = state["sim_time"]
        msg.clock.sec, msg.clock.nanosec = int(t), int(round((t - int(t)) * 1e9)) % 1000000000
        clock_pub.publish(msg)

    def publish_odom():
        pos, quat = robot.get_world_pose()          # base_link; base_footprint differs only in z
        w, x, y, z = (float(q) for q in quat)
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        lin = robot.get_linear_velocity()
        ang = robot.get_angular_velocity()
        msg = Odometry()
        stamp(msg)
        msg.header.frame_id, msg.child_frame_id = "odom", "base_footprint"
        p = msg.pose.pose.position
        p.x, p.y, p.z = float(pos[0]), float(pos[1]), float(pos[2]) - BASE_LINK_HEIGHT
        o = msg.pose.pose.orientation
        o.w, o.x, o.y, o.z = w, x, y, z
        # Twist in the body frame, as gazebo_ros_diff_drive reports it.
        msg.twist.twist.linear.x = float(math.cos(yaw) * lin[0] + math.sin(yaw) * lin[1])
        msg.twist.twist.linear.y = float(-math.sin(yaw) * lin[0] + math.cos(yaw) * lin[1])
        msg.twist.twist.angular.z = float(ang[2])
        odom_pub.publish(msg)

    def publish_scan():
        ranges, _ = lidar.ranges()
        msg = LaserScan()
        stamp(msg)
        msg.header.frame_id = "base_scan"
        msg.angle_min, msg.angle_max, msg.angle_increment = lidar.angle_min, lidar.angle_max, lidar.angle_increment
        msg.scan_time, msg.time_increment = sensor_every * dt, 0.0
        msg.range_min, msg.range_max = RANGE_MIN, RANGE_MAX
        msg.ranges = [float(r) for r in ranges]
        scan_pub.publish(msg)

    sensor_every = max(1, int(round(args.physics_hz / args.sensor_hz)))
    clock_every = max(1, int(round(args.physics_hz / args.clock_hz)))
    dt = 1.0 / args.physics_hz
    print(f"isaac_turtlebot3 ready: scene={args.scene}, physics {args.physics_hz:.0f} Hz, "
          f"sensors {args.physics_hz / sensor_every:.0f} Hz, clock {args.physics_hz / clock_every:.0f} Hz, "
          f"rtf cap {args.rtf or 'none'}, {'running' if state['playing'] else 'paused'}", flush=True)
    publish_clock()
    wall_start, sim_start = time.monotonic(), 0.0
    try:
        while app.is_running():
            rclpy.spin_once(node, timeout_sec=0.0)
            if not state["playing"]:
                time.sleep(0.001)
                wall_start, sim_start = time.monotonic(), state["sim_time"]
                continue
            targets = [0.0] * robot.num_dof
            targets[wheels[0]], targets[wheels[1]] = drive.step(dt)
            robot.apply_action(ArticulationAction(joint_velocities=targets))
            world.step(render=False)
            state["steps"] += 1
            state["sim_time"] = state["steps"] * dt
            if state["steps"] % clock_every == 0:
                publish_clock()
            if state["steps"] % sensor_every == 0:
                publish_odom()
                publish_scan()
            if args.rtf > 0:
                # Serve ROS while throttling, so a pause requested after this
                # step's sensors stops time here instead of one step later; the
                # env's 0.1 s steps then stay aligned with the 50 Hz sensor ticks.
                end = wall_start + (state["sim_time"] - sim_start) / args.rtf
                while state["playing"] and time.monotonic() < end:
                    rclpy.spin_once(node, timeout_sec=max(0.0, end - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        app.close()


if __name__ == "__main__":
    main()
