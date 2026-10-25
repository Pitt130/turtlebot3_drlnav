"""TurtleBot3 LDS emulation with Isaac's physics raycast sensor.

Mirrors the Gazebo ray sensor in turtlebot3_burger/model.sdf: rays in the
base_scan plane, range 0.12-3.5 m, Gaussian noise (stddev 0.01) clamped to the
range, no return reported as +inf (REP-117), 50 Hz.

LowRayLayer adds a second, world-horizontal set of rays close to the floor for
furnished scenes: sofa, bed and cabinet bases sit below the 17 cm scan plane, so
the LDS alone lets the robot push against them until the episode times out.
Combined per ray with min(), the scan keeps its 40-ray layout; in the box
arenas (full-height walls) it is unchanged.
"""
import math

import numpy as np

RANGE_MIN = 0.12
RANGE_MAX = 3.5
NOISE_STDDEV = 0.01
UPDATE_HZ = 50.0
# Gazebo's layout: 40 rays from 0 to 6.28 rad, both ends included.
GAZEBO_ANGLE_MIN, GAZEBO_ANGLE_MAX, GAZEBO_SAMPLES = 0.0, 6.28, 40


def ray_angles(samples=GAZEBO_SAMPLES, angle_min=GAZEBO_ANGLE_MIN, angle_max=GAZEBO_ANGLE_MAX):
    return np.linspace(angle_min, angle_max, samples)


class TurtleBot3Lidar:
    def __init__(self, base_scan_path, samples=GAZEBO_SAMPLES, angle_min=GAZEBO_ANGLE_MIN,
                 angle_max=GAZEBO_ANGLE_MAX, noise_stddev=NOISE_STDDEV, seed=None):
        from isaacsim.sensors.experimental.physics import Raycast, RaycastSensor
        self.angles = ray_angles(samples, angle_min, angle_max)
        self.angle_min, self.angle_max = float(angle_min), float(angle_max)
        self.angle_increment = (angle_max - angle_min) / (samples - 1)
        directions = np.stack([np.cos(self.angles), np.sin(self.angles), np.zeros(samples)], axis=1)
        self.sensor = RaycastSensor(Raycast.create(
            f"{base_scan_path}/lds_raycast",
            ray_origins=np.zeros((samples, 3)), ray_directions=directions,
            min_range=RANGE_MIN, max_range=RANGE_MAX, output_frame="SENSOR",
        ))
        self.noise_stddev = float(noise_stddev)
        self.rng = np.random.default_rng(seed)
        self.low = None

    def raw(self):
        """Depths as reported by the sensor plus hit positions, for checks."""
        frame = self.sensor.get_data()
        return np.asarray(frame["depths"], dtype=np.float64), frame

    def ranges(self):
        """LaserScan ranges with Gazebo semantics (plus the low layer when attached)."""
        depths, frame = self.raw()
        if self.low is not None:
            # A low obstacle closer than range_min is already under the robot front (a shoe the
            # wheels ride onto); report it at range_min so it reads as contact, not as no return.
            low = self.low.depths(self.angles)
            depths = np.minimum(depths, np.where(np.isfinite(low), np.maximum(low, RANGE_MIN), np.inf))
        hit = np.isfinite(depths) & (depths >= RANGE_MIN) & (depths < RANGE_MAX)
        out = np.full(depths.shape, math.inf)
        noisy = depths[hit] + (self.rng.normal(0.0, self.noise_stddev, hit.sum()) if self.noise_stddev else 0.0)
        out[hit] = np.clip(noisy, RANGE_MIN, RANGE_MAX)
        return out, float(frame["time"])


class LowRayLayer:
    """World-horizontal raycasts at a fixed height above the floor, from the scan origin.

    Cast with PhysX scene queries rather than a sensor on the robot, so a robot
    tilted by a rug edge does not see the floor as a wall. Hits on the robot
    itself are ignored.
    """

    def __init__(self, robot, floor_z, height=0.06, scan_offset=-0.032, ignore_prefix="/World/burger"):
        from omni.physx import get_physx_scene_query_interface
        self.robot, self.z, self.scan_offset, self.ignore = robot, floor_z + height, scan_offset, ignore_prefix
        self.pq = get_physx_scene_query_interface()

    def depths(self, angles):
        from pxr import Gf
        pos, quat = self.robot.get_world_pose()
        w, x, y, z = (float(q) for q in quat)
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        ox, oy = float(pos[0]) + self.scan_offset * math.cos(yaw), float(pos[1]) + self.scan_offset * math.sin(yaw)
        origin = Gf.Vec3f(ox, oy, self.z)
        out = np.full(len(angles), math.inf)
        for i, a in enumerate(angles):
            d = Gf.Vec3f(math.cos(yaw + a), math.sin(yaw + a), 0.0)
            r = self.pq.raycast_closest(origin, d, RANGE_MAX)
            if r["hit"] and not (str(r.get("rigidBody") or "").startswith(self.ignore)
                                 or str(r.get("collision") or "").startswith(self.ignore)):
                out[i] = r["distance"]
        return out
