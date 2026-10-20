"""TurtleBot3 LDS emulation with Isaac's physics raycast sensor.

Mirrors the Gazebo ray sensor in turtlebot3_burger/model.sdf: rays in the
base_scan plane, range 0.12-3.5 m, Gaussian noise (stddev 0.01) clamped to the
range, no return reported as +inf (REP-117), 50 Hz.
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

    def raw(self):
        """Depths as reported by the sensor plus hit positions, for checks."""
        frame = self.sensor.get_data()
        return np.asarray(frame["depths"], dtype=np.float64), frame

    def ranges(self):
        """LaserScan ranges with Gazebo semantics."""
        depths, frame = self.raw()
        hit = np.isfinite(depths) & (depths >= RANGE_MIN) & (depths < RANGE_MAX)
        out = np.full(depths.shape, math.inf)
        noisy = depths[hit] + (self.rng.normal(0.0, self.noise_stddev, hit.sum()) if self.noise_stddev else 0.0)
        out[hit] = np.clip(noisy, RANGE_MIN, RANGE_MAX)
        return out, float(frame["time"])
