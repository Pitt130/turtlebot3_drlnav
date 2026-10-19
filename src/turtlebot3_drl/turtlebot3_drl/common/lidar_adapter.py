"""Adapt a full-circle LaserScan to the existing policy's angular bins."""

import math
import numpy as np

# Slack for float32 ranges that sit exactly on the published range limits.
RANGE_TOLERANCE = 1e-6


def clean_ranges(ranges, cap, range_min=None, range_max=None):
    """Map LaserScan ranges to [0, cap] with REP-117 semantics.

    NaN and +inf (no return) become the cap; -inf (closer than range_min)
    becomes 0. Finite values outside [range_min, range_max] are discarded as
    the cap, as the LaserScan definition requires; some drivers report a
    missing return as 0.0, which must not read as a collision.
    """
    values = np.asarray(ranges, dtype=np.float64)
    low = 0.0 if range_min is None else float(range_min)
    high = math.inf if range_max is None else float(range_max)
    if not math.isfinite(low) or low < 0 or math.isnan(high) or high <= low:
        raise ValueError(f"Invalid LaserScan range limits: range_min={range_min}, range_max={range_max}")
    valid = np.isfinite(values) & (values >= low - RANGE_TOLERANCE) & (values <= high + RANGE_TOLERANCE)
    cleaned = np.where(valid, values, cap)
    cleaned = np.where(values == -np.inf, 0.0, cleaned)
    return np.clip(cleaned, 0.0, cap)


def policy_scan(ranges, angle_min, angle_increment, count, cap, range_min=None, range_max=None):
    """Return distances in bins starting at +X, counterclockwise.

    The policy was trained on Gazebo rays at linspace(0, 6.28, count), one ray
    per direction. Equal-size legacy scans at the original angles pass through
    unchanged; denser full-circle scans contribute the ray nearest to each
    legacy angle. Partial or sparse scans are rejected rather than inventing
    free space. Range values are cleaned by clean_ranges().
    """
    values = np.asarray(ranges, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or count < 2:
        raise ValueError("A full-circle scan and at least two policy bins are required")
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("LiDAR cap must be positive and finite")
    if not math.isfinite(angle_min) or not math.isfinite(angle_increment) or angle_increment == 0:
        raise ValueError("Invalid LaserScan angular metadata")
    step = abs(angle_increment)
    coverage = step * (values.size - 1)
    # Permit an exclusive final ray and the legacy 6.28-radian endpoint.
    if coverage < 2 * math.pi - step - 0.01 or coverage > 2 * math.pi + 0.01:
        raise ValueError("Expected one complete 360-degree LaserScan")
    angles = angle_min + np.arange(values.size) * angle_increment
    values = clean_ranges(values, cap, range_min, range_max)
    legacy_angles = np.linspace(0.0, 6.28, count)
    # Preserve the old SDF sampling, including its near-duplicate endpoint.
    if values.size == count and np.allclose(angles, legacy_angles, atol=0.005, rtol=0):
        return values.astype(np.float32)
    # Taking the minimum over each angular sector would widen obstacles and
    # change the pretrained policy's actions, so sample like Gazebo instead.
    if step > 2 * math.pi / (count - 1):
        raise ValueError("Scan resolution cannot populate all policy directions")
    offsets = np.mod(legacy_angles[:, None] - angles[None, :] + math.pi, 2 * math.pi) - math.pi
    return values[np.argmin(np.abs(offsets), axis=1)].astype(np.float32)
