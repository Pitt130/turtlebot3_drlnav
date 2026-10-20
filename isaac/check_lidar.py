"""Compare the emulated LDS with analytic ranges to two known walls.

    ~/isaacsim/.venv/bin/python isaac/check_lidar.py --usd <burger.usda> [--render <png>]
"""
import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

WALLS = {  # name: (center xy, size xy), 0.5 m tall
    "wall_x": ((0.6, 0.0), (0.15, 1.0)),
    "wall_y": ((0.0, 0.7), (1.0, 0.15)),
}


def analytic_range(origin, angle, range_min, range_max):
    """Nearest axis-aligned wall hit along a ray, like a geometric ray sensor."""
    dx, dy = math.cos(angle), math.sin(angle)
    best = math.inf
    for (cx, cy), (sx, sy) in WALLS.values():
        lo = [cx - sx / 2 - origin[0], cy - sy / 2 - origin[1]]
        hi = [cx + sx / 2 - origin[0], cy + sy / 2 - origin[1]]
        t0, t1 = -math.inf, math.inf
        for d, a, b in ((dx, lo[0], hi[0]), (dy, lo[1], hi[1])):
            if abs(d) < 1e-12:
                if not a <= 0 <= b:
                    t0, t1 = math.inf, -math.inf
                continue
            ta, tb = sorted((a / d, b / d))
            t0, t1 = max(t0, ta), min(t1, tb)
        if t0 <= t1 and t0 >= range_min:
            best = min(best, t0)
    return best if best <= range_max else math.inf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", required=True)
    parser.add_argument("--render", type=Path)
    args = parser.parse_args()

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": 1280, "height": 720})

    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid, VisualSphere
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import UsdGeom
    from build_turtlebot3 import use_sphere_wheel_collisions
    from turtlebot3_lidar import RANGE_MAX, RANGE_MIN, TurtleBot3Lidar

    world = World(physics_dt=1 / 200, rendering_dt=1 / 60)
    world.scene.add_default_ground_plane()
    prim = add_reference_to_stage(usd_path=args.usd, prim_path="/World/burger")
    prim.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
    use_sphere_wheel_collisions(world.stage, "/World/burger")
    for name, ((cx, cy), (sx, sy)) in WALLS.items():
        world.scene.add(FixedCuboid(prim_path=f"/World/{name}", name=name, position=np.array([cx, cy, 0.25]),
                                    scale=np.array([sx, sy, 0.5]), color=np.array([0.55, 0.4, 0.25])))
    base_scan = "/World/burger/Geometry/base_footprint/base_link/base_scan"
    lidar = TurtleBot3Lidar(base_scan, noise_stddev=0.0)
    world.reset()
    for _ in range(100):
        world.step(render=False)

    xf = UsdGeom.Xformable(world.stage.GetPrimAtPath(base_scan)).ComputeLocalToWorldTransform(0)
    origin = xf.ExtractTranslation()
    yaw = math.atan2(xf[0][1], xf[0][0])
    depths, frame = lidar.raw()
    ranges, stamp = lidar.ranges()
    print(f"base_scan world position {tuple(round(v, 4) for v in origin)}, yaw {math.degrees(yaw):.2f} deg; "
          f"sensor time {stamp:.3f} s", flush=True)
    print(f"raw depths for misses look like: {sorted(set(np.round(depths[~np.isfinite(ranges)], 3)))[:3]}", flush=True)

    errors, mismatched = [], []
    for i, (angle, measured) in enumerate(zip(lidar.angles, ranges)):
        expected = analytic_range(origin, yaw + angle, RANGE_MIN, RANGE_MAX)
        if math.isinf(expected) != math.isinf(measured):
            mismatched.append((i, round(math.degrees(angle), 1), expected, measured))
        elif math.isfinite(expected):
            errors.append(abs(measured - expected))
        if i in (0, 10, 20, 30, 39):
            print(f"  ray {i:2d} at {math.degrees(angle):6.1f} deg: measured {measured:.4f}  expected {expected:.4f}",
                  flush=True)
    print(f"RESULT: {len(errors)} hits, max |error| {max(errors):.5f} m, mean {np.mean(errors):.5f} m; "
          f"hit/miss mismatches: {len(mismatched)} {mismatched[:5]}", flush=True)

    if args.render:
        from isaacsim.core.utils.viewports import set_camera_view
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        for i, (angle, r) in enumerate(zip(lidar.angles, ranges)):
            if math.isfinite(r):
                p = (origin[0] + r * math.cos(yaw + angle), origin[1] + r * math.sin(yaw + angle), origin[2])
                world.scene.add(VisualSphere(prim_path=f"/World/hits/h{i}", name=f"h{i}", position=np.array(p),
                                             radius=0.012, color=np.array([1.0, 0.1, 0.1])))
        set_camera_view(eye=np.array([0.15, 0.2, 1.7]), target=np.array([0.15, 0.2, 0.0]),
                        camera_prim_path="/OmniverseKit_Persp")
        for _ in range(80):
            world.step(render=True)
        args.render.parent.mkdir(parents=True, exist_ok=True)
        capture_viewport_to_file(get_active_viewport(), str(args.render))
        for _ in range(20):
            app.update()
        print("saved", args.render, flush=True)
    app.close()


if __name__ == "__main__":
    main()
