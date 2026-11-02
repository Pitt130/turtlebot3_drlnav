"""Body-height occupancy grid (v5) for the waypoint planner.

The v4 grid scans one plane, the lidar's 17.2 cm, with four 5 cm rays from each cell centre. Chair
legs that fall between those rays and seats just above the plane are missing from it, so the
planner routes the robot into them. v5 starts from v4 and adds, for each FREE cell, two PhysX box
overlaps covering the whole cell: something 6-20 cm above the floor (the robot's body) makes the
cell OCCUPIED, something 0.5-6 cm makes it LOW. Floors ("ground" colliders) are ignored. Cells are
only ever added, never cleared, and the start/goal sampler keeps using v4.

    OMNI_KIT_ACCEPT_EULA=YES ~/isaacsim/.venv/bin/python isaac/build_body_map.py --scene <scene_dir>
"""
import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

BODY = (0.06, 0.20)
LOW_BAND = (0.005, 0.06)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--usd-name", default="start_result_navigation.usd")
    args = parser.parse_args()
    scene_root = args.scene.resolve()
    # Same MDL search path as the runtime (see turtlebot3_isaac_sim.py).
    mdl_dir = scene_root.parents[2] / "mdl_search"
    mdl_dir.mkdir(exist_ok=True)
    if not (mdl_dir / "Materials").exists():
        (mdl_dir / "Materials").symlink_to(scene_root.parents[1] / "Materials")
    os.environ["MDL_SYSTEM_PATH"] = str(mdl_dir)

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    import carb
    from isaacsim.core.api import World
    from omni.physx import get_physx_scene_query_interface
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from grscene import FREE, LOW, OCCUPIED, Occupancy, build_occupancy, load_grscene

    world = World()
    scene = load_grscene(world, scene_root, args.usd_name)
    world.reset()
    v4 = build_occupancy(scene)                     # cached grid; the sampler's
    pq = get_physx_scene_query_interface()

    def floor_at(x, y):
        r = pq.raycast_closest(carb.Float3(x, y, float(scene.hi[2]) + 1.0), carb.Float3(0, 0, -1),
                               float(scene.hi[2] - scene.lo[2]) + 2.0)
        return r["position"][2] if r["hit"] else None

    hits = [floor_at(x, y) for x in np.linspace(scene.lo[0], scene.hi[0], 30) for y in np.linspace(scene.lo[1], scene.hi[1], 30)]
    floor = float(np.percentile([z for z in hits if z is not None], 10))

    def blocked(x, y, z0, z1):
        found = []

        def report(hit):
            if "/ground/" in hit.collision:
                return True
            found.append(hit.collision)
            return False
        pq.overlap_box(carb.Float3(v4.cell / 2, v4.cell / 2, (z1 - z0) / 2),
                       carb.Float3(x, y, floor + (z0 + z1) / 2), carb.Float4(0, 0, 0, 1), report, False)
        return found

    t0 = time.time()
    grid = v4.grid.copy()
    added = {"body": 0, "low": 0}
    objects = {}
    for r, c in zip(*np.nonzero(v4.grid == FREE)):
        x, y = v4.to_world(r, c)
        found = blocked(x, y, *BODY)
        if found:
            grid[r, c] = OCCUPIED
            added["body"] += 1
        else:
            found = blocked(x, y, *LOW_BAND)
            if found:
                grid[r, c] = LOW
                added["low"] += 1
        for f in found:
            cat = f.split("/")[-3] if f.count("/") > 3 else f
            objects[cat] = objects.get(cat, 0) + 1
    v5 = Occupancy(v4.origin, v4.cell, grid)
    out = scene_root / f"occupancy_v5_{int(round(v4.cell * 100))}cm_body{int(BODY[1] * 100)}cm.npz"
    v5.save(out)
    free4 = int((v4.grid == FREE).sum())
    print(f"BODYMAP {scene_root.name}: floor {floor:.3f}, {free4} free cells in v4, +{added['body']} OCCUPIED, "
          f"+{added['low']} LOW in {time.time() - t0:.1f} s -> {out.name}", flush=True)
    print(f"BODYMAP objects added: {dict(sorted(objects.items(), key=lambda kv: -kv[1]))}", flush=True)
    try:
        from PIL import Image
        Image.fromarray(np.concatenate([v4.to_image(), np.full((v4.shape[0], 4, 3), 255, np.uint8), v5.to_image()], 1)) \
            .save(scene_root / "occupancy_v4_vs_v5.png")
    except Exception as e:                           # image is optional
        print("image skipped:", e)
    app.close()


if __name__ == "__main__":
    main()
