"""Gap-free occupancy grid (v6) for the waypoint planner.

The v4 grid tests each 5 cm cell with four 5 cm rays from its centre at the lidar height and one
downward ray at its centre. A 2-3 cm table leg between those rays, or the spoke of a chair's star
base between two cell centres, leaves its cells FREE, and the planner routes the robot into it.
v6 starts from v4 and tests every FREE cell with PhysX box overlaps that cover the whole cell:

    something in the lidar slab (lidar height +- 1 cm)       -> OCCUPIED (as v4 would have seen it)
    something between 0.5 cm and the slab, entirely below it -> LOW      (as v4's downward ray would)

It keeps v4's meaning of each class and adds nothing above the lidar plane (unlike v5, whose 6-20 cm
body band also caught bed frames and seats). Floors ("ground" colliders) are ignored. Cells are only
added, never cleared, and the start/goal sampler keeps using v4.

    OMNI_KIT_ACCEPT_EULA=YES isaac/isaac_python.sh isaac/build_fine_map.py --scene <scene_dir>
"""
import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

SLAB = 0.01          # half height of the lidar slab
LOW_MIN = 0.005      # same as build_occupancy's low_object


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
    from grscene import FREE, LIDAR_HEIGHT, LOW, OCCUPIED, Occupancy, build_occupancy, load_grscene

    world = World()
    scene = load_grscene(world, scene_root, args.usd_name)
    world.reset()
    v4 = build_occupancy(scene)                     # cached grid; the sampler's
    floor = getattr(v4, "floor", None)
    pq = get_physx_scene_query_interface()
    if floor is None:                               # loaded from the cache: measure the floor as build_occupancy does
        def floor_at(x, y):
            r = pq.raycast_closest(carb.Float3(x, y, float(scene.hi[2]) + 1.0), carb.Float3(0, 0, -1),
                                   float(scene.hi[2] - scene.lo[2]) + 2.0)
            return r["position"][2] if r["hit"] else None
        hits = [floor_at(x, y) for x in np.linspace(scene.lo[0], scene.hi[0], 30)
                for y in np.linspace(scene.lo[1], scene.hi[1], 30)]
        floor = float(np.percentile([z for z in hits if z is not None], 10))

    def blocked(x, y, z0, z1):
        found = []

        def report(hit):
            if "/ground/" in hit.collision or hit.collision.startswith("/World/burger"):
                return True
            found.append(hit.collision)
            return False
        pq.overlap_box(carb.Float3(v4.cell / 2, v4.cell / 2, (z1 - z0) / 2),
                       carb.Float3(x, y, floor + (z0 + z1) / 2), carb.Float4(0, 0, 0, 1), report, False)
        return found

    t0 = time.time()
    grid = v4.grid.copy()
    added = {"occupied": 0, "low": 0}
    objects = {}
    for r, c in zip(*np.nonzero(v4.grid == FREE)):
        x, y = v4.to_world(r, c)
        found = blocked(x, y, LIDAR_HEIGHT - SLAB, LIDAR_HEIGHT + SLAB)
        if found:
            grid[r, c] = OCCUPIED
            added["occupied"] += 1
        else:
            found = blocked(x, y, LOW_MIN, LIDAR_HEIGHT - SLAB)
            if found:
                grid[r, c] = LOW
                added["low"] += 1
        for f in found:
            cat = f.split("/")[-3] if f.count("/") > 3 else f
            objects[cat] = objects.get(cat, 0) + 1
    v6 = Occupancy(v4.origin, v4.cell, grid)
    out = scene_root / f"occupancy_v6_{int(round(v4.cell * 100))}cm_{int(round(LIDAR_HEIGHT * 1000))}mm_fine.npz"
    v6.save(out)
    free4 = int((v4.grid == FREE).sum())
    print(f"FINEMAP {scene_root.name}: floor {floor:.3f}, {free4} free cells in v4, +{added['occupied']} OCCUPIED, "
          f"+{added['low']} LOW in {time.time() - t0:.1f} s -> {out.name}", flush=True)
    print(f"FINEMAP objects added: {dict(sorted(objects.items(), key=lambda kv: -kv[1]))}", flush=True)
    try:
        from PIL import Image
        Image.fromarray(np.concatenate([v4.to_image(), np.full((v4.grid.shape[0], 4, 3), 255, np.uint8),
                                        v6.to_image()], 1)).save(scene_root / "occupancy_v4_vs_v6.png")
    except Exception as e:                           # image is optional
        print("image skipped:", e)
    app.close()


if __name__ == "__main__":
    main()
