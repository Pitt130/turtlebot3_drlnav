"""What is around the robot at given poses in a GRScenes apartment (static colliders only), for collision analysis.

For every pose: 360 horizontal rays (1 deg) from the lidar origin at several heights, up to 1 m; per height the
closest hit (distance, bearing relative to the heading, collider path and category), the closest hit among the 40
policy rays (every 9 deg, the LDS layout), and the colliders overlapping the robot body box (+2 cm).

    OMNI_KIT_ACCEPT_EULA=YES ~/isaacsim/.venv/bin/python isaac/collision_probe.py --scene DIR --poses poses.json \
        --out probe.json
poses.json: [{"id": ..., "x": ..., "y": ..., "yaw": ...}, ...] (base_footprint pose, as in the env's odometry).
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

SCAN_OFFSET = -0.032                       # base_scan relative to base_footprint, along +x
HEIGHTS = {"low2": 0.02, "h06": 0.06, "h10": 0.10, "h14": 0.14, "lds": 0.172, "top19": 0.19}
BODY = (0.07, 0.09)                        # burger half length/width around base_footprint (m), height 0.01-0.19
MAX_RANGE = 1.0


def category(path):
    parts = path.split("/")
    return parts[-3] if len(parts) > 3 else path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--usd-name", default="start_result_navigation.usd")
    args = parser.parse_args()
    scene_root = args.scene.resolve()
    mdl_dir = scene_root.parents[2] / "mdl_search"           # same MDL search path as the runtime
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
    from grscene import load_grscene

    world = World()
    scene = load_grscene(world, scene_root, args.usd_name)
    world.reset()
    pq = get_physx_scene_query_interface()

    def floor_at(x, y):
        r = pq.raycast_closest(carb.Float3(x, y, float(scene.hi[2]) + 1.0), carb.Float3(0, 0, -1),
                               float(scene.hi[2] - scene.lo[2]) + 2.0)
        return r["position"][2] if r["hit"] else None
    hits = [floor_at(x, y) for x in np.linspace(scene.lo[0], scene.hi[0], 30)
            for y in np.linspace(scene.lo[1], scene.hi[1], 30)]
    floor = float(np.percentile([z for z in hits if z is not None], 10))

    def ray(ox, oy, z, a):
        r = pq.raycast_closest(carb.Float3(ox, oy, z), carb.Float3(math.cos(a), math.sin(a), 0.0), MAX_RANGE)
        if r["hit"] and "/ground/" not in r["collision"]:
            return float(r["distance"]), r["collision"]
        return math.inf, ""

    def body_overlaps(x, y, yaw):
        found = []

        def report(hit):
            if "/ground/" not in hit.collision:
                found.append(hit.collision)
            return True
        q = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))   # w, x, y, z
        pq.overlap_box(carb.Float3(BODY[0] + 0.02, BODY[1] + 0.02, 0.09), carb.Float3(x, y, floor + 0.10),
                       carb.Float4(q[1], q[2], q[3], q[0]), report, False)
        return sorted(set(found))

    out = []
    for pose in json.loads(args.poses.read_text()):
        x, y, yaw = pose["x"], pose["y"], pose["yaw"]
        ox, oy = x + SCAN_OFFSET * math.cos(yaw), y + SCAN_OFFSET * math.sin(yaw)
        res = {"id": pose["id"], "heights": {}}
        for name, h in HEIGHTS.items():
            dense = [ray(ox, oy, floor + h, yaw + math.radians(d)) for d in range(360)]
            i = int(np.argmin([d for d, _ in dense]))
            j = min(range(0, 360, 9), key=lambda k: dense[k][0])
            res["heights"][name] = {
                "min": round(dense[i][0], 3) if math.isfinite(dense[i][0]) else None,
                "bearing": i if i <= 180 else i - 360, "path": dense[i][1], "cat": category(dense[i][1]),
                "min40": round(dense[j][0], 3) if math.isfinite(dense[j][0]) else None,
                "bearing40": j if j <= 180 else j - 360, "cat40": category(dense[j][1]),
            }
            if name in ("lds", "low2"):          # every ray of the two sensor planes, for plots (mm, -1 = none)
                res.setdefault("dense", {})[name] = [int(round(d * 1000)) if math.isfinite(d) else -1 for d, _ in dense]
        res["body"] = [[category(p), p] for p in body_overlaps(x, y, yaw)]
        out.append(res)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"PROBE {scene_root.name}: {len(out)} poses, floor {floor:.3f} -> {args.out}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
