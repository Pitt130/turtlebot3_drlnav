"""Compare stage_scene's obstacle animation with a Gazebo /obstacle/odom recording.

    python3 isaac/check_obstacle_paths.py --stage 4 --csv gazebo_stage4_obstacles.csv
The CSV (t,child,x,y,z) is recorded from Gazebo starting at t=0 without resets.
"""
import argparse
import csv
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_scene import load_stage  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--repo", default=".")
    args = parser.parse_args()
    stage = load_stage(args.repo, args.stage)
    by_body = {o.frame_id: o for o in stage.obstacles}
    errors = {}
    for row in csv.DictReader(open(args.csv)):
        o = by_body.get(row["child"])
        if o is None:
            continue
        (x, y, _), _ = o.pose(float(row["t"]))
        errors.setdefault(o.frame_id, []).append(
            (math.hypot(x - float(row["x"]), y - float(row["y"])), float(row["t"])))
    ok = True
    for body, errs in sorted(errors.items()):
        worst = max(errs)
        mean = sum(e for e, _ in errs) / len(errs)
        span = max(t for _, t in errs)
        good = worst[0] < 0.02
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] {body}: {len(errs)} samples over {span:.0f} s, "
              f"mean error {mean * 1000:.1f} mm, max {worst[0] * 1000:.1f} mm at t={worst[1]:.2f}s")
    missing = set(by_body) - set(errors)
    if missing:
        ok = False
        print(f"[FAIL] no Gazebo samples for {sorted(missing)}")
    print("PATHS MATCH" if ok else "PATHS DIFFER")


if __name__ == "__main__":
    main()
