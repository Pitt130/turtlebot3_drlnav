"""Who hit whom: classify the moving-obstacle collisions (COLL_OBST) of evaluation track CSVs.

Uses the last 0.5 s of each such episode from check_grscene.py --track-out (columns x, y and dyn_x, dyn_y, the
nearest moving obstacle) and compares how fast the robot and the obstacle closed in along the line between them:

    robot closing <= 0.03 m/s      obstacle hit robot (robot still/moving away)
    obstacle closing <= 0.03 m/s   robot drove into obstacle
    otherwise                      both closing, robot faster / obstacle faster

    python3 isaac/experiments/collision_blame.py --dir <eval dir> <label>...   # reads <dir>/<label>_*.track.csv
"""
import argparse
import collections
import csv
import math
from pathlib import Path

TAIL_STEPS = 6          # 0.5 s at 0.1 s steps
COLL_OBST = "3"


def analyse(directory, label):
    cases = []
    for f in sorted(Path(directory).glob(f"{label}_*.track.csv")):
        episodes = collections.defaultdict(list)
        for r in csv.DictReader(open(f)):
            episodes[r["episode"]].append(r)
        for rows in episodes.values():
            if rows[-1]["outcome"] != COLL_OBST:
                continue
            seg = [r for r in rows[-TAIL_STEPS:] if r["dyn_x"]]
            if len(seg) < 3:
                cases.append(("no obstacle pose",))
                continue
            a, b = seg[0], seg[-1]
            dt = 0.1 * (len(seg) - 1)
            rx, ry, ox, oy = float(b["x"]), float(b["y"]), float(b["dyn_x"]), float(b["dyn_y"])
            vr = ((rx - float(a["x"])) / dt, (ry - float(a["y"])) / dt)
            vo = ((ox - float(a["dyn_x"])) / dt, (oy - float(a["dyn_y"])) / dt)
            if math.hypot(*vo) > 1.5:            # the nearest obstacle changed within the window
                cases.append(("obstacle switched",))
                continue
            d = math.hypot(ox - rx, oy - ry)
            u = ((ox - rx) / d, (oy - ry) / d)
            robot_in = vr[0] * u[0] + vr[1] * u[1]       # robot speed towards the obstacle
            obst_in = -(vo[0] * u[0] + vo[1] * u[1])     # obstacle speed towards the robot
            speed = math.hypot(*vr)
            if speed > 0.03:                             # where the obstacle was relative to the robot's motion
                ang = abs(math.degrees(math.atan2(vr[0] * u[1] - vr[1] * u[0], vr[0] * u[0] + vr[1] * u[1])))
                side = "front" if ang < 60 else ("side" if ang < 120 else "behind")
            else:
                side = "robot stopped"
            if robot_in <= 0.03:
                kind = "obstacle hit robot (robot still/moving away)"
            elif obst_in <= 0.03:
                kind = "robot drove into obstacle"
            elif robot_in >= obst_in:
                kind = "both closing, robot faster"
            else:
                kind = "both closing, obstacle faster"
            cases.append((kind, side, robot_in, obst_in))
    print(f"== {label}: {len(cases)} obstacle collisions")
    for kind, n in collections.Counter(c[0] for c in cases).most_common():
        print(f"   {n:3d}  {kind}")
    full = [c for c in cases if len(c) > 1]
    print("   obstacle direction vs robot motion:", dict(collections.Counter(c[1] for c in full)))
    if full:
        print(f"   mean speed towards each other: robot {sum(c[2] for c in full) / len(full):.2f} m/s, "
              f"obstacle {sum(c[3] for c in full) / len(full):.2f} m/s")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels", nargs="+", help="evaluation labels (eval_scenes.sh <label>)")
    parser.add_argument("--dir", default=".", help="folder with the <label>_<scene>.track.csv files")
    args = parser.parse_args()
    for label in args.labels:
        analyse(args.dir, label)


if __name__ == "__main__":
    main()
