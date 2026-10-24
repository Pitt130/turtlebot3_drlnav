"""Success by task difficulty for check_grscene runs: grid path length / straight-line distance.

    .venv/bin/python isaac/difficulty_breakdown.py <scene_dir> <label>=<run_dir> [<label>=<run_dir> ...]
run_dir holds episodes.csv (runtime --episode-log) and check_raw.log (check_grscene output).
"""
import collections
import csv
import glob
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grscene  # noqa: E402

BINS = ((1.0, 1.25, "nearly straight"), (1.25, 1.7, "around furniture"), (1.7, 99, "other room / walls"))


def main():
    scene_dir, runs = sys.argv[1], sys.argv[2:]
    occ = grscene.Occupancy.load(sorted(glob.glob(f"{scene_dir}/occupancy_v2_*.npz"))[0])
    grid = grscene.StartGoalSampler(occ, inflate=0.25).occ
    for spec in runs:
        label, run_dir = spec.split("=", 1)
        eps = list(csv.DictReader(open(f"{run_dir}/episodes.csv")))
        outs = {}
        for line in open(f"{run_dir}/check_raw.log"):
            m = re.match(r"episode\s+(\d+):.*-> ([A-Z_]+) after (\d+) steps", line)
            if m:
                outs[int(m.group(1))] = (m.group(2), int(m.group(3)))
        rows = []
        for i, e in enumerate(eps, 1):
            if i not in outs:
                continue
            sx, sy, gx, gy = (float(e[k]) for k in ("robot_x", "robot_y", "goal_x", "goal_y"))
            steps = grid.path_steps(*grid.to_cell(sx, sy), 400)
            rg, cg = grid.to_cell(gx, gy)
            path = steps[rg, cg] * grid.cell * 1.2      # 8-connected steps underestimate by up to 29%
            eu = math.hypot(gx - sx, gy - sy)
            rows.append((max(1.0, path / eu) if np.isfinite(path) else math.inf, eu, path, outs[i][0], outs[i][1]))
        counts = collections.Counter(r[3] for r in rows)
        print(f"== {label}: {len(rows)} episodes, success {100 * counts['SUCCESS'] / len(rows):.0f}%  "
              f"{dict(counts)}; mean {np.mean([r[4] for r in rows]):.0f} steps")
        for lo, hi, name in BINS:
            sel = [r for r in rows if lo <= r[0] < hi]
            if sel:
                by = {o: sum(r[3] == o for r in sel) for o in sorted({r[3] for r in sel})}
                print(f"   {name:20s} n={len(sel):2d}  success {100 * by.get('SUCCESS', 0) / len(sel):3.0f}%  {by}  "
                      f"mean path {np.mean([r[2] for r in sel]):.1f} m")


if __name__ == "__main__":
    main()
