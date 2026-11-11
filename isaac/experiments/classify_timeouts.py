"""Classify the timeout episodes of evaluation track CSVs by what the robot did in its last 15 s.

    python3 isaac/experiments/classify_timeouts.py --dir <eval dir> <name>...   # reads <dir>/<name>.track.csv
"""
import argparse
import collections
import csv
from pathlib import Path

TIMEOUT = "4"
TAIL_STEPS = 150        # 15 s at 0.1 s steps


def classify(path):
    by_ep = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        by_ep[int(r["episode"])].append(r)
    timeouts = [v for v in by_ep.values() if v[-1]["outcome"] == TIMEOUT]
    kinds, where = collections.Counter(), collections.Counter()
    for v in timeouts:
        tail = v[-TAIL_STEPS:]
        xs = [float(r["x"]) for r in tail]
        ys = [float(r["y"]) for r in tail]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        lin = sum(float(r["lin"]) for r in tail) / len(tail)
        gd0, gd1 = float(tail[0]["goal_distance"]), float(tail[-1]["goal_distance"])
        if span < 0.15 and lin > 0.3:
            k = "pushing against something, not moving"
        elif span < 0.4:
            k = "turning / oscillating in place"
        elif gd1 < gd0 - 0.5:
            k = "still making progress (too slow)"
        else:
            k = "moving around without progress"
        kinds[k] += 1
        where[(round(xs[-1]), round(ys[-1]))] += 1
    return len(timeouts), kinds, where


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="+", help="track file names without .track.csv")
    parser.add_argument("--dir", default=".", help="folder with the track CSVs")
    args = parser.parse_args()
    for name in args.names:
        path = Path(args.dir) / f"{name}.track.csv"
        if not path.exists():
            print(f"{name}: no track")
            continue
        n, kinds, where = classify(path)
        print(f"{name}: {n} timeouts -> {dict(kinds)}; end spots (1 m grid): {where.most_common(5)}")


if __name__ == "__main__":
    main()
