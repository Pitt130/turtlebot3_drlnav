#!/usr/bin/env python3
"""Has a train_grscene.sh run stopped improving? For its CONVERGED_CMD: exit 0 = converged (stop), 1 = keep training.

    converged.py <run_dir> [--window 150000] [--min-steps 300000] [--min-gain 2.0]

Takes the training episodes of every chunk (chunk_*/logs/monitor.csv, in order) and compares the success rate of
the episodes that ended in the last `window` steps with that of the `window` steps before. Converged once at least
`min_steps` steps have run and the gain is below `min_gain` percentage points. With ~700 episodes per 150k steps at
~90 % success, one window's rate has a standard error of ~1.1 points and the gain ~1.6. Prints one line.
"""
import argparse
import csv
from pathlib import Path
import sys


def episodes(run):
    """[(steps, success)] of every finished training episode, in training order."""
    rows = []
    for chunk in sorted(run.glob("chunk_*")):
        monitor = chunk / "logs" / "monitor.csv"
        if monitor.exists():
            lines = monitor.read_text().splitlines()[1:]          # the first line is SB3's JSON header
            rows += [(int(float(r["l"])), int(float(r["success"])) == 1) for r in csv.DictReader(lines)]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--window", type=int, default=150_000)
    parser.add_argument("--min-steps", type=int, default=300_000)
    parser.add_argument("--min-gain", type=float, default=2.0)
    args = parser.parse_args()
    rows = episodes(args.run)
    total = sum(n for n, _ in rows)
    recent, before, end = [], [], 0
    for n, ok in rows:
        end += n
        if end > total - args.window:
            recent.append(ok)
        elif end > total - 2 * args.window:
            before.append(ok)
    k = args.window // 1000
    if not recent or not before:
        print(f"convergence at {total} steps: fewer than two {k}k windows -> continue")
        return 1
    rate, rate_before = 100 * sum(recent) / len(recent), 100 * sum(before) / len(before)
    gain = rate - rate_before
    done = total >= args.min_steps and gain < args.min_gain
    print(f"convergence at {total} steps: last {k}k {rate:.1f}% ({len(recent)} ep), previous {k}k "
          f"{rate_before:.1f}% ({len(before)} ep), gain {gain:+.1f} -> {'converged' if done else 'continue'}")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
