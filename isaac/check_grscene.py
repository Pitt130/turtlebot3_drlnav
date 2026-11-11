"""Episodes with random actions in a GRScenes apartment, through the real Gymnasium environment.

Start turtlebot3_isaac_sim.py --grscene <dir> --episode-log <csv> first, then:
    DRLNAV_SIM_BACKEND=isaac DRLNAV_STAGE=4 .venv/bin/python isaac/check_grscene.py [episodes] [--log <csv>]

Checks, per episode: the robot starts clear of obstacles (every lidar ray well above the
collision threshold), the goal is published and lies within the sampler's distance
range, every step advances 0.1 s, and the robot does not tumble.
"""
import argparse
import csv
import math
import sys

import numpy as np

from turtlebot3_drl.common.settings import THRESHOLD_COLLISION
from turtlebot3_drl.drl_environment.drl_environment_gymnasium import make_env

NAMES = {0: "UNKNOWN", 1: "SUCCESS", 2: "COLL_WALL", 3: "COLL_OBST", 4: "TIMEOUT", 5: "TUMBLE"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=int, nargs="?", default=20)
    parser.add_argument("--log", help="the runtime's --episode-log CSV, to cross-check starts and goals")
    parser.add_argument("--goal-min", type=float, default=1.5)
    parser.add_argument("--goal-max", type=float, default=5.5)
    parser.add_argument("--model", help="SB3 TD3 zip: drive with this policy (deterministic) instead of random actions")
    parser.add_argument("--track-out", help="CSV of every step: episode, x, y, tilt_deg, outcome")
    parser.add_argument("--trace-out", help="JSON lines, one per step (step 0 = after reset), for collision "
                        "analysis: pose, heading, action, waypoint, stall recovery, the ranges in m, "
                        "moving obstacles as [index, stamp, x, y]")
    parser.add_argument("--waypoints", type=float, metavar="LOOKAHEAD",
                        help="hierarchical: plan on the scene occupancy grid and give the policy a waypoint this "
                             "far along the path instead of the goal (needs --model and --scene)")
    parser.add_argument("--scene", help="GRScenes scene folder (for --waypoints): its occupancy_v3_*.npz is used")
    parser.add_argument("--plan-inflate", type=float, default=0.18)
    parser.add_argument("--plan-low-inflate", type=float, help="planning margin around obstacles below the lidar plane")
    parser.add_argument("--plan-map", default="occupancy_v4",
                        help="occupancy grid file prefix the planner uses (occupancy_v5: body-height grid from "
                             "isaac/build_body_map.py); the runtime samples starts and goals on v4 either way")
    parser.add_argument("--plan-cost", type=float, default=0.0,
                        help="clearance-aware planner: cells near the inflated margin cost up to (1 + this) times "
                             "as much (cost-to-go field, octile steps); 0 = plain step-count wavefront")
    parser.add_argument("--plan-fallback-map", default=None,
                        help="occupancy grid prefix (e.g. occupancy_v4) planned on where --plan-map has no path")
    parser.add_argument("--plan-cost-dist", type=float, default=0.25,
                        help="with --plan-cost: distance beyond the inflated margin over which the extra cost fades")
    parser.add_argument("--stall-recovery", action="store_true",
                        help="move the waypoint 1.5 m ahead for 2 s after 3 s without 0.1 m of path progress")
    parser.add_argument("--path-timeout", type=float, metavar="K",
                        help="episode time limit max(EPISODE_TIMEOUT_SECONDS, K * grid path length / max speed) "
                             "instead of the fixed limit (needs --scene)")
    parser.add_argument("--scan-stack-skip", type=int, default=2,
                        help="control steps between stacked scans, as trained (train.py --scan-stack-skip); the number "
                             "of scans per observation comes from the model's observation size")
    args = parser.parse_args()
    env = make_env(step_sim_time=0.1)
    rng = np.random.default_rng(0)
    policy = None
    if args.model:
        import sys
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "sb3"))   # tomas_policy
        from stable_baselines3 import TD3
        policy = TD3.load(args.model, device="cuda")
    stack = None
    if policy is not None and policy.observation_space.shape[0] != env.observation_space.shape[0]:
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
        from waypoints import ScanStack
        scan_size = env.observation_space.shape[0] - 4
        frames = 1 + (policy.observation_space.shape[0] - env.observation_space.shape[0]) // scan_size
        stack = ScanStack(frames, args.scan_stack_skip, scan_size)
        print(f"scans per observation: {frames}, {args.scan_stack_skip} steps apart", flush=True)
    planner = None
    if args.waypoints or args.path_timeout:
        import glob, sys
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
        from grscene import Occupancy
        from waypoints import WaypointPlanner
        from turtlebot3_drl.common.settings import EPISODE_TIMEOUT_SECONDS, SPEED_LINEAR_MAX
        from turtlebot3_drl.drl_environment.drl_environment_gymnasium import MAX_GOAL_DISTANCE
        from waypoints import make_planner
        planner = make_planner(args.scene, args.plan_map, args.plan_fallback_map, inflate=args.plan_inflate,
                               lookahead=args.waypoints or 1.5, low_inflate=args.plan_low_inflate,
                               cost_weight=args.plan_cost, cost_dist=args.plan_cost_dist)
        if args.waypoints:
            print(f"waypoint planner: lookahead {args.waypoints} m, planning grid inflated {args.plan_inflate} m"
                  f" (below-lidar obstacles {args.plan_low_inflate or args.plan_inflate} m),"
                  f" stall recovery {'on' if args.stall_recovery else 'off'}, map {args.plan_map} (fallback {args.plan_fallback_map}), "
                  f"plan cost {args.plan_cost}/{args.plan_cost_dist} m", flush=True)
            from geometry_msgs.msg import Pose
            waypoint_pub = env.create_publisher(Pose, "waypoint", 10)     # for the runtime's marker
        if args.path_timeout:
            print(f"time limit: max({EPISODE_TIMEOUT_SECONDS} s, {args.path_timeout} x path / {SPEED_LINEAR_MAX} m/s)", flush=True)
    track_file = open(args.track_out, "w") if args.track_out else None
    if track_file:
        track_file.write("episode,step,x,y,tilt_deg,min_obstacle,goal_distance,lin,ang,outcome,dyn_x,dyn_y\n")
    trace_file = open(args.trace_out, "w") if args.trace_out else None

    def trace(ep, n, action=None, waypoint=None, recovering=False, outcome=0):
        if trace_file is None:
            return
        import json
        from turtlebot3_drl.common.settings import LIDAR_DISTANCE_CAP
        samples = getattr(env, "_obstacle_samples", {})
        trace_file.write(json.dumps({
            "ep": ep + 1, "step": n, "x": round(env.robot_x, 4), "y": round(env.robot_y, 4),
            "yaw": round(env.robot_heading, 4), "tilt": round(math.degrees(env.robot_tilt), 2),
            "lin": None if action is None else round(float(action[0]), 3),
            "ang": None if action is None else round(float(action[1]), 3),
            "wp": None if waypoint is None else [round(float(waypoint[0]), 3), round(float(waypoint[1]), 3)],
            "rec": int(bool(recovering)), "out": outcome,
            "scan": [round(float(r) * LIDAR_DISTANCE_CAP, 3) for r in env.scan_ranges],
            "peds": [[int(i), round(float(t), 2), round(float(x), 3), round(float(y), 3)]
                     for i, (t, x, y) in sorted(samples.items())],
        }) + "\n")
    ok = True

    def report(label, passed, detail=""):
        nonlocal ok
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] {label} {detail}", flush=True)

    def nearest_moving_obstacle(env):
        """x,y of the closest moving obstacle with a fresh pose, or an empty pair."""
        d = list(getattr(env, "obstacle_distances", []))
        samples = getattr(env, "_obstacle_samples", {})
        if not d or not math.isfinite(min(d)) or d.index(min(d)) not in samples:
            return ","
        _, x, y = samples[d.index(min(d))]
        return f"{x:.3f},{y:.3f}"

    starts, goals, min_scans, dts, outcomes, lengths, stalls = [], [], [], [], [], [], []
    for ep in range(args.episodes):
        obs, info = env.reset()
        starts.append((env.robot_x, env.robot_y))
        goals.append((env.goal_x, env.goal_y))
        min_scans.append(min(env.scan_ranges) * 3.5)
        n, max_tilt, tilt_at, track = 0, 0.0, None, []
        recovering = False
        trace(ep, 0)
        unreachable, path_len = 0, math.nan
        if planner is not None:
            planner.set_goal(env.goal_x, env.goal_y)
            planner.reset_progress()
            path_len = planner.path_length(env.robot_x, env.robot_y)
            if args.path_timeout and math.isfinite(path_len):
                env.episode_deadline = env.time_sec + max(EPISODE_TIMEOUT_SECONDS,
                                                          args.path_timeout * path_len / SPEED_LINEAR_MAX)
        while True:
            if args.waypoints:
                if args.stall_recovery:
                    wx, wy, reachable, recovering = planner.step(env.robot_x, env.robot_y)
                else:
                    wx, wy, reachable = planner.waypoint(env.robot_x, env.robot_y)
                unreachable += not reachable
                msg = Pose()
                msg.position.x, msg.position.y = float(wx), float(wy)
                waypoint_pub.publish(msg)
                policy_obs = planner.observation_with(obs, env.robot_x, env.robot_y, env.robot_heading, wx, wy,
                                                      MAX_GOAL_DISTANCE)
            else:
                policy_obs = obs
            if stack is not None:
                policy_obs = stack.push(policy_obs) if n else stack.reset(policy_obs)
            action = policy.predict(policy_obs, deterministic=True)[0] if policy is not None else rng.uniform(-1, 1, 2).astype(np.float32)
            obs, r, term, trunc, info = env.step(action)
            dts.append(info["step_sim_dt"]); n += 1
            track.append((round(env.robot_x, 3), round(env.robot_y, 3), round(math.degrees(env.robot_tilt), 2)))
            trace(ep, n, action, (wx, wy) if args.waypoints else None, recovering,
                  int(info["success"]) if (term or trunc) else 0)
            if track_file:
                track_file.write(f"{ep + 1},{n},{env.robot_x:.3f},{env.robot_y:.3f},{math.degrees(env.robot_tilt):.2f},"
                                 f"{env.obstacle_distance:.3f},{env.goal_distance:.3f},{float(action[0]):.3f},{float(action[1]):.3f},"
                                 f"{int(info['success']) if (term or trunc) else 0},{nearest_moving_obstacle(env)}\n")
            if env.robot_tilt > max_tilt:
                max_tilt, tilt_at = env.robot_tilt, (env.robot_x, env.robot_y)
            if term or trunc:
                break
        outcomes.append(NAMES[int(info["success"])]); lengths.append(n)
        stalls.append(planner.stalls if planner is not None else 0)
        d = math.hypot(goals[-1][0] - starts[-1][0], goals[-1][1] - starts[-1][1])
        print(f"episode {ep + 1:2d}: start ({starts[-1][0]:+.2f},{starts[-1][1]:+.2f}) goal ({goals[-1][0]:+.2f},"
              f"{goals[-1][1]:+.2f}) dist {d:.2f} m, min scan at start {min_scans[-1]:.2f} m -> {outcomes[-1]} after {n} steps"
              f"; max tilt {math.degrees(max_tilt):.1f} deg at ({tilt_at[0]:+.2f},{tilt_at[1]:+.2f})"
              + (f"; path {path_len:.2f} m" if planner is not None else "")
              + (f"; waypoint unreachable {unreachable} steps" if unreachable else "")
              + (f"; path_blocked {stalls[-1]}" if stalls[-1] else ""), flush=True)
        if outcomes[-1] == "TUMBLE":
            print("   last 8 steps (x, y, tilt deg):", track[-8:], flush=True)
    if trace_file:
        trace_file.close()
    env.close()

    report("starts clear of obstacles", all(m > THRESHOLD_COLLISION + 0.08 for m in min_scans),
           f"(min scan at start: min {min(min_scans):.2f} m)")
    dists = [math.hypot(g[0] - s[0], g[1] - s[1]) for s, g in zip(starts, goals)]
    report("goal distance within range", all(args.goal_min - 0.05 <= d <= args.goal_max + 0.05 for d in dists),
           f"(min {min(dists):.2f}, max {max(dists):.2f} m)")
    report("starts vary between episodes", len({(round(x, 2), round(y, 2)) for x, y in starts}) > 1 or args.episodes == 1)
    report("every step advances 0.1 s", all(abs(d - 0.1) < 1e-6 for d in dts), f"({len(dts)} steps)")
    report("no tumble", "TUMBLE" not in outcomes, f"(outcomes {dict((o, outcomes.count(o)) for o in set(outcomes))})")
    if args.log:
        rows = list(csv.DictReader(open(args.log)))[-args.episodes:]      # task_fail and task_succeed alike
        match = all(abs(float(r["goal_x"]) - g[0]) < 1e-3 and abs(float(r["goal_y"]) - g[1]) < 1e-3
                    for r, g in zip(rows, goals)) if len(rows) == len(goals) else False
        report("episode log matches the env's goals", match, f"({len(rows)} logged rows)")
    counts = {o: outcomes.count(o) for o in NAMES.values() if outcomes.count(o)}
    print(f"SUMMARY episodes={len(outcomes)} success={100 * outcomes.count('SUCCESS') / len(outcomes):.1f}% "
          f"mean_steps={np.mean(lengths):.0f} {counts} path_blocked_episodes={sum(s > 0 for s in stalls)}", flush=True)
    print("GRSCENE CHECK PASS" if ok else "GRSCENE CHECK FAIL", flush=True)


if __name__ == "__main__":
    main()
