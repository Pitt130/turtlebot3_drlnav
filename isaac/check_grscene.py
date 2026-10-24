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
    parser.add_argument("--waypoints", type=float, metavar="LOOKAHEAD",
                        help="hierarchical: plan on the scene occupancy grid and give the policy a waypoint this "
                             "far along the path instead of the goal (needs --model and --scene)")
    parser.add_argument("--scene", help="GRScenes scene folder (for --waypoints): its occupancy_v2_*.npz is used")
    parser.add_argument("--plan-inflate", type=float, default=0.18)
    parser.add_argument("--path-timeout", type=float, metavar="K",
                        help="episode time limit max(EPISODE_TIMEOUT_SECONDS, K * grid path length / max speed) "
                             "instead of the fixed limit (needs --scene)")
    args = parser.parse_args()
    env = make_env(step_sim_time=0.1)
    rng = np.random.default_rng(0)
    policy = None
    if args.model:
        import sys
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "sb3"))   # tomas_policy
        from stable_baselines3 import TD3
        policy = TD3.load(args.model, device="cuda")
    planner = None
    if args.waypoints or args.path_timeout:
        import glob, sys
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
        from grscene import Occupancy
        from waypoints import WaypointPlanner
        from turtlebot3_drl.common.settings import EPISODE_TIMEOUT_SECONDS, SPEED_LINEAR_MAX
        from turtlebot3_drl.drl_environment.drl_environment_gymnasium import MAX_GOAL_DISTANCE
        occ = Occupancy.load(sorted(glob.glob(f"{args.scene}/occupancy_v2_*.npz"))[0])
        planner = WaypointPlanner(occ, inflate=args.plan_inflate, lookahead=args.waypoints or 1.5)
        if args.waypoints:
            print(f"waypoint planner: lookahead {args.waypoints} m, planning grid inflated {args.plan_inflate} m", flush=True)
            from geometry_msgs.msg import Pose
            waypoint_pub = env.create_publisher(Pose, "waypoint", 10)     # for the runtime's marker
        if args.path_timeout:
            print(f"time limit: max({EPISODE_TIMEOUT_SECONDS} s, {args.path_timeout} x path / {SPEED_LINEAR_MAX} m/s)", flush=True)
    track_file = open(args.track_out, "w") if args.track_out else None
    if track_file:
        track_file.write("episode,step,x,y,tilt_deg,min_obstacle,goal_distance,lin,ang,outcome\n")
    ok = True

    def report(label, passed, detail=""):
        nonlocal ok
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] {label} {detail}", flush=True)

    starts, goals, min_scans, dts, outcomes, lengths = [], [], [], [], [], []
    for ep in range(args.episodes):
        obs, info = env.reset()
        starts.append((env.robot_x, env.robot_y))
        goals.append((env.goal_x, env.goal_y))
        min_scans.append(min(env.scan_ranges) * 3.5)
        n, max_tilt, tilt_at, track = 0, 0.0, None, []
        unreachable, path_len = 0, math.nan
        if planner is not None:
            planner.set_goal(env.goal_x, env.goal_y)
            path_len = planner.path_length(env.robot_x, env.robot_y)
            if args.path_timeout and math.isfinite(path_len):
                env.episode_deadline = env.time_sec + max(EPISODE_TIMEOUT_SECONDS,
                                                          args.path_timeout * path_len / SPEED_LINEAR_MAX)
        while True:
            if args.waypoints:
                wx, wy, reachable = planner.waypoint(env.robot_x, env.robot_y)
                unreachable += not reachable
                msg = Pose()
                msg.position.x, msg.position.y = float(wx), float(wy)
                waypoint_pub.publish(msg)
                policy_obs = planner.observation_with(obs, env.robot_x, env.robot_y, env.robot_heading, wx, wy,
                                                      MAX_GOAL_DISTANCE)
            else:
                policy_obs = obs
            action = policy.predict(policy_obs, deterministic=True)[0] if policy is not None else rng.uniform(-1, 1, 2).astype(np.float32)
            obs, r, term, trunc, info = env.step(action)
            dts.append(info["step_sim_dt"]); n += 1
            track.append((round(env.robot_x, 3), round(env.robot_y, 3), round(math.degrees(env.robot_tilt), 2)))
            if track_file:
                track_file.write(f"{ep + 1},{n},{env.robot_x:.3f},{env.robot_y:.3f},{math.degrees(env.robot_tilt):.2f},"
                                 f"{env.obstacle_distance:.3f},{env.goal_distance:.3f},{float(action[0]):.3f},{float(action[1]):.3f},"
                                 f"{int(info['success']) if (term or trunc) else 0}\n")
            if env.robot_tilt > max_tilt:
                max_tilt, tilt_at = env.robot_tilt, (env.robot_x, env.robot_y)
            if term or trunc:
                break
        outcomes.append(NAMES[int(info["success"])]); lengths.append(n)
        d = math.hypot(goals[-1][0] - starts[-1][0], goals[-1][1] - starts[-1][1])
        print(f"episode {ep + 1:2d}: start ({starts[-1][0]:+.2f},{starts[-1][1]:+.2f}) goal ({goals[-1][0]:+.2f},"
              f"{goals[-1][1]:+.2f}) dist {d:.2f} m, min scan at start {min_scans[-1]:.2f} m -> {outcomes[-1]} after {n} steps"
              f"; max tilt {math.degrees(max_tilt):.1f} deg at ({tilt_at[0]:+.2f},{tilt_at[1]:+.2f})"
              + (f"; path {path_len:.2f} m" if planner is not None else "")
              + (f"; waypoint unreachable {unreachable} steps" if unreachable else ""), flush=True)
        if outcomes[-1] == "TUMBLE":
            print("   last 8 steps (x, y, tilt deg):", track[-8:], flush=True)
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
          f"mean_steps={np.mean(lengths):.0f} {counts}", flush=True)
    print("GRSCENE CHECK PASS" if ok else "GRSCENE CHECK FAIL", flush=True)


if __name__ == "__main__":
    main()
