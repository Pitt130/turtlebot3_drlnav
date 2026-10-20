"""Run the real Gymnasium agent for several episodes against the Isaac runtime.

Start turtlebot3_isaac_sim.py --stage N first, then:
    DRLNAV_SIM_BACKEND=isaac DRLNAV_STAGE=N .venv/bin/python isaac/check_episodes.py [episodes] [timeout_s]

Checks every reset (robot back at the stage start after a failure, obstacles
back at their start, goal from the stage list and >= 2 m from the previous one), every
step (0.1 s of sim time, replay done == terminated), and task_succeed (new goal,
robot not moved).
"""
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import rclpy
from turtlebot3_msgs.srv import RingGoal

from turtlebot3_drl.common.goal_poses import GOALS_STAGE_4_5_7, GOALS_STAGE_8_9_12, GOALS_STAGE_11
from turtlebot3_drl.drl_agent.drl_agent_gymnasium import DrlAgentGymnasium

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_scene import load_stage  # noqa: E402

EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 12
TIMEOUT_S = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
NAMES = {0: "UNKNOWN", 1: "SUCCESS", 2: "COLL_WALL", 3: "COLL_OBST", 4: "TIMEOUT", 5: "TUMBLE"}
STAGE = int(os.environ.get("DRLNAV_STAGE", "4"))
SCENE = load_stage(Path(__file__).resolve().parent.parent, STAGE)
# The env indexes obstacles by the first "obstacleN" in the p3d child frame, minus one.
OBSTACLE_STARTS = {int(re.search(r"obstacle(\d+)", o.frame_id).group(1)) - 1: tuple(o.pose(0.0)[0][:2])
                   for o in SCENE.obstacles}
ROBOT_START = SCENE.robot_start[:2]
GOAL_LIST = {4: GOALS_STAGE_4_5_7, 5: GOALS_STAGE_4_5_7, 7: GOALS_STAGE_4_5_7, 8: GOALS_STAGE_8_9_12,
             9: GOALS_STAGE_8_9_12, 12: GOALS_STAGE_8_9_12, 11: GOALS_STAGE_11}.get(STAGE)


def main():
    agent = DrlAgentGymnasium(training=True, algorithm="td3", enable_storage=False)
    env = agent.env
    env.episode_timeout = TIMEOUT_S
    ok = True

    def report(label, passed, detail=""):
        nonlocal ok
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] {label} {detail}", flush=True)

    resets, records, dts, last = [], [], [], {}
    env_reset, env_step = env.reset, env.step

    def reset(**kwargs):
        obs, info = env_reset(**kwargs)
        obstacles = {k: (round(x, 3), round(y, 3)) for k, (_, x, y) in env._obstacle_samples.items()}
        resets.append(dict(pose=info["robot_pose"], goal=(env.goal_x, env.goal_y), time=info["time_sec"],
                           obstacles=obstacles))
        return obs, info

    def step(action):
        out = env_step(action)
        last.update(terminated=bool(out[2]), truncated=bool(out[3]), outcome=int(out[4]["success"]))
        dts.append(out[4]["step_sim_dt"])
        return out

    env.reset, env.step = reset, step
    add = agent.replay_buffer.add_sample
    agent.replay_buffer.add_sample = lambda *t: (records.append(dict(last, done=bool(t[4][0]))), add(*t))[1]

    outcomes = []
    t0 = time.time()
    try:
        for ep in range(EPISODES):
            n0 = len(records)
            agent.finish_episode(**agent.run_episode())
            final = records[-1]
            outcomes.append(NAMES[final["outcome"]])
            r = resets[-1]
            print(f"episode {ep + 1:2d}: start pose ({r['pose'][0]:+.3f},{r['pose'][1]:+.3f}) goal {r['goal']} "
                  f"-> {outcomes[-1]} after {len(records) - n0} steps "
                  f"(terminated={final['terminated']}, truncated={final['truncated']}, done stored={final['done']})",
                  flush=True)

        # Resets: the first and every reset after a failure start at the stage start.
        bad_pose = [i for i, r in enumerate(resets) if (i == 0 or outcomes[i - 1] != "SUCCESS")
                    and math.hypot(r["pose"][0] - ROBOT_START[0], r["pose"][1] - ROBOT_START[1]) > 0.05]
        report("robot back at the start after task_fail", not bad_pose, f"({len(resets)} resets; bad: {bad_pose})")
        bad_obst = [i for i, r in enumerate(resets) if (i == 0 or outcomes[i - 1] != "SUCCESS")
                    and any(math.hypot(r["obstacles"][k][0] - sx, r["obstacles"][k][1] - sy) > 0.15
                            for k, (sx, sy) in OBSTACLE_STARTS.items() if k in r["obstacles"])]
        seen = sorted({k for r in resets for k in r["obstacles"]})
        report("obstacles restart with the episode", not bad_obst and seen == sorted(OBSTACLE_STARTS),
               f"({len(seen)}/{len(OBSTACLE_STARTS)} obstacles seen; bad: {bad_obst}; e.g. {resets[-1]['obstacles']})")
        goals = [r["goal"] for r in resets]
        in_list = GOAL_LIST is None or all(list(g) in GOAL_LIST for g in goals)
        apart = all(abs(a[0] - b[0]) + abs(a[1] - b[1]) >= 2 for a, b in zip(goals, goals[1:]))
        report(f"goals from the stage {STAGE} list, >= 2 m apart", in_list and apart, f"{goals}")
        report("every step advances 0.1 s", all(abs(d - 0.1) < 1e-6 for d in dts),
               f"({len(dts)} steps, min {min(dts):.4f}, max {max(dts):.4f})")
        report("replay done == terminated", all(r["done"] == r["terminated"] for r in records),
               f"({len(records)} transitions; outcomes {dict((o, outcomes.count(o)) for o in set(outcomes))})")

        # task_succeed: a new goal, robot left where it is.
        env._set_physics(False)
        env._spin_until(lambda: False, 0.5)        # let the robot come to rest first
        before = (env.robot_x, env.robot_y, env.goal_x, env.goal_y)
        client = env.create_client(RingGoal, "task_succeed")
        env.new_goal = False
        future = client.call_async(RingGoal.Request())
        env._spin_until(lambda: future.done() and env.new_goal, 5.0)
        env._spin_until(lambda: False, 0.3)
        env._set_physics(True)
        moved = math.hypot(env.robot_x - before[0], env.robot_y - before[1])
        report("task_succeed: new goal, robot not reset",
               future.done() and (env.goal_x, env.goal_y) != before[2:] and moved < 0.05,
               f"goal {before[2:]} -> {(env.goal_x, env.goal_y)}, robot moved {moved:.3f} m")
    finally:
        env.close()
    print(f"wall time {time.time() - t0:.0f} s", flush=True)
    print("EPISODE CHECK PASS" if ok else "EPISODE CHECK FAIL", flush=True)


if __name__ == "__main__":
    main()
