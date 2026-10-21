#!/bin/bash
# SB3 TD3 training on the Isaac runtime, with sb3/train.py's defaults (the
# settings of the Gazebo stage 4 baselines td3_seed1/2).
#
#   bash isaac/train_isaac.sh start [stage] [seed] [steps]
#   bash isaac/train_isaac.sh status [run_dir]
#   bash isaac/train_isaac.sh stop      # SIGINT: train.py saves model + replay buffer
#
# Environment: RESULTS (default ~/drl_results_isaac), USD (robot USD), PUBLIC_IP
# (address livestream clients use; empty disables the stream), RTF (0 = run as
# fast as possible; all env timing is sim time), PCORES (CPU list for taskset).
# Run from the repository root after `source install/setup.bash`.
cd "$(dirname "$0")/.." || exit 1
source /opt/ros/jazzy/setup.bash; source install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-231} OMNI_KIT_ACCEPT_EULA=YES
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
USD=${USD:-$HOME/isaac_assets/turtlebot3_burger_usd/turtlebot3_burger_isaac/turtlebot3_burger_isaac.usda}
RESULTS=${RESULTS:-$HOME/drl_results_isaac}
STAGE=${2:-4}; SEED=${3:-0}; STEPS=${4:-300000}; RTF=${RTF:-0}
# i7-14700KF: CPUs 0-15 are the P-cores. On an E-core the runtime loop is ~2x slower.
PCORES=${PCORES:-0-15}
STREAM=(); [ -n "${PUBLIC_IP-10.100.100.104}" ] && STREAM=(--livestream --public-ip "${PUBLIC_IP-10.100.100.104}")
RUN=$RESULTS/isaac_s${STAGE}_base_seed${SEED}
STATE=$RESULTS/.current_run

case "$1" in
start)
  [ -e "$RUN" ] && { echo "$RUN exists"; exit 1; }
  mkdir -p "$RUN/run_logs"; echo "$RUN" > "$STATE"
  setsid taskset -c "$PCORES" ~/isaacsim/.venv/bin/python isaac/turtlebot3_isaac_sim.py --usd "$USD" --stage "$STAGE" \
      --rtf "$RTF" "${STREAM[@]}" > "$RUN/run_logs/isaac_runtime.log" 2>&1 < /dev/null &
  echo $! > "$RUN/run_logs/runtime.pid"
  for i in $(seq 240); do grep -qE "isaac_turtlebot3 ready|Traceback" "$RUN/run_logs/isaac_runtime.log" && break; sleep 1; done
  grep -E "isaac_turtlebot3 ready" "$RUN/run_logs/isaac_runtime.log" || { tail -5 "$RUN/run_logs/isaac_runtime.log"; exit 1; }
  DRLNAV_SIM_BACKEND=isaac DRLNAV_STAGE=$STAGE setsid taskset -c "$PCORES" .venv/bin/python -u sb3/train.py --output-dir "$RUN" \
      --total-timesteps "$STEPS" --seed "$SEED" --checkpoint-freq 50000 > "$RUN/run_logs/train.log" 2>&1 < /dev/null &
  echo $! > "$RUN/run_logs/train.pid"
  # Stop the runtime when training ends.
  setsid nohup bash -c "while kill -0 $(cat "$RUN/run_logs/train.pid") 2>/dev/null; do sleep 60; done; \
      kill -INT -- -$(cat "$RUN/run_logs/runtime.pid") 2>/dev/null; sleep 10; kill -KILL -- -$(cat "$RUN/run_logs/runtime.pid") 2>/dev/null" \
      > /dev/null 2>&1 < /dev/null &
  echo "started $RUN"
  ;;
status)
  RUN=${2:-$(cat "$STATE" 2>/dev/null)}; [ -n "$RUN" ] || { echo "no run started by this script yet"; exit 1; }
  for f in runtime train; do kill -0 "$(cat "$RUN/run_logs/$f.pid" 2>/dev/null)" 2>/dev/null && echo "$f: running" || echo "$f: not running"; done
  .venv/bin/python - "$RUN" <<'PY'
import csv, sys
from pathlib import Path
run = Path(sys.argv[1]); mon = run / "logs" / "monitor.csv"
if not mon.exists():
    print("no monitor.csv yet"); sys.exit()
rows = list(csv.DictReader(mon.open().readlines()[1:]))
names = {"1": "SUCCESS", "2": "COLL_WALL", "3": "COLL_OBST", "4": "TIMEOUT", "5": "TUMBLE"}
steps = sum(int(r["l"]) for r in rows); wall = float(rows[-1]["t"]) if rows else 0.0
last = rows[-100:]; counts = {}
for r in last:
    k = names.get(str(int(float(r["success"]))), r["success"]); counts[k] = counts.get(k, 0) + 1
print(f"{run.name}: {len(rows)} episodes, {steps} steps, wall {wall / 3600:.2f} h, sim/wall {steps * 0.1 / max(wall, 1):.2f}")
print(f"last {len(last)}: success {100 * counts.get('SUCCESS', 0) / max(len(last), 1):.0f}%  {counts}")
PY
  grep -E "Traceback|Error" "$RUN/run_logs/train.log" | tail -3
  ;;
stop)
  RUN=$(cat "$STATE")
  kill -INT "$(cat "$RUN/run_logs/train.pid")" 2>/dev/null
  for i in $(seq 120); do kill -0 "$(cat "$RUN/run_logs/train.pid")" 2>/dev/null || break; sleep 1; done
  tail -2 "$RUN/run_logs/train.log"
  kill -INT -- -"$(cat "$RUN/run_logs/runtime.pid")" 2>/dev/null; sleep 5; kill -KILL -- -"$(cat "$RUN/run_logs/runtime.pid")" 2>/dev/null
  pgrep -af "[t]urtlebot3_isaac_sim|[s]b3/train.py" || echo "stopped"
  ;;
*) echo "usage: $0 start [stage] [seed] [steps] | status [run_dir] | stop"; exit 2 ;;
esac
