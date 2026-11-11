#!/bin/bash
# Train the hierarchical local controller (SB3 TD3, sb3/train.py --waypoints) in one or more GRScenes apartments.
#
#   train_grscene.sh start <name> <total_steps> <chunk_steps> <seed> <scene_dir>...
#   train_grscene.sh status [run_dir]
#   train_grscene.sh stop                    # the current chunk saves its model and the driver exits
#
# One Isaac instance at a time; the apartment rotates every chunk_steps. Each chunk resumes the previous chunk's
# model, VecNormalize and replay buffer, so the buffer holds experience from every apartment. For one apartment,
# pass chunk_steps = total_steps. Results: $RESULTS/<name>/chunk_NN/ (train.py output), checkpoints/,
# final_model.zip, progress.txt. The runtime of chunk i uses task seed seed*1000+i.
#
# Environment:
#   TRAIN_ARGS     train.py settings for every chunk (default below: GRSCENES.md's planner and reward)
#   TRAIN_EXTRA    more train.py args for every chunk, e.g. "--scan-stack 3 --scan-stack-skip 2 --learning-rate 3e-4"
#   INIT_ARGS      train.py args for chunk 1 only, e.g. "--init-from <slow run>/final_model.zip"
#   RUNTIME_EXTRA  runtime args, e.g. "--dyn-obstacles 3 --dyn-min 0"
#   RESULTS (~/drl_results_isaac), USD, PCORES (0-15), ROS_DOMAIN_ID (231), PUBLIC_IP (livestream address;
#   empty disables the stream)
#   CONVERGED_CMD  run with the run dir after every chunk; exit status 0 ends training early (converged), e.g.
#                  "python3 isaac/experiments/converged.py" (training success, last 150k steps vs the 150k before)
#   RESUME=1       continue a stopped run after its last finished chunk (pass the same arguments as before)
#
# touch <run>/STOP stops the driver cleanly before its next chunk, after the running one has finished and saved;
# RESUME=1 with the same arguments then continues from there, e.g. with the livestream switched on (PUBLIC_IP).
#
# The two models of GRSCENES.md:
#   slow: TRAIN_EXTRA="--learning-rate 3e-3 --learning-starts 128 --random-action-until 25000" \
#         train_grscene.sh start slow 300000 300000 0 $S1
#   dyn:  TRAIN_EXTRA="--learning-rate 3e-4 --learning-starts 10000 --random-action-until 0 --scan-stack 3 --scan-stack-skip 2" \
#         INIT_ARGS="--init-from $RESULTS/slow/final_model.zip" RUNTIME_EXTRA="--dyn-obstacles 3 --dyn-min 0" \
#         train_grscene.sh start dyn 150000 150000 0 $S1
cd "$(dirname "$0")/../.." || exit 1
source /opt/ros/jazzy/setup.bash; source install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-231} OMNI_KIT_ACCEPT_EULA=YES
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
USD=${USD:-$HOME/isaac_assets/turtlebot3_burger_usd/turtlebot3_burger_isaac/turtlebot3_burger_isaac.usda}
RESULTS=${RESULTS:-$HOME/drl_results_isaac}; STATE=$RESULTS/.grscene_train
# i7-14700KF: CPUs 0-15 are the P-cores. On an E-core the runtime loop is ~2x slower.
PCORES=${PCORES:-0-15}
STREAM=(); [ -n "${PUBLIC_IP-10.100.100.104}" ] && STREAM=(--livestream --public-ip "${PUBLIC_IP-10.100.100.104}")
TRAIN_ARGS=${TRAIN_ARGS:-"--waypoints 0.75 --plan-inflate 0.25 --plan-low-inflate 0.30 --stall-recovery --path-reward \
  --obstacle-penalty 20 --obstacle-penalty-dist 0.22 --slow-dist 0.35 --slow-min-frac 0.3"}

stop_runtime() {  # chunk dir
  local pid=$(cat $1/run_logs/runtime.pid 2>/dev/null); [ -z "$pid" ] && return
  kill -INT -- -$pid 2>/dev/null
  local j; for j in $(seq 30); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -KILL -- -$pid 2>/dev/null; sleep 5
}

run_all() {  # name total chunk seed scenes...
  local NAME=$1 TOTAL=$2 CHUNK=$3 SEED=$4; shift 4; local SCENES=("$@")
  local RUN=$RESULTS/$NAME prev="" i N=$(( (TOTAL + CHUNK - 1) / CHUNK )) first=1
  mkdir -p $RUN/checkpoints
  printf "%s\n" "${SCENES[@]}" > $RUN/scenes.txt
  if [ "${RESUME:-0}" = 1 ]; then                # skip the chunks that finished and saved their model
    while grep -q "^chunk $first done" $RUN/progress.txt 2>/dev/null && \
          [ -s $RUN/chunk_$(printf %02d $first)/models/final_model.zip ]; do
      prev=$RUN/chunk_$(printf %02d $first); first=$(( first + 1 ))
    done
    rm -f $RUN/STOP
    echo "resumed at chunk $first $(date -u)" >> $RUN/progress.txt
  fi
  for i in $(seq $first $N); do
    [ -e $RUN/STOP ] && { echo "stopped before chunk $i $(date -u)" >> $RUN/progress.txt; return; }
    local SCENE=${SCENES[$(( (i - 1) % ${#SCENES[@]} ))]}
    local D=$RUN/chunk_$(printf %02d $i) CSEED=$(( SEED * 1000 + i ))
    mkdir -p $D/run_logs; echo $D > $RUN/current
    taskset -c $PCORES ~/isaacsim/.venv/bin/python isaac/turtlebot3_isaac_sim.py --usd $USD --grscene $SCENE --rtf 0 \
        --seed $CSEED --low-rays 0.02 --reset-every-episode --episode-log $D/run_logs/episodes.csv $RUNTIME_EXTRA \
        "${STREAM[@]}" > $D/run_logs/isaac_runtime.log 2>&1 < /dev/null &
    # set -m: each background job is its own process group (pgid = pid), so kill -- -pid reaches all of it
    echo $! > $D/run_logs/runtime.pid
    for t in $(seq 600); do grep -qE "isaac_turtlebot3 ready|Traceback" $D/run_logs/isaac_runtime.log && break; sleep 1; done
    if ! grep -q "isaac_turtlebot3 ready" $D/run_logs/isaac_runtime.log; then
      echo "chunk $i: runtime failed on $(basename $SCENE) $(date -u)" >> $RUN/progress.txt; stop_runtime $D; return 1
    fi
    local RES="$INIT_ARGS"
    [ -n "$prev" ] && RES="--resume-model $prev/models/final_model.zip --resume-vecnormalize $prev/models/vecnormalize.pkl \
        --resume-replay-buffer $prev/models/replay_buffer.pkl"
    (cd sb3 && DRLNAV_SIM_BACKEND=isaac DRLNAV_STAGE=4 DRLNAV_TUMBLE_TILT=0.26 taskset -c $PCORES ../.venv/bin/python -u train.py \
        --output-dir $D --total-timesteps $CHUNK --seed $CSEED --checkpoint-freq 50000 \
        --scene $SCENE $TRAIN_ARGS $TRAIN_EXTRA $RES > $D/run_logs/train.log 2>&1 < /dev/null) &
    echo $! > $D/run_logs/train.pid; wait $!
    stop_runtime $D
    if [ ! -s $D/models/final_model.zip ] || [ ! -s $D/models/replay_buffer.pkl ]; then
      echo "chunk $i: training produced no model $(date -u)" >> $RUN/progress.txt; return 1
    fi
    cp $D/models/final_model.zip $RUN/checkpoints/step_$(( i * CHUNK )).zip
    echo "chunk $i done $(date -u) scene $(basename $SCENE)" >> $RUN/progress.txt
    prev=$D
    if [ -n "$CONVERGED_CMD" ] && $CONVERGED_CMD $RUN >> $RUN/progress.txt 2>&1; then
      echo "converged after chunk $i $(date -u)" >> $RUN/progress.txt; break
    fi
  done
  cp $prev/models/final_model.zip $RUN/final_model.zip
  echo "ALL DONE $(date -u)" >> $RUN/progress.txt
}

case "$1" in
start)
  shift; NAME=$1; RUN=$RESULTS/$NAME
  [ $# -ge 5 ] || { sed -n 2,6p "$0"; exit 1; }
  [ -e "$RUN" ] && { echo "$RUN exists"; exit 1; }
  mkdir -p $RUN; echo $RUN > $STATE
  set -m
  nohup bash $0 _run "$@" > $RUN/driver.log 2>&1 < /dev/null &
  echo "started $RUN (driver pid $!)"
  ;;
_run)
  shift; set -m; run_all "$@"
  ;;
status)
  RUN=${2:-$(cat $STATE)}; D=$(cat $RUN/current 2>/dev/null)
  tail -3 $RUN/progress.txt 2>/dev/null; echo "current: $D"
  for f in runtime train; do kill -0 $(cat $D/run_logs/$f.pid 2>/dev/null) 2>/dev/null && echo "$f: running" || echo "$f: not running"; done
  .venv/bin/python - $RUN <<'PY'
import csv, sys
from pathlib import Path
run = Path(sys.argv[1]); rows = []
names = {"1": "SUCCESS", "2": "COLL_WALL", "3": "COLL_OBST", "4": "TIMEOUT", "5": "TUMBLE"}
for d in sorted(run.glob("chunk_*")):
    mon = d / "logs" / "monitor.csv"
    if mon.exists():
        part = list(csv.DictReader(mon.open().readlines()[1:]))
        for r in part: r["chunk"] = d.name
        rows += part
if not rows:
    print("no episodes yet"); sys.exit()
steps = sum(int(r["l"]) for r in rows)
last = rows[-100:]; counts = {}
for r in last:
    k = names.get(str(int(float(r["success"]))), r["success"]); counts[k] = counts.get(k, 0) + 1
print(f"{run.name}: {len(rows)} episodes, {steps} steps")
print(f"last {len(last)}: success {100 * counts.get('SUCCESS', 0) / len(last):.0f}%  {counts}")
by = {}
for r in rows: by.setdefault(r["chunk"], []).append(int(float(r["success"])))
print("by chunk:", "  ".join(f"{k[-2:]}:{100*v.count(1)/len(v):.0f}%" for k, v in by.items()))
PY
  grep -E "Traceback|Error" $D/run_logs/train.log 2>/dev/null | grep -v rosout | tail -2
  ;;
stop)
  RUN=$(cat $STATE); D=$(cat $RUN/current); touch $RUN/STOP
  kill -INT -- -$(cat $D/run_logs/train.pid) 2>/dev/null
  echo "stop requested; the current chunk saves and the driver exits"
  ;;
*)
  sed -n 2,6p "$0"; exit 1
  ;;
esac
