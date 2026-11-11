#!/bin/bash
# Fixes: 0.30 m planning margin around obstacles below the lidar plane + stall recovery. Same task set as sweep_v3.
cd ~/turtlebot3_drlnav
BASE=~/drl_results_isaac/grscene_sweep_v4; mkdir -p $BASE
SCENE=$HOME/grscenes/home_scenes/scenes/MVUHLWYKTKJ5EAABAAAAAAI8_usd
MODEL=$HOME/drl_results_isaac/isaac_s4_base_seed0/models/final_model.zip
run() {  # name domain extra-check-args
  OUT=$BASE/$1 ROS_DOMAIN_ID=$2 SEED=0 CHECK_TIMEOUT=5400 \
    CHECK_ARGS="--model $MODEL --scene $SCENE --track-out $BASE/$1.track.csv $3" \
    bash isaac/sweeps/grscene_test.sh 100 --low-rays 0.02 --reset-every-episode > $BASE/$1.out 2>&1
}
FIX="--plan-low-inflate 0.30 --stall-recovery"
run la0.75_fix 225 "--waypoints 0.75 $FIX" &
for i in $(seq 300); do grep -q "isaac_turtlebot3 ready" $BASE/la0.75_fix/runtime.log 2>/dev/null && break; sleep 2; done
run la0.5_fix 226 "--waypoints 0.5 $FIX" &
wait
V3=~/drl_results_isaac/grscene_sweep_v3
args=("e2e=$V3/e2e" "la0.5=$V3/la0.5" "la0.75=$V3/la0.75" "la0.5_fix=$BASE/la0.5_fix" "la0.75_fix=$BASE/la0.75_fix")
{ for d in la0.5_fix la0.75_fix; do echo "== $d: $(grep -E "^SUMMARY" $BASE/$d/check_raw.log)"; done
  grep -E "occupancy:" $BASE/la0.75_fix/runtime.log
  echo "same tasks as v3: $(cmp -s <(cut -d, -f4,5,7,8 $V3/la0.75/episodes.csv) <(cut -d, -f4,5,7,8 $BASE/la0.75_fix/episodes.csv) && echo yes || echo NO)"
  .venv/bin/python isaac/difficulty_breakdown.py $SCENE "${args[@]}"
  .venv/bin/python isaac/experiments/classify_timeouts.py --dir $BASE la0.5_fix la0.75_fix; } > $BASE/report.txt 2>&1
echo "report written $(date -u)" >> $BASE/progress.txt
