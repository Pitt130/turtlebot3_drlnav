#!/bin/bash
# Lookahead sweep for hierarchical navigation in the GRScenes apartment.
# Every config sees the same task sequence (--seed 0 --reset-every-episode), 3 configs in parallel per batch.
#   lookahead_sweep.sh <episodes>      results: ~/drl_results_isaac/grscene_lookahead_sweep/<name>/, report.txt
EPISODES=${1:-100}
cd ~/turtlebot3_drlnav
BASE=~/drl_results_isaac/grscene_lookahead_sweep; mkdir -p $BASE
SCENE=$HOME/grscenes/home_scenes/scenes/MVUHLWYKTKJ5EAABAAAAAAI8_usd
MODEL=$HOME/drl_results_isaac/isaac_s4_base_seed0/models/final_model.zip
COMMON="--model $MODEL --scene $SCENE"

run() {   # name domain "check args"
  local name=$1 domain=$2 cargs=$3
  OUT=$BASE/$name ROS_DOMAIN_ID=$domain SEED=0 CHECK_TIMEOUT=5400 CHECK_ARGS="$COMMON $cargs" \
    bash isaac/sweeps/grscene_test.sh $EPISODES --low-rays 0.04 --reset-every-episode > $BASE/$name.out 2>&1
}

batch() {
  run "$1" 225 "$2" & run "$3" 226 "$4" & run "$5" 227 "$6" &
  wait
}

echo "batch 1 $(date -u)" > $BASE/progress.txt
batch e2e_50s "" la1.0_50s "--waypoints 1.0" la1.5_50s "--waypoints 1.5"
echo "batch 2 $(date -u)" >> $BASE/progress.txt
batch la2.0_50s "--waypoints 2.0" la2.5_50s "--waypoints 2.5" la1.5_path2 "--waypoints 1.5 --path-timeout 2"
echo "done $(date -u)" >> $BASE/progress.txt

{
  for d in e2e_50s la1.0_50s la1.5_50s la2.0_50s la2.5_50s la1.5_path2; do
    echo "== $d: $(grep -E '^SUMMARY' $BASE/$d/check_raw.log)  $(grep -c 'GRSCENE CHECK PASS' $BASE/$d/check_raw.log) pass"
  done
  args=()
  for d in e2e_50s la1.0_50s la1.5_50s la2.0_50s la2.5_50s la1.5_path2; do args+=("$d=$BASE/$d"); done
  .venv/bin/python isaac/difficulty_breakdown.py $SCENE "${args[@]}"
} > $BASE/report.txt 2>&1
echo "report written $(date -u)" >> $BASE/progress.txt
