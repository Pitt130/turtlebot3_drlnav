#!/usr/bin/env bash
set -eo pipefail
# Source ROS, the built workspace and the SB3 virtual environment first.
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-28}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export DRLNAV_BASE_PATH="$PWD" TURTLEBOT3_MODEL=burger PYTHONUNBUFFERED=1
run_dir="experiment_results/sb3_td3_original_params_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"
python sb3/train.py \
  --output-dir "$run_dir" \
  --total-timesteps "${1:-200000}" --checkpoint-freq 2500 --seed "${2:-0}" \
  --learning-rate 0.003 --buffer-size 1000000 \
  --learning-starts 128 --random-action-until 25000 --batch-size 128 \
  --tau 0.003 --gamma 0.99 --noise ou --noise-sigma 0.1 --noise-dt 1.0 \
  --step-sim-time 0.1 --no-normalize-reward --critic-loss smooth_l1 \
  --max-grad-norm 2.0 --optimizer adamw \
  --critic-architecture tomas --weight-init tomas \
  2>&1 | tee "$run_dir/train.log"
