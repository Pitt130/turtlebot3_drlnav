#!/bin/bash
# Evaluate one model on every apartment: the same tasks (task seed SEED) and planner settings as GRSCENES.md,
# deterministic policy, 40-ray LDS + 2 cm low rays, N moving obstacles in every episode (0 = static).
#   eval_scenes.sh <label> <model.zip> <obstacles> <cpus> <ros_domain_id> [extra runtime args...]
#   e.g. eval_scenes.sh dyn_steer <run>/final_model.zip 3 0-9 224 --dyn-yield 0 --dyn-avoid 0.5
# Environment: PLAN_EXTRA (more check_grscene.py planner args, e.g. "--plan-map occupancy_v6 --plan-cost 3"), OUT_BASE (default ~/drl_results_isaac/grscene_eval), EPISODES (100), SEED (0),
# SCENES ("name:scene_id ..." under SCENES_DIR; default the five apartments of GRSCENES.md).
# Results: $OUT_BASE/<label>_<scene>/ (check_raw.log ends with a SUMMARY line), <label>_<scene>.track.csv,
# and one line per apartment in $OUT_BASE/progress.txt. Runs the apartments one after another.
cd "$(dirname "$0")/../.." || exit 1
LABEL=$1; M=$2; N=$3; CPUSET=$4; DOMAIN=$5; shift 5; EXTRA="$*"
[ -n "$DOMAIN" ] || { sed -n 2,9p "$0"; exit 1; }
OUT_BASE=${OUT_BASE:-$HOME/drl_results_isaac/grscene_eval}; mkdir -p $OUT_BASE
SCENES_DIR=${SCENES_DIR:-$HOME/grscenes/home_scenes/scenes}
SCENES=${SCENES:-"s1:MVUHLWYKTKJ5EAABAAAAAAI8_usd s2:MVUCSQAKTKJ5EAABAAAAACI8_usd dy8:MVUCSQAKTKJ5EAABAAAAADY8_usd \
ei8:MVUCSQAKTKJ5EAABAAAAAEI8_usd di8:MV7J6NIKTKJZ2AABAAAAADI8_usd"}
FIX="--waypoints 0.75 --plan-inflate 0.25 --plan-low-inflate 0.30 --stall-recovery ${PLAN_EXTRA:-}"
DYN=""; [ "$N" -gt 0 ] && DYN="--dyn-obstacles $N"
echo "$LABEL start $(date -u) model $M obstacles $N $EXTRA" >> $OUT_BASE/progress.txt
for entry in $SCENES; do
  name=${entry%%:*}; scene=$SCENES_DIR/${entry#*:}
  SCENE=$scene OUT=$OUT_BASE/${LABEL}_$name ROS_DOMAIN_ID=$DOMAIN SEED=${SEED:-0} CPUS=$CPUSET CHECK_TIMEOUT=7200 \
    CHECK_ARGS="--model $M --scene $scene --track-out $OUT_BASE/${LABEL}_$name.track.csv $FIX" \
    bash isaac/sweeps/grscene_test.sh ${EPISODES:-100} --low-rays 0.02 --reset-every-episode $DYN $EXTRA \
    > $OUT_BASE/${LABEL}_$name.out 2>&1
  echo "${LABEL}_$name done $(date -u): $(grep -E "^SUMMARY" $OUT_BASE/${LABEL}_$name/check_raw.log 2>/dev/null)" >> $OUT_BASE/progress.txt
done
echo "$LABEL ALL DONE $(date -u)" >> $OUT_BASE/progress.txt
