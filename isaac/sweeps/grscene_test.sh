#!/bin/bash
# Launch the Isaac runtime on a GRScenes apartment and run check_grscene.py through the env.
#   OUT=<dir> CHECK_ARGS="--model ... --track-out ..." grscene_test.sh <episodes> [extra runtime args...]
# Environment: SCENE (scene folder), OUT (must be new or empty), SEED (task seed, default 0), CPUS (taskset list,
# default 0-15), ROS_DOMAIN_ID (default 230, keep <= 232 and distinct per parallel run), CHECK_TIMEOUT (s), USD.
# Output: $OUT/check.log (filtered), $OUT/check_raw.log (everything), $OUT/episodes.csv, $OUT/runtime.log
EPISODES=${1:-10}; shift
cd "$(dirname "$0")/../.." || exit 1
source /opt/ros/jazzy/setup.bash; source install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-230} OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
USD=${USD:-$HOME/isaac_assets/turtlebot3_burger_usd/turtlebot3_burger_isaac/turtlebot3_burger_isaac.usda}
SCENE=${SCENE:-$HOME/grscenes/home_scenes/scenes/MVUHLWYKTKJ5EAABAAAAAAI8_usd}
OUT=${OUT:-$HOME/drl_results_isaac/grscene_test}
if [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then echo "$OUT is not empty; move it away first"; exit 1; fi
mkdir -p "$OUT"
setsid taskset -c ${CPUS:-0-15} ~/isaacsim/.venv/bin/python isaac/turtlebot3_isaac_sim.py --usd $USD --grscene $SCENE --rtf 0 \
    --seed ${SEED:-0} --episode-log $OUT/episodes.csv "$@" > $OUT/runtime.log 2>&1 < /dev/null &
PG=$!; echo $PG > $OUT/runtime.pid
for i in $(seq 600); do grep -qE "isaac_turtlebot3 ready|Traceback" $OUT/runtime.log && break; sleep 1; done
grep -E "grscene |occupancy |isaac_turtlebot3 ready" $OUT/runtime.log | cut -c1-200
grep -A12 Traceback $OUT/runtime.log | grep -vE "^\[|^\s*$" | tail -8 | cut -c1-200
grep -q "isaac_turtlebot3 ready" $OUT/runtime.log || { kill -INT -- -$PG 2>/dev/null; sleep 5; kill -KILL -- -$PG 2>/dev/null; exit 1; }
DRLNAV_SIM_BACKEND=isaac DRLNAV_STAGE=4 DRLNAV_TUMBLE_TILT=${DRLNAV_TUMBLE_TILT:-0.26} timeout ${CHECK_TIMEOUT:-1800} \
    .venv/bin/python isaac/check_grscene.py $EPISODES --log $OUT/episodes.csv $CHECK_ARGS > $OUT/check_raw.log 2>&1
grep -E "^episode|^   last|^\[|CHECK|Traceback|Error|error" $OUT/check_raw.log | cut -c1-260 | tee $OUT/check.log
echo "--- runtime errors:"; grep -iE "Traceback|\[Error\]" $OUT/runtime.log | grep -viE "omni.hydra|grpc|protobuf|rtx.materials|neuray|USD_MDL|MDLC" | tail -5 | cut -c1-200
kill -INT -- -$PG 2>/dev/null; for i in $(seq 30); do kill -0 $PG 2>/dev/null || break; sleep 1; done; kill -KILL -- -$PG 2>/dev/null
head -3 $OUT/episodes.csv
