# Navigation in real apartments (GRScenes)

Status: 2026-11-02, branch `isaac-migration`. This document continues [MIGRATION.md](MIGRATION.md), which covers the Isaac Sim runtime and the stage 4 baseline. Here the TurtleBot3 Burger navigates furnished apartments from [GRScenes](https://huggingface.co/datasets/InternRobotics/GRScenes) with a hierarchical controller: a grid planner picks a waypoint, and a TD3 policy drives to it on lidar.

## Summary

- **Main model**: `grscene_hier_td3_dyn3_stack3_from_slow035_lr3e4_seed0` ("dyn"). It is trained in one apartment and evaluated in five, four of which it has never seen:

  | Condition | Success |
  |---|---|
  | No moving obstacles | **90.0%** |
  | 3 pedestrians that yield | 86.4% |
  | 3 pedestrians that steer around the robot | 83.0% |
  | 3 pedestrians that never yield | 63.4% |

- **What made the difference**:
  1. Hierarchical navigation instead of end-to-end: 85% vs 54% in the training apartment.
  2. The original proximity penalty: −20 per step within 0.22 m.
  3. Low raycasts that see furniture bases and shoes.
  4. Fine-tuning with moving obstacles and stacked scans.
- **What did not help**:
  - Fine-tuning the stage 4 model directly in an apartment
  - Multi-apartment training (tried twice)
  - A finer body-height planning grid (v5)
  - 360-ray lidar pooled into 40 inputs
- **Remaining bottleneck: perception.**
  - Most remaining failures are collisions with chair and table legs. They are 2–4 cm thick and fall between the 40 lidar rays, which are 9.2° apart.
  - Furniture above or below the 17 cm scan plane is also missed.
  - These failures concentrate in two apartments, S2 (75–78%) and DY8 (83%).

## Setup

### Scenes

GRScenes is licensed CC BY-NC-SA 4.0, for non-commercial use only. Scenes live under `~/grscenes/home_scenes/scenes/<id>_usd/`, with the shared material library in `~/grscenes/home_scenes/Materials/`.

| Name | Scene id | Role |
|---|---|---|
| S1 | `MVUHLWYKTKJ5EAABAAAAAAI8` | training apartment, 23 m² usable free space |
| S2 | `MVUCSQAKTKJ5EAABAAAAACI8` | unseen; the hardest (open plan, many chairs) |
| DY8 | `MVUCSQAKTKJ5EAABAAAAADY8` | unseen |
| EI8 | `MVUCSQAKTKJ5EAABAAAAAEI8` | unseen |
| DI8 | `MV7J6NIKTKJZ2AABAAAAADI8` | unseen |
| BA8 | `MVUCSQAKTKJ5EAABAAAAABA8` | excluded: broken scene data |

**Downloading scenes.** The dataset ships as a 51 GB split zip (`home_scenes.z01/.z02/.zip`). `grscenes_fetch.py` reads the zip's central directory over HTTP range requests, then extracts one scene together with the models and materials it references (about 1 GB):

```bash
python3 grscenes_fetch.py list                  # scenes with their asset sizes
python3 grscenes_fetch.py get <scene_id> <out>  # extract one scene
```

The tool is currently at `~/grscenes_tools/` on the training server and will move into this repository.

**Loading** (`isaac/grscene.py`, runtime `--grscene DIR`):
- Units are converted from cm to m, and the scene's own physics scene is deactivated.
- MDL material paths are made absolute.
- Ceilings and the aerial backdrop are hidden, and a dome light is added.
- Rugs (meshes named `blanket`, under 5 cm) lose their colliders: a 1.5 cm rug edge stops the Burger's 1 cm caster.

### Occupancy grid and tasks

- **Grid**: on first load the runtime builds a 5 cm occupancy grid by raycasting and caches it in the scene folder as `occupancy_v4_5cm_172mm_5mm.npz`. Each cell is one of:
  - **OCCUPIED**: something at the lidar height (17.2 cm)
  - **LOW**: something on the floor above 0.5 cm, entirely below the lidar plane, such as shoes or bed bases
  - **FREE**
- **Starts**: drawn from the largest free component, with a 0.25 m margin.
- **Goals**: 1.5–5.5 m from the start in a straight line, and at most 7 m along the grid path.
- **Reproducibility**: with `--seed S --reset-every-episode`, the sequence of tasks depends only on the seed. All evaluations use seed 0, so every model faces the same 100 tasks.

## Method

### Hierarchical navigation

```
goal ──► grid A* planner (isaac/waypoints.py) ──► waypoint 0.75 m ahead on the path
                                                        │
                    lidar scans (3 frames) + waypoint ──► TD3 policy ──► (v, ω) ──► /cmd_vel
```

- **Planning margins**: 0.25 m around obstacles the lidar sees, 0.30 m around LOW cells.
- **Stall recovery**: after 3 s with less than 0.1 m of path progress, the waypoint moves 1.5 m ahead for 2 s.
- **Implementation**: `isaac/waypoint_env.py` wraps the Gymnasium environment for training. `isaac/check_grscene.py` does the same for evaluation.

### Robot and sensors

| Item | Value |
|---|---|
| Robot | TurtleBot3 Burger, differential drive |
| Wheels | radius 0.033 m, separation 0.16 m |
| Rim acceleration | limited to 1 m/s², as in Gazebo's diff-drive plugin |
| Action | body velocities: v in 0–0.22 m/s (no reverse), ω in ±2.0 rad/s |
| Lidar (LDS) | 0.172 m above the floor, 0.032 m behind the centre; tilts with the body |
| Lidar rays | 40 rays over 360° (9.2° apart) |
| Lidar range and rate | 0.12–3.5 m, Gaussian noise σ = 0.01 m, 50 Hz |
| Low ray layer | 0.02 m above the floor, always horizontal, same 40 directions |
| Combined reading | per-direction minimum of the lidar and the low layer; low hits closer than 0.12 m read as 0.12 m |
| Collision | minimum combined range below 0.13 m |
| Tumble | tilt above 0.26 rad (`DRLNAV_TUMBLE_TILT=0.26`; rug edges tilt the robot 4–6°) |
| Episode limit | 50 s, i.e. 500 control steps of 0.1 s |
| Simulation rates | physics 200 Hz, sensors 50 Hz, control 10 Hz |

The low ray layer has no counterpart on the real Burger. A real deployment needs either retraining without it or a depth camera that provides the same information.

### Observation (124 values)

- 3 lidar scans of 40 rays each, 2 control steps apart: [scan(t−4), scan(t−2), scan(t)]
- Distance and angle to the waypoint
- The previous action

`train.py --init-from` starts a model with a longer observation from one with a shorter observation. The first layers get zero weights for the new inputs, so the new model initially behaves exactly like the old one.

### Reward (`--path-reward`)

Per step:

```
r = −|θ_waypoint| − ω² − 10²·(v_target − v)² + 50·Δpath − 1 − 20·[d_min < 0.22 m]
```

- `Δpath` is the decrease in grid-path length to the goal, clipped to ±0.2 m.
- `v_target` is 0.22 m/s. Within 0.35 m of an obstacle it drops linearly to 30% of that (`--slow-dist 0.35 --slow-min-frac 0.3`).
- Terminal rewards: success +2500, collision −2000, timeout −500, tumble −1000.

A weaker proximity penalty (−10 within 0.15 m) made the policy reckless: 62% success instead of 96%.

### Moving obstacles (`isaac/dynamic_obstacles.py`)

| Option | Meaning |
|---|---|
| `--dyn-obstacles N`, `--dyn-min M` | M–N kinematic cylinders per episode |
| `--dyn-radius` | radius 0.12 m |
| `--dyn-speed` | speed 0.2–0.5 m/s, drawn for each leg of the walk |
| `--dyn-yield D` | yield: wait rather than come closer than D (default 0.45 m, centre to centre) |
| `--dyn-yield-prob P` | chance that a given obstacle yields at all |
| `--dyn-avoid A` | pedestrian steering: turn 30/60/90° (away side first) to keep a predicted miss distance of A over 2 s |

- The obstacles walk between random free cells along grid paths.
- Obstacle draws depend on the seed and episode, so models meet the same obstacles.
- A hit is `COLL_OBST`: centre distance below 0.13 + 0.16 + 0.05 m.

### TD3

| Parameter | slow (from scratch) | dyn (fine-tune) |
|---|---|---|
| Steps | 300k | 150k |
| Learning rate | 3e-3 | **3e-4** |
| Learning starts | 128 | 10k |
| Random actions until | 25k steps | 0 |
| Scans per observation | 1 | 3 (skip 2) |
| Moving obstacles | none | 0–3, yielding |

Shared settings:
- batch 128, buffer 1M, γ 0.99, τ 0.003
- OU noise σ 0.1
- AdamW, gradient clipping 2.0, smooth-L1 critic loss
- seed 0

Fine-tuning at learning rate 3e-3 with a fresh optimizer wrecks the pretrained policy.

## Usage

All commands run from the repository root on the training server.

```bash
source /opt/ros/jazzy/setup.bash; source install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=231 OMNI_KIT_ACCEPT_EULA=YES
export DRLNAV_SIM_BACKEND=isaac DRLNAV_STAGE=4 DRLNAV_TUMBLE_TILT=0.26
USD=~/isaac_assets/turtlebot3_burger_usd/turtlebot3_burger_isaac/turtlebot3_burger_isaac.usda
S1=~/grscenes/home_scenes/scenes/MVUHLWYKTKJ5EAABAAAAAAI8_usd
```

Practical rules:
- **ROS_DOMAIN_ID ≤ 232.** Higher ids exceed FastDDS's port range and the runtime crashes. Use a different id for each runtime that runs in parallel.
- **Pin processes to P-cores** (`taskset -c 0-15` on the i7-14700KF). The runtime is about 2× slower on E-cores.
- **At most 3 Isaac instances at once** on the 12 GB RTX 4070 Ti. A fourth instance runs out of GPU memory.

### Training

Start the runtime, then `train.py`. The runtime starts paused, and the environment drives it.

```bash
taskset -c 0-15 ~/isaacsim/.venv/bin/python isaac/turtlebot3_isaac_sim.py --usd $USD --grscene $S1 --rtf 0 \
    --seed 0 --low-rays 0.02 --reset-every-episode --dyn-obstacles 3 --dyn-min 0 &

COMMON="--waypoints 0.75 --scene $S1 --plan-inflate 0.25 --plan-low-inflate 0.30 --stall-recovery --path-reward \
        --obstacle-penalty 20 --obstacle-penalty-dist 0.22 --slow-dist 0.35 --slow-min-frac 0.3"

# slow: from scratch (runtime without --dyn-obstacles)
taskset -c 0-15 .venv/bin/python -u sb3/train.py --output-dir <run> --total-timesteps 300000 --seed 0 \
    --learning-rate 3e-3 --learning-starts 128 --random-action-until 25000 $COMMON

# dyn: fine-tune slow with moving obstacles and stacked scans
taskset -c 0-15 .venv/bin/python -u sb3/train.py --output-dir <run> --total-timesteps 150000 --seed 0 \
    --learning-rate 3e-4 --learning-starts 10000 --random-action-until 0 $COMMON --scan-stack 3 --scan-stack-skip 2 \
    --init-from <slow run>/models/final_model.zip
```

With 40 rays, 300k steps take about 3.5–5 h, depending on the options.

### Evaluation

Each evaluation starts one runtime per apartment and drives it with `isaac/check_grscene.py`, using the deterministic policy:

```bash
ROS_DOMAIN_ID=224 taskset -c 0-9 ~/isaacsim/.venv/bin/python isaac/turtlebot3_isaac_sim.py --usd $USD --grscene <scene> \
    --rtf 0 --seed 0 --low-rays 0.02 --reset-every-episode --episode-log <out>/episodes.csv \
    [--dyn-obstacles 3 [--dyn-yield 0 --dyn-avoid 0.5]] &

ROS_DOMAIN_ID=224 .venv/bin/python isaac/check_grscene.py 100 --log <out>/episodes.csv --model <model.zip> \
    --scene <scene> --waypoints 0.75 --plan-inflate 0.25 --plan-low-inflate 0.30 --stall-recovery \
    --track-out <out>/track.csv
```

- **Summary line**: `SUMMARY episodes=100 success=… {'SUCCESS': …, 'COLL_WALL': …, 'COLL_OBST': …, 'TIMEOUT': …}`.
- **Trajectory CSV** (`--track-out`): one row per step with episode, step, x, y, tilt_deg, min_obstacle, goal_distance, lin, ang, outcome, dyn_x, dyn_y. The last two are the nearest moving obstacle. Collision maps and the responsibility analysis are computed from this file.
- **Frame stacking**: `check_grscene.py` stacks scans automatically when the model expects a longer observation.

## Results

Protocol: 5 apartments × 100 episodes, seed 0, deterministic policy, the same tasks for every model. Only one training seed was run, so differences of a few points may be noise.

### How we got here (static apartments)

| Step | Model and setting | S1 | Notes |
|---|---|---|---|
| Zero-shot | stage 4 model, end-to-end | 41% | stalls on rug edges and shoes below the lidar plane |
| Fine-tune in S1 | stage 4 model + TD3 | 17–48% | ✗ worse than zero-shot, with two different rewards |
| Simulation fixes | rugs drivable, low ray layer | 67% | the failures were perception, not the policy |
| Hierarchical | stage 4 model, waypoints 0.75 m, LOW cells, stall recovery | 85% | end-to-end on the same tasks: 54% |
| From scratch, weak penalty | −10 within 0.15 m | 62% | ✗ reckless |
| From scratch, r20 | −20 within 0.22 m + path progress | 96% | unseen apartments: S2 78, DY8 83, EI8 89, DI8 97 |
| + slow reward | target speed drops near obstacles | 98% | 5-apartment mean 89.6% vs 88.6%; collisions 36 vs 41 |
| + moving obstacles (dyn) | fine-tune, 3 stacked scans, 0–3 yielding obstacles | 97% | **main model** |

### Main model per apartment

| Condition | S1 | S2 | DY8 | EI8 | DI8 | Mean |
|---|---|---|---|---|---|---|
| Static | 97 | 78 | 83 | 93 | 99 | **90.0** |
| 3 yielding pedestrians | 93 | 75 | 83 | 89 | 92 | **86.4** |
| 3 steering pedestrians | 87 | 77 | 80 | 82 | 89 | **83.0** |

### Moving obstacles: dyn vs slow

| Pedestrian behaviour | dyn | slow | Obstacle hits (dyn / slow) |
|---|---|---|---|
| Yield (stop within 0.45 m) | 86.4% | 77.4% | 14 / 59 |
| Steer around the robot | 83.0% | 78.4% | 35 / 48 |
| Never yield | 63.4% | 62.8% | 138 / – |

- **Responsibility analysis** of obstacle hits. It uses the last 0.5 s of each trajectory and compares the closing speeds of robot and obstacle.
  - Robot-caused hits are rare for dyn: 1 of 35 with steering pedestrians, 7 of 138 with pedestrians that never yield.
  - In the remaining hits, the obstacle was faster than the robot, or the robot had already stopped.
- **Pedestrians that never yield are mostly unavoidable.**
  - They move at up to 0.5 m/s, while the robot is limited to 0.22 m/s and cannot reverse.
  - Corridors are about 0.8 m wide.
- Training with 50% non-yielding pedestrians did not help.

### Negative results

| Experiment | Result |
|---|---|
| Multi-apartment training from scratch (S1/DY8/EI8, 300k steps) | S2 63% vs 78% single-apartment; DY8 78% vs 83% |
| Multi-apartment fine-tune from dyn (S1/DY8/EI8, 150k steps) | mean 86.0% vs 86.4%; held-out S2 76 vs 75, DI8 91 vs 92. More cautious (obstacle hits 4 vs 14, timeouts 23 vs 14) |
| Body-height grid v5 for planning (`isaac/build_body_map.py`) | helps an existing model at test time (S2 78 → 87%); training on v5 is worse (S2 53%), and a bed-frame artifact makes 23% of S1 tasks unreachable |
| 360-ray lidar, sector minima to 40 inputs (`DRLNAV_SCAN_POOL=min`) | 82.2% vs 88.8% without pooling |

## Known issues

1. **Perception gaps.**
   - Thin legs fall between the 40 rays: in a collision analysis of 124 wall hits, 78% were chair and table legs, scraped from the side at full speed.
   - Nothing between 2 cm and 17 cm, or above 17 cm, is seen. This misses chair rails and table tops.
   - A depth camera is the planned fix.
2. **Pedestrian model.** A steering pedestrian that cannot turn in a narrow corridor keeps walking instead of stopping. Most of dyn's remaining obstacle hits come from this.
3. **Gaps to the real robot.**
   - The low ray layer does not exist on the real robot.
   - Lidar noise is Gaussian only.
   - Rugs have no colliders.
   - No per-wheel speed limit: full v with full ω asks one wheel for 0.38 m/s, above the real Burger's roughly 0.22 m/s. This needs checking against the motor specification.
4. **Statistics.**
   - All results come from one training seed. 100 episodes per apartment give roughly ±4–8 points of uncertainty.
   - Training and evaluation both use task seed 0, so the S1 evaluation tasks are likely the first 100 training tasks. Future evaluations should use an unused seed. The other four apartments are unaffected.
5. **Tooling.** The training, evaluation and analysis wrapper scripts still live outside the repository.
