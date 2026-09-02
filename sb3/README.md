# Stable-Baselines3 experiments

These scripts train and evaluate SB3 TD3 against the same
`DRLGymnasiumEnvironment` used by the native agent.

## Environment

On the current lab server, activate the existing environment after sourcing ROS
and the workspace:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
source /home/ubuntu/venvs/sb3_test/bin/activate
export DRLNAV_BASE_PATH=$PWD
export TURTLEBOT3_MODEL=burger
```

The validated environment contains SB3 2.7.1, Gymnasium 1.2.3, PyTorch
2.7.1+cu118, and NumPy 1.26.4.

Gazebo and `gazebo_goals` must already be running in the same `ROS_DOMAIN_ID`.

## Check the environment

```bash
python sb3/check_env.py --seed 0
```

## Train

This 5,000-step command matches the native comparison values for learning rate,
buffer size, warm-up, batch size, tau, gamma, and network width:

```bash
python sb3/train.py \
  --output-dir experiment_results/sb3_td3_seed0_5k \
  --total-timesteps 5000 \
  --seed 0
```

Each run stores its configuration, Monitor log, final model, VecNormalize state,
replay buffer, and recovery checkpoints under its output directory.
Periodic checkpoints omit the approximately 355 MB replay buffer by default;
add `--checkpoint-replay-buffer` when full intermediate recovery snapshots are
needed. The final replay buffer is always saved.

## Resume

```bash
python sb3/train.py \
  --output-dir experiment_results/sb3_td3_seed0_resumed \
  --total-timesteps 5000 \
  --seed 0 \
  --resume-model experiment_results/sb3_td3_seed0_5k/models/final_model.zip \
  --resume-vecnormalize experiment_results/sb3_td3_seed0_5k/models/vecnormalize.pkl \
  --resume-replay-buffer experiment_results/sb3_td3_seed0_5k/models/replay_buffer.pkl
```

If an old model has no replay buffer, omit `--resume-replay-buffer`; the script
then collects `--replay-warmup-steps` fresh transitions before updating again.

## Evaluate

```bash
python sb3/evaluate.py \
  --model experiment_results/sb3_td3_seed0_5k/models/final_model.zip \
  --vecnormalize experiment_results/sb3_td3_seed0_5k/models/vecnormalize.pkl \
  --output-dir experiment_results/sb3_td3_seed0_5k/evaluation \
  --episodes 20 \
  --seed 0
```

Evaluation is deterministic at the policy level and disables action noise. The
summary includes success, collision, stationary, turning, and in-place spinning
metrics. `turning_fraction` counts all high-angular-speed motion, while
`spin_fraction` is the subset with near-zero linear speed.

The seed covers Python, NumPy, PyTorch, CUDA, SB3, action spaces, and environment
motor noise. Gazebo physics and the separately running goal manager remain
nondeterministic unless they are also given explicit seeds.
