# Gymnasium / SB3 fixes and validation

## Changes

- Synchronize steps with simulation time and fresh odometry / scan samples; include the executed action in the next observation.
- Defer episode reset to reset(), retain terminal observations, and distinguish termination from truncation.
- Detect clock rewind, invalidate old sensor timestamps, and wait for post-reset data. Unexpected rewind during a step produces a diagnostic error.
- Restore drl_gazebo.py to upstream main f018b2cf49077d3e236ff6c3273525d055a2624f, including reset_simulation. No obstacle-plugin or robot-model changes are required by this patch.
- Restore OBSERVE_STEPS=25000 and support Tomas-compatible TD3 loss, architecture, initialization, optimizer, gradient clipping, OU discretization and separate random-action warmup through SB3.
- Avoid double reset during evaluation with DummyVecEnv.

## Running

Source ROS Humble, the built workspace and your SB3 virtual environment first. Set DRLNAV_BASE_PATH to the workspace root. Start Gazebo server, GUI and gazebo_goals with the same ROS_DOMAIN_ID and workspace. Include the workspace models directory in GAZEBO_MODEL_PATH and obstacle_plugin/lib directory in GAZEBO_PLUGIN_PATH. Verify moving obstacles and GUI before training.

Run `bash sb3/run_td3_fixed.sh` (default 200000 steps). This script trains against the existing simulation; it does not launch Gazebo. The SB3 path does not invoke drl_agent_gymnasium.py.

Validated configuration: learning rate 0.003, buffer 1000000, learning_starts 128, random actions until 25000, batch 128, tau 0.003, gamma 0.99, OU sigma 0.1 / dt 1, simulation interval 0.1 seconds, raw rewards, smooth_l1 loss, gradient clipping 2, AdamW and Tomas architecture / initialization. Parameter alignment does not imply an identical original training loop.

## Validation

200000 training steps completed with the original Gazebo goal manager. Seed-0 evaluation: 100 episodes, 86 successes, 7 obstacle collisions, 5 wall collisions, 2 timeouts. Spin-command fraction weighted by steps: 169 / 16470 = 1.03%; unweighted mean episode fraction: 2.12%. These are commanded-action metrics, not measured trajectory rotation. Spin means absolute commanded linear speed below 0.02 m/s and absolute angular speed above 0.5 rad/s.

Some collision episodes still contain spinning. Timeout episodes mainly contain low commanded forward speed. Additional evaluation seeds 1–4 were running at submission time. Per-episode summaries cannot recover the longest consecutive spinning interval. No claim of universal resolution is made.

The upload was verified with 15 regression checks for clock reset, environment semantics and SB3 replay / noise behavior. The extra diagnostic test files are kept locally. The existing repository semantic tests remain available, with ROS and Python dependencies sourced:

```bash
python gymnasium_semantics_test.py
```

Models, replay buffers and large logs are excluded from this commit.
