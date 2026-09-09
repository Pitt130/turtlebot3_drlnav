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

## Reliability follow-up, 2026-09-09

The same final model completed evaluation seeds 0–4 (100 episodes each): success rates 86%, 80%, 88%, 79%, 81%. Total: 414 successes / 500 episodes (82.8%), 47 obstacle collisions, 23 wall collisions, 16 timeouts. Weighted spin-command counts were 169/16470, 443/17033, 222/15020, 271/17196 and 464/17008. These runs precede the follow-up changes. Evaluation seeds do not imply independent training runs or fully controlled Gazebo goal/obstacle randomness.

- A reset retry continues waiting for the original request and epoch instead of resending a reset. Difficulty updates occur once after successful reset and pause. An ambiguous failed service does not silently start a new transaction; recovery may require recreating the environment. Automatic training retry is intentionally not enabled.
- Pause failures preserve an existing exception. Redundant Gymnasium action-history storage and callback-based deadline state were removed; step control and observation semantics are unchanged.
- The shared obstacle odometry queue holds at least 10 messages. Samples are cached by obstacle ID, older updates rejected, and distances recomputed against the current robot pose. Samples older than max(0.2 seconds, two configured steps) are excluded from dynamic-collision classification. Missing samples still leave laser-detected collisions classified as walls; this change may alter collision subtype counts and needs further evaluation.
- The CLI defaults now match the validated learning settings, including 128 learning-starts, 25000 random-action steps, smooth_l1, AdamW, Tomas initialization/critic, OU dt=1 and no reward normalization. Total-timesteps retains the small 5000 CLI default; the supplied training script defaults to 200000.
- Training exceptions attempt emergency model, normalization and replay saves independently, then attempt all cleanup operations without masking the original error. This handles Python exceptions during learning, not SIGKILL, power loss, or failures before model initialization. Emergency files have the emergency_ prefix and are not completed training results.
- Original reset_simulation, epoch checking, 0.1-second control interval, OU episode reset behavior and tumble threshold are retained. Service discovery waits were not optimized without latency measurements; each step still pauses/unpauses.

Validation: semantic and clock regression tests, targeted exception/default/cache checks, real Gazebo injected reset-timeout retry and 10 reset/step cycles, plus a 160-step SB3 training smoke run. Extra diagnostic tests remain local. These checks establish interface reliability, not improved policy success rate; the trained model is unchanged.
