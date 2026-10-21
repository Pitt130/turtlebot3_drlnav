# Gazebo → Isaac Sim 迁移

更新：2026-10-21。分支 `isaac-migration` 基于 `gymnasium-interface` 的 `aca872e`。Gazebo 那套代码原样保留，Isaac Sim 是第二个仿真后端：Gymnasium 环境、奖励、碰撞判定、terminated/truncated 语义和 `sb3/train.py` 两边共用，只有仿真器不同。

| 位置 | 用途 |
|---|---|
| lab：`~/turtlebot3_isaac_migration_latest` | Ubuntu 22.04、ROS 2 Humble、Gazebo Classic；跑 Gazebo 侧测试 |
| 新服务器：`~/turtlebot3_drlnav` | Ubuntu 24.04、ROS 2 Jazzy、Isaac Sim 6.2（pip，`~/isaacsim/.venv`）、RTX 4070 Ti、i7-14700KF；训练用 `.venv`（torch 2.14 + cu130、SB3 2.7.1） |

## 结论

在 stage 4 上，Isaac 可以替代 Gazebo，训练速度约为 Gazebo 的 2.3 倍（见[结果](#结果)）。

## 代码结构

### 共用部分（`src/turtlebot3_drl`）

| 文件 | 行为 |
|---|---|
| `common/simulation_config.py` | 读取后端（`DRLNAV_SIM_BACKEND`）、stage、雷达点数、仿真速度；Isaac 模式不读 Gazebo SDF |
| `common/lidar_adapter.py` | 全圆扫描适配为 40 维策略输入：每个方向取最近的一条射线（与 Gazebo 单射线采样一致）；距离按 REP-117 和 `range_min`/`range_max` 处理 |
| `common/goal_poses.py` | 各 stage 的目标列表和"相邻目标 ≥ 2 m"的选取逻辑，`drl_gazebo` 和 Isaac 任务控制器共用；`DRLNAV_GOAL_SEED` 可固定序列。`tests/test_goal_poses.py` 逐步比对它与原 `generate_goal_pose` 的输出（12 个 stage × 5 个种子 × 200 个目标） |
| `common/arena_config.py`、`tools/export_arena.py`、`config/arena_inner_walls.json` | 共享地图 JSON，用于目标有效性判断 |
| `drl_environment/drl_environment_gymnasium.py` | stage 通过 `current_stage()` 读取（Gazebo 以 launch 写入的 `/tmp/drlnav_current_stage.txt` 为准，Isaac 用 `DRLNAV_STAGE`）；雷达走适配器；碰撞和奖励用全部有效射线的最小距离；`reset_rewinds_clock` 在 Isaac 下为 False（不等待 `/clock` 回退） |
| `drl_gazebo/drl_gazebo.py` | 删除、复位、生成每一步完成后才进行下一步，全部完成后才发布 `goal_pose` 并回复；多线程执行器，单个请求超时 10 s |

Gymnasium 语义：agent 只在 `terminated=True` 时停止 bootstrap；环境超时和步数上限都是 `truncated`。原 ROS agent（`drl_agent.py`）未改，仍把超时当作终止。

### Isaac 侧（`isaac/`）

| 文件 | 内容 |
|---|---|
| `build_turtlebot3.py` | 导入 TurtleBot3 Burger URDF 为 USD（选 `physx` 变体），轮子改为点接触球形碰撞体，并做开环驾驶检查 |
| `turtlebot3_lidar.py` | 用 Isaac 的物理射线传感器模拟 LDS：40 条射线、0.12–3.5 m、噪声 0.01 m、无回波 +inf |
| `stage_scene.py` | 从 Gazebo 的 world/model 文件生成 stage：静态墙为盒子；动态障碍物按 `obstacle_plugin/obstacleN.cc` 的关键帧动画（ignition Catmull-Rom 样条）驱动 |
| `turtlebot3_isaac_sim.py` | 运行时：物理 200 Hz，差速驱动（轮缘加速度 1 m/s²），`/clock` 10 Hz，`odom`/`scan`/`/obstacle/odom` 50 Hz，`/pause_physics`、`/unpause_physics`、`task_fail`、`task_succeed`；可选俯视录像和 WebRTC 推流 |
| `scene_recorder.py` | 俯视录像（叠加雷达射线），`make_video()` 合成 MP4 |
| `train_isaac.sh` | 启动运行时 + `sb3/train.py` 的训练脚本（`start`/`status`/`stop`） |
| `check_*.py`、`render_preview.py` | 验证脚本，见下 |
| `reference/gazebo_stage{4,5,9}_obstacles.csv` | Gazebo 录制的障碍物轨迹，用于对齐测试 |

`task_fail` 先把障碍物复位到起点并单独推进一个物理步，再把机器人放回起点。否则机器人可能被放进尚未移开的障碍物里被弹飞，记成假的 TUMBLE（早期训练中约 5% 的回合）。

## 使用方法

### 一次性准备（新服务器）

```bash
cd ~/turtlebot3_drlnav
source /opt/ros/jazzy/setup.bash
colcon build --packages-select turtlebot3_msgs turtlebot3_drl --symlink-install
# 机器人 USD：从 ROBOTIS turtlebot3_description（jazzy 分支）导入
OMNI_KIT_ACCEPT_EULA=YES ~/isaacsim/.venv/bin/python isaac/build_turtlebot3.py \
    --description ~/isaac_assets/turtlebot3/turtlebot3_description --out ~/isaac_assets/turtlebot3_burger_usd
```

### 启动运行时

```bash
source /opt/ros/jazzy/setup.bash; source install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=231 OMNI_KIT_ACCEPT_EULA=YES
~/isaacsim/.venv/bin/python isaac/turtlebot3_isaac_sim.py \
    --usd ~/isaac_assets/turtlebot3_burger_usd/turtlebot3_burger_isaac/turtlebot3_burger_isaac.usda \
    --stage 4 --rtf 0 --livestream --public-ip <本机IP>
```

- `--stage N`：目前支持 4、5、9（其他 stage 的 world 文件只要是盒子墙 + 圆柱障碍物也能加载，但未验证）。
- `--rtf`：实时因子上限，`0` 为不限速。环境的计时全部按 `/clock` 走，不限速不改变训练结果。
- `--livestream`：WebRTC 推流（TCP 49100、UDP 47998），用 NVIDIA 的 Isaac Sim WebRTC Streaming Client 连 `--public-ip`。有客户端连接时每秒渲染 `--view-hz`（20）帧，无人观看时 `--idle-view-hz`（1）帧。`--viewport-only` 只推视口不推界面。
- `--record DIR`：每 0.1 s 仿真时间保存一帧俯视图，`python isaac/scene_recorder.py DIR out.mp4 20` 合成视频。
- 启动后处于暂停状态，由环境通过 `/unpause_physics` 驱动，和 Gazebo `pause:=true` 一致。
- 随机目标（`ENABLE_TRUE_RANDOM_GOALS`）和动态目标（`ENABLE_DYNAMIC_GOALS`）尚未移植，开启时运行时拒绝启动。

### 训练

```bash
bash isaac/train_isaac.sh start 4 0 300000     # stage 4，种子 0，30 万步
bash isaac/train_isaac.sh status
bash isaac/train_isaac.sh stop                 # SIGINT，train.py 保存模型和 replay buffer
```

结果在 `~/drl_results_isaac/isaac_s<stage>_base_seed<seed>/`，布局和 lab 的 `~/drl_results` 相同（`config.json`、`logs/monitor.csv`、`logs/actions.csv`、`models/`）。手动启动时相当于：

```bash
DRLNAV_SIM_BACKEND=isaac DRLNAV_STAGE=4 .venv/bin/python sb3/train.py --output-dir <dir> --total-timesteps 300000 --seed 0
```

性能相关：

- 进程绑到 P 核（`taskset -c 0-15`）。Isaac 主循环落到 E 核时速度只有一半（0.9 倍实时 vs 1.8 倍）。
- 不限速时 stage 4 约 1.8 倍实时，30 万步 4.6 小时；Gazebo（lab，TITAN X）0.78 倍实时，同样步数约 10.7 小时。
- `train.py` 的 checkpoint 前缀写死为 `td3_stage4`，与实际 stage 无关。
- 用 SIGINT 停止时，rclpy 先关闭了 ROS 上下文，`train.py` 把退出当作异常，保存的文件带 `emergency_` 前缀，但内容完整。

### 评估与交叉评估

`sb3/evaluate.py` 两边通用，模型文件可以互换加载（SB3 2.7.1；lab 为 torch 2.7，新服务器为 torch 2.14）：

```bash
DRLNAV_SIM_BACKEND=isaac DRLNAV_STAGE=4 .venv/bin/python sb3/evaluate.py \
    --model <dir>/final_model.zip --vecnormalize <dir>/vecnormalize.pkl --output-dir <out> --episodes 100 --seed 0
```

## 验证

每一步都和 Gazebo 对比过（lab 上 Gazebo Classic，相同协议）：

| 检查 | 脚本 | 结果 |
|---|---|---|
| 开环驾驶：直行 3 s、原地转 3 s、弧线 4 s | `check_ros_interface.py --scene empty` | Isaac 0.573 m / 165.9° / 0.331 m–110.2°，Gazebo 0.574 m / 167.7° / 0.329 m–110.8° |
| 雷达 vs 解析距离 | `check_lidar.py` | 误差约 1e-5 m（不含噪声） |
| 时钟、话题频率、暂停/恢复、每步 0.1 s | `check_ros_interface.py` | /clock 10 Hz，odom/scan 50 Hz，步长恰好 0.1000 s |
| 障碍物轨迹 vs Gazebo 录制 | `check_obstacle_paths.py`、`check_stage.py --reference` | stage 4/5/9 最大偏差 < 0.05 mm |
| 扫描 vs 场景几何（墙 + 运动中的障碍物） | `check_stage.py` | 99.9% 以上的回波在 4 cm 内 |
| 回合流程：复位、目标、步长、done==terminated、task_succeed | `check_episodes.py` | stage 4/5/9 各 12 回合全部通过 |
| 离线/接口单元测试 | `tests/`、`test_drl_agent_integration.py`、`gymnasium_semantics_test.py` | lab 21/3/3/13/7/5 通过，新服务器 21/13/5 通过（Jazzy 没有 `gazebo_msgs`，Gazebo 同步测试不适用） |

## 结果

### stage 4 训练（SB3 TD3，`sb3/train.py` 默认参数，种子 0）

| | Isaac（新服务器） | Gazebo（lab，`aca872e`） |
|---|---|---|
| 训练量 | 30 万步 / 1587 回合 | 1000 回合 / 20.3 万步 |
| 耗时 | 4.6 h（1.80 倍实时） | 7.3 h（0.78 倍实时） |
| 前 1000 回合成功率 | 64.9% | 65.3% |
| 前 1000 回合结局（成功/撞墙/撞障碍/超时） | 649 / 164 / 161 / 26 | 653 / 169 / 147 / 31 |
| 最后 100 回合成功率 | 87% | 83% |

每 100 回合的成功率（0–1000）：Isaac 4/38/62/68/80/80/71/85/75/86%，Gazebo 10/26/59/75/74/79/84/78/85/83%。lab 上同设置的 Gazebo 基线 `td3_seed1`/`td3_seed2` 在 30 万步时为 83%/89%。

### 交叉评估（各 100 回合，seed 0）

| 评估环境 ＼ 模型 | Isaac 训出（30 万步） | Gazebo 训出（1000 回合） |
|---|---|---|
| Isaac | 84%（撞障碍 7，超时 9） | 85%（撞障碍 11，超时 4） |
| Gazebo | 93%（撞障碍 4，撞墙 1，超时 2） | 85%（撞障碍 12，超时 3） |

两个模型在对方的仿真器里都不比在自己的仿真器里差，说明雷达、运动、碰撞和奖励在两个仿真器里对策略而言是一致的。

## 未完成

- 只有种子 0 的对比；stage 5（6 个动态障碍物）的 60 万步训练进行中（`isaac_s5_base_seed0`），lab 基线为 `s5_base_seed0` + `c600_base_seed0`。
- 随机目标、动态目标模式；stage 4/5/9 以外的场景未验证。
- 早先审查中暂缓的两项：scan 回调的快速失败、旧流程的保护。
