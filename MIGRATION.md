# Gazebo → Isaac Sim 迁移（基于 GitHub 最新代码）

日期：2026-10-19。基线为 GitHub `Pitt130/turtlebot3_drlnav` 的 `gymnasium-interface` 分支 `5bc0933`，改动在本地分支 `isaac-migration` 上，尚未提交或推送。

| 位置 | 用途 |
|---|---|
| lab：`~/turtlebot3_isaac_migration_latest` | ROS 2 Humble + Gazebo，跑完整测试 |
| 新服务器：`~/turtlebot3_drlnav` | Ubuntu 24.04、ROS 2 Jazzy、RTX 4070 Ti，跑 Isaac；Python 依赖在 `.venv` |

10 月 14 日基于 `0fe4e8d` 的旧迁移副本（lab：`~/turtlebot3_isaac_migration_20261014`）只作参考，不再更新。

## 从旧迁移带过来的部分

| 文件 | 行为 |
|---|---|
| `common/simulation_config.py` | 统一读取后端、stage、雷达点数、仿真速度；Isaac 模式不读 Gazebo SDF |
| `common/lidar_adapter.py` | 全圆扫描适配为 40 维策略输入：每个原始方向取最近的一条射线（与 Gazebo 单射线采样一致）；距离按 REP-117 和 `range_min`/`range_max` 处理 |
| `common/arena_config.py`、`tools/export_arena.py`、`config/arena_inner_walls.json` | 共享地图 JSON，可覆盖 arena 长宽并用于目标有效性判断 |
| `common/settings.py`、`common/utilities.py`、`setup.py` | 接入上述配置；安装地图 JSON |

## 针对上游代码的调整

- **stage：** 环境、目标节点和 `utilities` 都通过 `current_stage()` 读取。Gazebo 以 launch 写入的 `/tmp/drlnav_current_stage.txt` 为准，`DRLNAV_STAGE` 与之不一致时报错；Isaac 优先使用 `DRLNAV_STAGE`。stage 未知时报错，不默认为 1，与上游 agent 的要求一致。
- **雷达：** 上游 `scan_callback` 仍直接取前 40 条射线，现改用适配器；保留上游的时钟回退保护和 `scan_stamp`。碰撞判断和奖励使用全部有效射线的最小距离。
- **reset 与时钟回退：** 上游在 `task_fail` 后等待 `reset_simulation` 造成的 `/clock` 回退。新增 `reset_rewinds_clock`：Gazebo 下保持原行为，Isaac 下不等待回退，以任务服务回复后的新观测为准。
- **`drl_gazebo`（基于上游的 `reset_simulation`）：** 删除、复位、生成每一步完成后才进行下一步，全部完成后才发布 `goal_pose` 并回复；改用多线程执行器，单个请求超时 10 s。
- **Gymnasium 标准语义：** 自定义 agent 只在 `terminated=True` 时停止 bootstrap；环境超时和 agent 自己的步数上限都属于 `truncated`，价值目标继续使用下一状态，与 Gymnasium 和 SB3 的约定一致。上游 `e009de5` 曾把环境超时按终止存储以对齐原 ROS agent，现已改回；原 ROS agent（`drl_agent.py`）未改，仍把超时当作终止，两者在这一点上不同。
- **numpy：** `logger.py`、`drl_environment.py` 中的 `numpy.core.numeric.Infinity` 改为 `math.inf`，避免 numpy 2 下导入失败。

上游已实现的 step/reset 同步（每步推进 0.1 s 仿真时间、等待新的 `/clock`、`/odom`、`/scan`，reset 失败抛错）原样保留。

## Isaac 侧需要提供的接口

环境变量：`DRLNAV_SIM_BACKEND=isaac`、`DRLNAV_STAGE=<n>`，可选 `DRLNAV_ARENA_CONFIG=<地图 JSON 绝对路径>`。观测维度在导入时确定，需在启动前设置。

| 接口 | 要求 |
|---|---|
| `/clock` | 仿真时间，env 以此推进 step 和计算超时 |
| `/odom`、`/scan`、`/cmd_vel` | `/scan` 每帧为完整 360° 扫描，前方与机器人 +X 对齐；`range_min` ≤ 0.12 m（碰撞阈值 0.13 m），`range_max` ≥ 3.5 m |
| `/pause_physics`、`/unpause_physics`（`std_srvs/Empty`） | env 每步先恢复、推进 0.1 s 后再暂停物理 |
| `task_succeed`、`task_fail`（`turtlebot3_msgs/RingGoal`） | 复位和生成目标都完成后才回复；`goal_pose` 在目标存在后才发布；不要求回退时钟 |

## 验证结果

| 检查 | lab（Humble） | 新服务器（Jazzy） |
|---|---|---|
| 迁移离线测试 | 21/21 | 21/21 |
| 上游 agent 集成测试（`DRL_AGENT_PATH` 指向 agent 源文件） | 13/13 | 13/13 |
| ROS 接口测试（Isaac 模式） | 5/5 | 5/5 |
| Gazebo 目标节点同步测试（假 Gazebo 服务） | 3/3 | 不适用：Jazzy 没有 `gazebo_msgs` |
| 上游语义测试 `gymnasium_semantics_test.py` | 7/7 | 不适用 |
| torch CUDA（RTX 4070 Ti） | — | 可用 |

改回标准语义前，语义测试中的 `test_truncated_transition_keeps_bootstrap` 在上游 `5bc0933` 上失败（`e009de5` 改了 agent 但没有更新这个测试）；agent 集成测试中对应的用例已改为断言标准语义。

尚未执行：真实 Gazebo 连续回合测试；Isaac Sim 的安装、机器人与场景搭建、任务控制器。

## 复现测试

lab（Humble）：

```bash
cd ~/turtlebot3_isaac_migration_latest
source /opt/ros/humble/setup.bash
source ~/turtlebot3_gym-interface_gitlab/install/turtlebot3_msgs/share/turtlebot3_msgs/local_setup.bash
export PYTHONPATH="$PWD/src/turtlebot3_drl:$PWD:${PYTHONPATH:-}" ROS_DOMAIN_ID=231 MPLBACKEND=Agg
export DRLNAV_BASE_PATH="$PWD"
python3 -m unittest discover -s tests -p 'test_simulator_migration.py' -v
python3 -m unittest discover -s tests -p 'test_gazebo_goal_sync.py' -v
DRL_AGENT_PATH="$PWD/src/turtlebot3_drl/turtlebot3_drl/drl_agent/drl_agent_gymnasium.py" python3 -m unittest test_drl_agent_integration
python3 gymnasium_semantics_test.py
unset DRLNAV_BASE_PATH
DRLNAV_SIM_BACKEND=isaac DRLNAV_STAGE=1 DRLNAV_ARENA_CONFIG="$PWD/config/arena_inner_walls.json" \
  python3 -m unittest discover -s tests -p 'test_migration_ros_contract.py' -v
```

新服务器（Jazzy）：

```bash
cd ~/turtlebot3_drlnav
source /opt/ros/jazzy/setup.bash
colcon build --packages-select turtlebot3_msgs turtlebot3_drl --symlink-install
source install/setup.bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}" ROS_DOMAIN_ID=231 MPLBACKEND=Agg DRLNAV_BASE_PATH="$PWD"
.venv/bin/python -m unittest discover -s tests -p 'test_simulator_migration.py' -v
DRL_AGENT_PATH="$PWD/src/turtlebot3_drl/turtlebot3_drl/drl_agent/drl_agent_gymnasium.py" .venv/bin/python -m unittest test_drl_agent_integration
unset DRLNAV_BASE_PATH
DRLNAV_SIM_BACKEND=isaac DRLNAV_STAGE=1 DRLNAV_ARENA_CONFIG="$PWD/config/arena_inner_walls.json" \
  .venv/bin/python -m unittest discover -s tests -p 'test_migration_ros_contract.py' -v
```

`colcon` 使用系统 Python 构建；训练和测试使用 `.venv/bin/python`，它能看到系统的 `rclpy`。

## 下一步

1. 安装 Isaac Sim（确认支持 Ubuntu 24.04、Jazzy 和当前驱动的版本），运行兼容性检查。
2. 导入 TurtleBot3 Burger，配置差速控制、2D 雷达和 ROS 2 Bridge（`/clock`、`/odom`、`/scan`、`/cmd_vel`）。
3. 按 `inner_walls`/`outer_walls` 的 SDF 搭建场景；外墙内侧在 ±2.35 m，地图 JSON 的 4.2 m 只是目标采样边界。
4. 实现 Isaac 任务控制器，提供上表的暂停/恢复和 `task_succeed`/`task_fail` 服务，并移植 `drl_gazebo.py` 中各 stage 的目标列表。
5. 在 Isaac 中跑连续回合，再开始训练。
