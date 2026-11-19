# 当前接口与适配缺口

`MujocoCruzrRobot` 支持末端位姿读取、夹爪开口读取、开合夹爪、末端目标执行和双相机渲染。位姿为世界系夹爪基座 `[x,y,z,qw,qx,qy,qz]`，距离米、关节弧度；指尖与夹爪基座存在约 0.22m 偏移，应按真实硬件标定。

`MujocoCruzrSession.get_observation()` 当前仅含 `agentview`、`wrist`、`ee_pose`、`gripper_width`。这不是 DreamZero 官方观测协议。

新的 `CruzrEnvironment` 是后续策略接入的基础 API：`reset()` 返回命名关节位置、命名关节速度、夹爪开口和仿真时间；`step(joint_position, gripper_width_m)` 每次推进配置中的固定控制周期，按 SDK 关节名顺序接收 7 个右臂位置（弧度）和夹爪开口（米），并施加速度限幅。它目前是关节动作接口，不是 DreamZero 已确认的动作协议。

这里的速度限制针对目标指令变化率，不保证物理关节实际速度或碰撞安全；位姿误差、惯性和接触仍可能导致实际速度不同。控制周期指仿真时间，未实现网络异步调度或真实时间同步。新运行器只支持配置好的右臂，左臂SDK范围不能照抄。

控制器一次末端指令通过 IK、120 步插值与 60 步稳定推进；夹爪调用推进 180 个物理步。当前 MJCF 时间步为 0.002s，因此一次调用约推进 0.36s 仿真时间。它不是固定频率动作块接口，也不模拟网络推理延迟。模型训练数据频率必须与未来执行语义对齐。

DreamZero 集成前需要补充：按统一时间戳记录图像、关节、夹爪和任务；确定关节或末端动作、绝对或相对语义；实现固定频率动作块执行及新观测刷新；匹配模型的相机布局和本体配置。不得直接把 DROID 输出维度当作 Cruzr 控制指令。

## Cruzr 参考数据适配（2026-11-19）

来源及文件指纹见 `docs/interface-sources.json`。只读取用户提供的 Pi0.5 转换/训练工作空间，没有导入其源码，也未在其中找到 Cruzr 专用真机执行节点；不据此声称完成部署链路核验。DreamZero 保持为目标策略，以下接口只定义 Cruzr 的数值契约。

| 项目 | 参考资料中的事实 | 当前仿真实现 |
| --- | --- | --- |
| 状态 | `observation.state`，7 个右臂实测角度 + `GripStatus.pos` | `observe_pi05()`，7 个仿真角度 + 几何开口；SDK pos 到几何开口的标定未完成 |
| 动作 | `action`，7 个关节绝对目标 + `GripCmd.pos`，rad/m | `step_pi05()`；关节超界拒绝，夹爪 0–0.05 m 输入裁剪至几何上限 0.045 m，返回裁剪标记 |
| 动作块 | 15 Hz 数据、默认 50 帧目标 | `step_pi05_chunk()`；先校验整个 `(N,8)`，再执行，`1 <= N <= action_horizon` |
| 头部图像 | `observation.images.head_left`，1920×1536，头部左 RGB | 暂由外部 `front_cam` 代替，默认 640×480，未标定 |
| 腕部图像 | `observation.images.wrist_left`，640×480，物理右腕 | `R_wrist_cam`；保留历史特征名，机位仍是资产近似 |

8 维顺序固定为 `R_shoulder_pitch_joint`、`R_shoulder_roll_joint`、`R_shoulder_yaw_joint`、`R_elbow_roll_joint`、`R_elbow_yaw_joint`、`R_wrist_pitch_joint`、`R_wrist_roll_joint`、`right_gripper`。观测读取物理状态，不能用指令替代实测值。`step_pi05` 返回 `requested_action`、`applied_action`（经过目标变化率限制的伺服目标，不是实测状态）、`gripper_target_clipped`。

默认物理频率 500 Hz、诊断/关节接口 `step()` 50 Hz、参考策略 15 Hz 分开处理：`SampleClock` 按累计时间把每个策略样本量化为 33/34 个物理步，长期误差不累计，50 个样本推进约 3.334 s。`step_pi05()` 消耗一个策略样本，不能假定它等于一次 20 ms 的 `step()`。这两个入口共享物理步和变化率限制，不应在一个 episode 中混用来表达同一个策略时钟。推理等待期间仿真暂停；未实现真实时间异步队列或网络延迟补偿。

`render()` 保留原有 `front_cam`、`R_wrist_cam` 键；`observe_pi05(images=True)` 按配置返回两个参考特征键以及 `camera_calibrated=false`。两路 RGB 在同一仿真时刻渲染。参考数据的原始分辨率可通过 `camera.head_width/head_height` 设置，但相机标定、场景和策略训练分布仍需独立确认。

采集主题 `/mc/joint_states`、`/mc/right_arm/joint_command` 与 SDK 对外聚合主题不是同一路径，不能直接替换。模拟环境自身没有 ROS2 发布器；后续增加的可选 `ros2_transport.py` 与其分离，默认只读，尚未真机联调。SDK 模式、状态字段和页码见 `docs/sdk-reference.md`。

DreamZero 官方 DROID 评估脚本使用 IsaacLab/sim_evals，且使用不同本体、三路相机和独立的夹爪处理。当前 Cruzr MuJoCo 环境是本项目的适配基础，不能将 8 维形状相同视为策略语义兼容。Cruzr 数据转换、embodiment 配置、相机视角和微调权重仍需按导入的 DreamZero 新本体指南完成。

当前模型用简化碰撞几何，并非完整自碰撞保障；IK 使用关节范围，但没有完整的轨迹碰撞规划。原 Show-Harness 的 z-floor、插件和安全控制器未随本次提取。

2026-11-19 复查补充：新环境在每次动作推进结束后执行 `mj_forward`，使关节状态、物体位姿和相机使用同一时刻的数据；执行中任何异常均锁存故障，必须 reset。故障后的拒绝调用不会消耗新的策略时钟样本。IK 对非法位置/旋转/初值提前报错，并处理 180° 姿态误差的轴角奇点；这些控制修改仍需服务器物理回归。

官方快照 `wan_flow_matching_action_tf.py:parallelize()` 仅支持 `ip_size=1/2`；`sim_policy.py` 常规路径先把模型移动到每个进程的 GPU，并捕获并行异常。因此不能直接使用四进程作为 4×4090 权重分片方案。模型服务和 Cruzr 仿真闭环是后续独立验收阶段，完整操作流程见 [server-validation.md](server-validation.md)。

## 展示任务、真机与微调入口

默认入口 `task.run_pick_present_return(backend, config)` 完成抓起、脸前展示、停留、放回和释放。历史 `scan` 路点表示展示位以兼容已有计划；扫码流程只在显式 `run_pick_scan_return()`/`--mode scan` 时启用。

`task.run_pick_scan_return(backend, scanner, config)` 是可选扫码入口；backend 实现具名路点运动、夹爪、等待、拍摄、归位核验和停止。`scan_sim.SimScanBackend` 在仿真保存原始物体位置；`real_robot.HardwareScanBackend` 使用现场示教关节计划，避免仿真坐标与真机混用。扫码失败可在健康抓持下返回；抓持/状态/运动异常则中止。

真机用 `HardwareController` 独立发布线程约束目标变化率、状态时效、跟踪误差和命令 lease；`Ros2Transport(..., writable=False)` 不创建运动发布器。`health_topic` 和 `stop_request_topic` 是现场 supervisor 契约，不是厂商已实现的接口；未实测的停止逻辑不能当作急停。夹爪用 SDK 状态确认初始化/回零/故障/夹持，开口不能单独证明抓住。`execute_chunk()` 接收反归一化后的动作块并只执行短前缀；夹爪目标由任务控制。

`policy_runtime.PolicyMotionBackend` 提供 `reset()/predict(observation,instruction,phase)` 的新本体插件边界。当前仿真 CLI 支持该插件，真机 CLI 默认只执行已示教计划；新模型上真机还需独立模型闭环验收。扫码必须来自 decoder 的真实结果，不让模型文本代替码值。

`check_dataset.py`、`prepare_finetune.py`、`train_cruzr.py` 是数据核对、训练副本/本体配置生成和候选 LoRA 启动入口；官方快照不改，在输出目录派生并保存补丁指纹。数据与模型环境仍与仿真分离。完整参数、返回字段和阶段流程见 [部署 SOP](deployment-sop.md) 与 [微调 SOP](finetuning-sop.md)。
