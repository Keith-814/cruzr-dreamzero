# Cruzr S2 抓起—脸前展示—放回 SOP

本 SOP 的默认任务是拿起一个物体，移动到机器人脸前的安全展示位，停留并采集一次观测，然后回到原位、松手和退离。当前不执行扫码；扫码是后续可选的任务层扩展。所有真机命令都必须在现场完成安全评审后执行。

## 1. 服务器仿真准备

先按 [完整服务器复现流程](server-validation.md) 迁移本地工作区。本轮修改尚未提交/推送，直接 clone 远端不会取得这些修改；私有模型资产也不在 Git 中。迁移完成后在 4×4090 服务器建立独立仿真环境，不与 DreamZero 模型环境混装：

```bash
cd /data/cruzr-dreamzero
bash scripts/setup_sim.sh
.venv-sim/bin/python scripts/import_assets.py --source-root /data1/oylx2
.venv-sim/bin/python scripts/check_assets.py
```

`models/cruzr_s2/scene.xml` 及其 STL/纹理必须通过相对路径和哈希检查。缺少资产时停止，不用空场景或猜测 body/camera 名称绕过。

先做无图像验收：

```bash
.venv-sim/bin/python scripts/validate_sim.py \
  --output-dir outputs/server-present-01 --repeats 5
```

需要 RGB 时在有 EGL 的服务器执行：

```bash
MUJOCO_GL=egl .venv-sim/bin/python scripts/validate_sim.py \
  --output-dir outputs/server-present-rgb-01 --repeats 5 --render
```

独立查看默认任务：

```bash
.venv-sim/bin/python scripts/run_present_task.py \
  --output-dir outputs/present-visual-01 --render
```

输出 `task-report.json` 的 `phases` 应包含 `approach, grasp, lift, present, return, release, retreat, complete`。`success=true` 还要求仿真后端确认物体回到原位。该脚本是已知方块位置的 oracle 轨迹，不是视觉策略成功率。

## 2. 展示位调整

默认配置位于 `configs/mujoco.toml`：

```toml
[task.presentation]
position = [0.05, 1.02, 1.22]
yaw = 0.0
dwell_s = 1.0
```

这是服务器上的起始候选世界坐标，代码启动时会先进行 IK 可达性预检。查看 RGB 画面后，以头部相机真实视场、物体与脸部的安全间距、腕部遮挡和关节限位为依据调整。将每次配置、截图和报告放在独立输出目录。不要把这个点直接当成真机坐标；真机使用现场示教的 `scan`/`present` 关节路点。

`--scan-offset dx dy dz` 仅用于仿真展示位微调，单位为世界坐标米。它不能替代相机内外参或碰撞规划。

## 3. 运行阶段与失败处理

任务层顺序固定为：预抓 → 抓取确认 → 抬起 → 脸前展示 → 停留/采集 → 回到原位上方 → 下降 → 松手 → 退离 → 归位核验。运动、状态、IK、抓持或物理异常会锁存并停止后续动作，不会在未知状态下自动回放、返回或松手。没有归位 verifier 时会报告运动完成但任务失败。

可选扫码只在明确需要时启用：

```bash
.venv-sim/bin/python scripts/run_scan_task.py --mode scan \
  --scanner fixture --output-dir outputs/optional-scan-motion-01
```

`fixture` 只测试顺序，不能证明相机解码；真实 `opencv-qr`/`zbar` 需要 RGB 画面和现场依赖。扫码失败时健康抓持仍可返回，但不会伪造扫码成功。

## 4. 真机前的只读流程

在机器人端使用厂商 SDK 已支持的 ROS2/Python 环境，先 source 现场 ROS2 与 SDK 的 `setup.bash`，确认能导入 `rclpy`、`mc_state_msgs`、`mc_task_msgs`、`ecat_task_msgs`、`sensor_msgs`、`std_msgs` 和 NumPy。路径由现场安装位置决定，不复制服务器 venv，也不把 MuJoCo/模型依赖装进 SDK 环境。脚本的真机模块不要求导入 MuJoCo。

先做不会创建 ROS 节点或发布器的消息结构预检：

```bash
python3 scripts/preflight.py --target robot \
  --profile configs/hardware.local.json --plan configs/present-plan.local.json \
  --output outputs/preflight-robot.json
```

预检通过只说明消息字段、profile 和 plan 可读取；`probe` 仍必须现场核对真实反馈、时钟、相机和 supervisor。

复制 `configs/hardware.example.json` 为本地 profile，填写现场审核后的主题、限位和 calibration id。模板默认所有 commissioning 标记为 false，不能直接执行。`/mc/sdk/robot_state`、`/mc/sdk/robot_command` 与右夹爪主题来自 SDK 参考；两路图像、`health_topic` 和 `stop_request_topic` 必须现场配置。后两者是独立 supervisor 的集成约定，不是声称厂家已有的主题。

```bash
cp configs/hardware.example.json configs/hardware.local.json
```

当前 adapter 接收带源时间戳的关节/夹爪状态和原始 `rgb8`/`bgr8` 图像。相机、机器人和 ROS 时钟要同步；收到数据不等于健康，`init_state=1`、`homed=1`、`error_code=0` 和抓持反馈也必须核对。夹爪的 SDK 行程与几何开口映射需要实际标定，不能直接将模板数值标为已验证。

```bash
python3 scripts/robot_workflow.py --mode probe \
  --profile configs/hardware.local.json --output-dir outputs/probe-01
```

probe 只读取一份状态，需人工核对时间戳、七个关节名/顺序、夹爪状态、健康 supervisor、相机和 SDK 消息类型。随后由获授权人员使用厂商批准方式外部示教，不由本脚本上电、回零或抢占控制权：

```bash
python3 scripts/robot_workflow.py --mode teach \
  --profile configs/hardware.local.json --plan configs/present-plan.local.json \
  --waypoint present --output-dir outputs/teach-present-01
```

先记录静止的 `start`，再重复记录 `pregrasp, grasp, lift, present, return_high, return_low, retreat`；程序把 `present` 保存为兼容旧计划的 `scan` 键。执行前必须回到并稳定在 `start`，不能从任意当前姿态直接进入 approach。逐段检查自碰、桌面、线缆和夹爪间隙。线性插值不是碰撞规划。

再执行只读 shadow：

```bash
python3 scripts/robot_workflow.py --mode shadow \
  --profile configs/hardware.local.json --plan configs/present-plan.local.json \
  --output-dir outputs/shadow-01
```

## 5. 发布收据与可写闸门

服务器仿真报告必须是同一版本、同一配置的成功 RGB 验收，并包含展示任务报告哈希。profile 必须通过关节零位、夹爪映射、相机、SDK 路由、稀疏发送、命令超时、独立停止和 supervisor 验收；默认展示任务不要求 `scanner_verified`。跨机器复制 outputs 后，收据使用相对证据路径并重新执行校验；新增/删除源码会被 code inventory 拒绝。

```bash
python3 scripts/prepare_deployment.py \
  --sim-report outputs/server-present-rgb-01/report.json \
  --task-report outputs/server-present-rgb-01/present-task/task-report.json \
  --profile configs/hardware.local.json --plan configs/present-plan.local.json \
  --output outputs/present-release-01.json
```

收据绑定 profile、plan、仿真报告和任务报告 SHA-256。任何代码、配置、资产、示教路点变化都必须重新验收和生成收据；跨机器时保留证据或在目标目录重生成，不复制本机虚拟环境。

## 6. 真机低速执行

先空载/单阶段，再单物体低速，现场人员保持急停可用：

```bash
python3 scripts/robot_workflow.py --mode execute --execute \
  --task-mode present \
  --profile configs/hardware.local.json --plan configs/present-plan.local.json \
  --release outputs/present-release-01.json \
  --output-dir outputs/robot-present-01
```

上面的命令会尝试动作，但没有归位验证器时最终 `success=false`。现场有可信检测后增加 `--placement-verifier site_quality:verify_return`，函数签名为 `verify_return(transport, plan) -> bool`，必须核验本次物体在原支撑位置且稳定。它不能无条件返回 True。

`--execute` 是唯一开启发布器的显式开关。控制器会检查反馈新鲜度、跟踪误差、夹持状态、目标限位、命令 lease 和发布线程异常；失败后锁存并发出现场 stop 请求。Python stop topic 不是厂商急停的替代品。没有独立 supervisor、可靠状态反馈或归位 verifier 时停留在 probe/teach/shadow。

## 7. 策略接入边界

模型插件必须实现：

```python
class CruzrPolicy:
    def reset(self): ...
    def predict(self, observation, *, instruction, phase):
        # 返回 (N, 8)，绝对关节 rad + 夹爪 m，已完成反归一化
        ...
```

任务指令为“拿起物体，在机器人脸前停留检查，然后放回原位”。模型输出先整体检查有限值、维度、关节限位和单位，再只执行最多 3 帧，刷新状态后重规划；运输阶段夹爪由任务层控制。真机 CLI 当前只执行示教计划，模型上机还需独立仿真闭环和发布绑定。

## 8. 现场停止与记录

异常时保持现场急停能力，确认物体和机械状态后新建会话；不要自动重试整段任务。保存 `robot-report.json`、状态/图像时间戳、停止原因、校准编号和示教计划版本。归位成功必须由治具/视觉/人工核验器给出证据，不能用“夹爪已打开”代替。

当前仓库未在本机运行 MuJoCo、ROS 或真机；服务器和现场输出才是对应环境的验证证据。
