# Cruzr DreamZero

面向 Cruzr S2「抓起 → 脸前展示 → 放回原位」的实验工程。当前版本包含从服务器已有 Show-Harness 仿真基础整理出的 MuJoCo 资产导入、按关节名的固定周期执行环境、七自由度 IK、夹爪、桌面方块抓放和相机接口。

**当前尚未完成 DreamZero 学习策略闭环及真机实测。** 已提供可选 SDK ROS2 适配、展示任务状态机、微调入口和阶段式部署流程；真机依赖现场标定、示教及独立监督停止接口。仿真抓放使用已知物体位置的确定性轨迹，不能代表学习策略成功率或真机性能。

当前任务的完整操作入口：

- [抓起、脸前展示、放回：仿真到真机 SOP](docs/deployment-sop.md)：任务运行、只读核对、示教、发布收据、现场执行及模型动作接口；扫码保留为后续可选扩展。
- [完整项目框架说明](docs/architecture.md)：模块、数据流、仿真/真机边界、训练入口和验证状态。
- [数据检查与 DreamZero 微调 SOP](docs/finetuning-sop.md)：指定服务器数据目录、LeRobot v2 检查、训练/验证拆分、Cruzr 本体配置生成、LoRA 计划与启动。

```bash
# 无模型、无真机，验证默认展示运动与返回。
.venv-sim/bin/python scripts/run_present_task.py --output-dir outputs/present-motion-01
# 只查看参数，不连接机器人或开始训练。
python3 scripts/robot_workflow.py --help
python3 scripts/train_cruzr.py --help
```

## 安装与运行

推荐 Linux + Python 3.11/3.12。依赖与 DreamZero 分开安装。

迁移当前修改到 4×4090 服务器时，按 [完整复现操作流程](docs/server-validation.md) 操作：包含代码与资产迁移、独立环境、无图像/双相机验收、报告解读、故障定位及 DreamZero 新本体后续步骤。未提交/推送的本地修改不能通过远端克隆获得。官方快照推理并行只接受 1/2 个进程，不能直接改成四进程来合并显存，具体源码限制见该流程第 9 节。

```bash
git clone https://github.com/Keith-814/cruzr-dreamzero.git
cd cruzr-dreamzero
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/import_assets.py --source-root /data1/oylx2
python scripts/check_mujoco.py
python -m unittest discover -s tests -v
python scripts/smoke_cruzr_pick_place.py --verbose
python scripts/smoke_pi05_contract.py  # 无模型推理的动作时序检查
```

如使用当前服务器已有环境，可跳过安装，设置包搜索路径：

```bash
export PYTHONPATH="$PWD/src"
/data1/oylx2/show-harness/.venv-sim/bin/python -m unittest discover -s tests -v
/data1/oylx2/show-harness/.venv-sim/bin/python scripts/smoke_cruzr_pick_place.py --verbose
/data1/oylx2/show-harness/.venv-sim/bin/python scripts/check_mujoco.py
```

渲染前视/腕部图片：

```bash
MUJOCO_GL=egl python scripts/smoke_cruzr_pick_place.py --render-dir outputs/smoke --verbose
```

无渲染测试不创建相机上下文；EGL 渲染需要可用 GPU/驱动。不要为此停止共享服务器上的其他任务。

## 目录

- `src/cruzr_sim/`：仿真会话、IK、初始姿态。
- `src/cruzr_sim/config.py`、`builder.py`、`environment.py`：SDK 参考边界、场景生成和固定周期命名关节 API。
- `scripts/`：本地模型导入与确定性抓放检查。
- `tests/`：已有控制器回归测试。
- `models/`：模型导入说明，本地模型不进入 Git。
- `docs/`：来源、接口约定和后续路线。
- `configs/mujoco.toml`：新运行器实际读取的物理步长、控制频率、关节限制、增益、工件和相机配置。`cruzr_sim.example.json` 仅为旧接口参考。

开发时保持 editable 安装；默认资产路径位于仓库内，当前不支持独立 wheel 携带模型发布。

## 后续步骤

1. 验证现有抓放与相机，记录初始基线。
2. 在服务器标定脸前展示位、静止工件、料框与任务评价。
3. 在已有动作块接口上增加训练用的同步图像记录、任务标注和数据导出。
4. 在独立环境运行官方 DreamZero 推理，再适配本体、数据和微调。
5. 仿真闭环通过后，进行真机 shadow 与单臂低速验证。

来源与许可边界见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。验证范围见 [docs/validation.md](docs/validation.md)。

## 服务器运行入口

在 `/data1/oylx2` 有权访问现有模型的服务器上：

```bash
git clone https://github.com/Keith-814/cruzr-dreamzero.git
cd cruzr-dreamzero
bash scripts/setup_sim.sh
.venv-sim/bin/python scripts/import_assets.py --source-root /data1/oylx2
.venv-sim/bin/python scripts/check_mujoco.py
MUJOCO_GL=egl .venv-sim/bin/python scripts/run_sim.py --mode hold --seconds 2
MUJOCO_GL=egl .venv-sim/bin/python scripts/run_sim.py --mode pick-place --render-dir outputs/smoke
```

`run_sim.py` 的 `pick-place` 仍是 oracle 位置控制，用于验证仿真控制链路，不是视觉策略，也不是 DreamZero。服务器上先用 `--mode hold` 检查物理模型；无图像入口禁用 GL，EGL 需通过后续渲染独立检查。

SDK 对照和仿真参数边界见 [`docs/sdk-reference.md`](docs/sdk-reference.md) 与 [`configs/mujoco.toml`](configs/mujoco.toml)。

## 策略接口与服务器验收

复现目标仍是 DreamZero。师兄提供的 Pi0.5 工程作为 **Cruzr 本体数据接口参考**：右臂 7 个关节角（弧度）加右夹爪位置（米），动作为绝对位置目标；`wrist_left` 是物理右腕图像的历史特征名。该目录包含转换与训练代码，本次未找到 Cruzr 专用的真机执行节点。

新增 `observe_pi05()`、`step_pi05()`、`step_pi05_chunk()`：动作按参考数据的 **15 Hz** 推进，50 帧约 3.33 s；底层 `step()` 仍为 50 Hz。整块动作执行前校验，记录请求目标、限速后的目标和夹爪裁剪。真实数据 0.05 m 的全开目标被裁剪为当前仿真几何上限 0.045 m，这是工程近似，尚未完成 SDK 位置与几何开口的真机标定。

默认两路仿真预览均为 640×480；参考数据原始图像为头部左相机 1920×1536、右腕 640×480。当前 `head_left` 使用外部前视相机代替，输出明确标为 `camera_calibrated=false`，不能据此评价 Pi0.5 或 DreamZero 的视觉迁移效果。按图像尺寸复制相机不能替代内外参标定。

迁移后使用同一个 Python 环境，先导入/复制本地机器人资产，再一键验收（输出目录必须不存在）：

```bash
SIM_PYTHON=python3.11 bash scripts/setup_sim.sh
.venv-sim/bin/python scripts/check_assets.py
.venv-sim/bin/python scripts/validate_sim.py --output-dir outputs/server-baseline --repeats 5
MUJOCO_EGL_DEVICE_ID=0 .venv-sim/bin/python scripts/validate_sim.py \
  --output-dir outputs/server-rgb --repeats 5 --render
```

结果包括资产/配置哈希、软件版本、GPU 信息、分阶段日志和报告；策略契约检查还保存带仿真时间戳的状态、请求动作和执行目标。重复固定场景用于回归检查，不是跨场景成功率。安装依赖、迁移资产、EGL 和 DreamZero 后续接入的分级验收见 [服务器验证说明](docs/server-validation.md)。按用户要求，本次不在本机运行物理或渲染实验；当前修改的物理验证状态见 [验证记录](docs/validation.md)。
