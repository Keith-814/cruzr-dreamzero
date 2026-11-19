# Cruzr S2 / DreamZero 复现操作流程

适用版本：本仓库 2026-11-19 的右臂桌面 MuJoCo 实现；目标机器为 Linux x86_64、4×RTX 4090。本文按「迁移代码和资产 → 建立仿真环境 → 物理验收 → 双相机验收 → 官方模型诊断 → Cruzr 策略适配」组织。前四步已有可执行入口，后两步列明现有源码限制及尚需开发的内容。

**当前尚未完成 DreamZero 驱动 Cruzr 的学习策略闭环。** 固定位置抓放使用 oracle 物体坐标；参考动作检查不加载 Pi0.5 或 DreamZero。2026-11-15 的成功记录属于旧版本历史参考。本次只做本机轻量检查，未安装或运行本机 MuJoCo，未验证目标服务器依赖、EGL、物理抓放或模型显存。

## 1. 准备条件和验收层级

| 层级 | 通过条件 | 可以得出的结论 |
| --- | --- | --- |
| A：资产 | 场景与所有网格清单/哈希一致 | 迁移包完整；不代表物理参数真实 |
| B：仿真物理 | 依赖检查、零跳过回归、10 秒保持、5 次抓放、策略时序均通过 | 当前固定场景的控制基础可用 |
| C：相机 | B 通过，RGB 尺寸和空间变化检查通过，人工复核视野 | 两路仿真图像可生成；不代表真机相机已标定 |
| D：模型服务 | 官方输入获得有限有效动作，记录每卡显存与延迟 | 对应模型/硬件上的服务可用；不代表 Cruzr 可用 |
| E：Cruzr 闭环 | 新本体数据/配置/权重与仿真适配完成，独立任务验收 | 才能讨论 DreamZero 在 Cruzr 仿真上的任务表现 |

必须准备有权使用的 Cruzr MJCF 和对应网格。本仓库不含机器人资产，SDK PDF 和 Pi0.5 工作空间也不能代替模型。优先复用原服务器的 `show-harness/sim/cruzr_s2_urdf.xml` 和 `cruzr_urdf/cruzr_s2_description_gripper/meshes_mjcf/`，或完整的已导入 `models/cruzr_s2/`。没有这些资产时，流程停在模型导入前，不能凭 SDK 尺寸生成替代模型后宣称复现。

仿真推荐 Python 3.11，也支持 3.12。MuJoCo 物理循环主要使用 CPU，EGL 图片使用一张 GPU；四卡不会自动加速本实现的物理循环。CPU、内存、磁盘容量和 GPU 空闲情况由实际场景/模型确定，本文不把未经测量的资源需求写成最低配置。

## 2. 迁移当前修改到服务器

**本机当前改动尚未提交或推送，单独从远端 `git clone` 不会包含它们。** 可以在提交并推送后克隆指定版本，或按下面方式复制当前整个工作目录，包括新增脚本和文档。不要复制 `.venv`，应在目标机器重建环境。

以下只是一份供你执行的迁移命令，本次没有连接任何服务器。将 `USER@SERVER` 和目标目录改为自己的值。目标选择一个新的空目录；不使用 `--delete`。

在本机终端执行：

```bash
SIM_SERVER=USER@SERVER
REMOTE_REPO=/data/cruzr-dreamzero
ssh "$SIM_SERVER" "mkdir -p '$REMOTE_REPO'"
rsync -a --info=progress2 \
  --exclude='.venv*/' --exclude='__pycache__/' --exclude='*.egg-info/' \
  --exclude='.env' --exclude='.env.*' --exclude='.imports/' \
  --exclude='outputs/' --exclude='logs/' --exclude='data/' --exclude='checkpoints/' \
  --exclude='models/cruzr_s2/' \
  /home/user/dreamzero/cruzr-dreamzero/ "$SIM_SERVER:$REMOTE_REPO/"
```

这里保留 Git 元数据，以便验收报告记录版本和未提交状态；机器人资产另行迁移。`third_party/dreamzero/debug_image/` 是官方测试输入，应保留。SDK、师兄工作空间和其中的访问凭据不需要复制进本仓库。

后续服务器命令均从仓库根目录执行：

```bash
cd /data/cruzr-dreamzero
pwd
git status --short
python3.11 --version
nvidia-smi
nvidia-smi topo -m
```

迁移后先运行只读预检，预检报告必须保存，失败时不要进入下一层：

```bash
python3.11 scripts/preflight.py --target sim \
  --config configs/mujoco.toml --output outputs/preflight-sim.json
python3.11 scripts/preflight.py --target model --gpus 4 \
  --output outputs/preflight-model.json
```

`preflight.py` 只检查 Python/包版本、资产配置和可见 CUDA 设备，不加载权重、不分配模型张量、不启动 MuJoCo 物理或训练。模型预检通过也不代表 14B 基座能放入 4×4090；显存和并行方案仍按后续短运行验收。

`nvidia-smi` 用于核对四张卡的型号、驱动、占用和互连，不修改任何进程。若没有 Python 3.11/3.12，先通过服务器已有的 Python/Conda 管理方式准备；不需要更改系统 Python 或共享驱动。

## 3. 建立独立仿真环境

```bash
SIM_PYTHON=python3.11 bash scripts/setup_sim.sh
.venv-sim/bin/python --version
.venv-sim/bin/python -m pip check
```

直接依赖固定为 `mujoco==4.0.1`、`numpy==2.2.6`、`opencv-python-headless==4.11.0.86`；完整依赖版本会写入每次验收的 `environment-freeze.txt`。这是待目标服务器干净安装验证的组合。若已有 `.venv-sim`，脚本会使用它；需要全新环境时先把旧目录改名保留，再运行脚本。

不想依赖系统 Python 路径时，可在一个独立 Conda 环境中创建本项目 venv：

```bash
conda create -n cruzr-sim-base python=3.11 -y
conda activate cruzr-sim-base
SIM_PYTHON="$(command -v python)" bash scripts/setup_sim.sh
```

之后统一使用 `.venv-sim/bin/python`。不要把仿真包和 DreamZero 装进同一个环境：官方快照固定 NumPy 1.26.4，而本仿真环境固定 NumPy 2.2.6。

## 4. 导入或复制机器人资产

三种方式任选一种，不要连续对同一个目标目录重复导入。

### 4.1 原服务器布局仍在

若目标机器确实存在相同目录：

```bash
.venv-sim/bin/python scripts/import_assets.py --source-root /data1/oylx2
```

`--source-root` 指包含 `show-harness/` 和 `cruzr_urdf/` 的父目录。其他机器没有该路径时，应使用下面两种方式之一。

### 4.2 已准备独立场景与网格

```bash
.venv-sim/bin/python scripts/import_assets.py \
  --scene /path/to/cruzr_s2_urdf.xml \
  --mesh-dir /path/to/meshes_mjcf
```

把示例路径换成已迁移的实际路径。导入只接受已展开 include 的独立 MJCF，复制文件型网格，保留场景内联定义；未打包的外部纹理、高度场、skin、子模型会明确报错，不会生成一个表面完整的导入包。输入场景必须是兼容的 Cruzr 仿真资产，任意厂商 MJCF 不保证具备本项目所需关节、执行器和相机。

### 4.3 私下迁移已导入资产包

把旧服务器的整个 `models/cruzr_s2/` 复制到本仓库同名目录。至少包含：

```text
models/cruzr_s2/
├── scene.xml
├── asset-manifest.json
└── meshes_mjcf/
    └── 所有被场景引用的网格
```

资产被 Git 忽略，不上传到公开仓库。不要只复制 XML，也不要只复制 STL 而丢掉 manifest。

三种方式完成后统一检查：

```bash
.venv-sim/bin/python scripts/check_assets.py
.venv-sim/bin/python scripts/check_mujoco.py
```

第一条必须输出 `success: true`；网格数量以实际清单为准，历史资产为 46 个。第二条必须打印模型维数、7 个命名关节、控制频率、初始姿态和夹爪开口。默认应为 50 Hz / 每周期 10 个物理步；模型维数受资产版本影响，不能只靠数字相同判断模型正确。

导入在临时目录完成复制、哈希和自身校验后整体移动，已有目标拒绝覆盖。新版清单校验导入后场景哈希；旧清单可能返回 `scene_hash_recorded_at_import=false`，表示只验证网格，无法证明 XML 未改。要建立完整基线，应从已知原始来源重新导入到新的资产包，而不是手动改哈希绕过检查。绝对 `meshdir`、绝对网格路径以及指向包外的依赖会被拒绝，避免换机器后失效。

## 5. 第一轮：无图像的物理基线

可先分别定位问题：

```bash
.venv-sim/bin/python scripts/run_tests.py \
  --config configs/mujoco.toml
.venv-sim/bin/python scripts/run_sim.py --mode hold --seconds 10
.venv-sim/bin/python scripts/run_sim.py --mode pick-place --seconds 8
.venv-sim/bin/python scripts/smoke_pi05_contract.py \
  --output-dir outputs/policy-debug-01
```

`run_tests.py` 比普通 `unittest` 更严格：失败、错误、没有测试、任何测试跳过都返回失败。测试数量会随契约测试变化，关键是零错误、零失败、零跳过。控制异常锁存、IK 缓存隔离、姿态时效和夹爪接触等物理测试必须在此环境实际执行。

推荐用完整验收入口统一保存结果：

```bash
.venv-sim/bin/python scripts/validate_sim.py \
  --config configs/mujoco.toml \
  --output-dir outputs/server-baseline-01 \
  --repeats 5
```

执行顺序：`pip check` → 资产 → 严格回归 → 10 秒保持姿态 → 5 次独立进程的固定场景抓放 → 50 帧参考动作时序。无图像入口显式禁用 GL 后端，避免 CPU 物理验收依赖 EGL。任一步失败即停止，返回非零退出码，并保存 `report.json` 和阶段日志。

`--seconds 8` 在抓放模式表示**任务开始前的姿态保持时间**，不是整个抓放的最大时长；报告中的总仿真时间还包括抓放、释放、退离和稳定阶段。`--timeout` 默认每阶段 600 秒，属于墙钟超时。

输出目录必须不存在。重跑改为 `server-baseline-02` 等新目录，不覆盖失败记录。需要延长超时：

```bash
.venv-sim/bin/python scripts/validate_sim.py \
  --output-dir outputs/server-baseline-02 --repeats 5 --timeout 1200
```

## 6. 第二轮：EGL 双相机

物理通过后，选择一张可用 GPU 进行渲染：

```bash
MUJOCO_EGL_DEVICE_ID=0 .venv-sim/bin/python scripts/validate_sim.py \
  --config configs/mujoco.toml \
  --output-dir outputs/server-rgb-01 \
  --repeats 5 --render --gl egl
```

`MUJOCO_EGL_DEVICE_ID` 选择 EGL 可见设备；`CUDA_VISIBLE_DEVICES` 选择 CUDA 可见设备。容器和调度系统中的编号可能不同，应分别核对。整个仿真验收不要求占用四张卡。显式 `--gl` 在图像入口优先生效，无图像入口仍禁用 GL。

自动检查只确认 RGB 是 `uint8`、尺寸正确、像素在空间上存在变化；纯色图片即使三个通道不同也会被判失败。还应打开以下图片人工检查：

- `pick-place-rgb/front_cam.png`：机器人、桌子、释放后的方块和目标区域是否可见。
- `pick-place-rgb/R_wrist_cam.png`：腕部视角是否正确，是否被夹爪完全挡住。
- `policy-contract/head_left.png`、`wrist_left.png`：参考特征键映射与视野是否一致。

默认两路均为 640×480。`head_left` 当前是外部前视替代视角，`wrist_left` 历史名称对应物理右腕，`camera_calibrated=false`。参考头部原始分辨率为 1920×1536；单纯修改像素数不等于完成相机标定。

若服务器已有 OSMesa 运行库，可单独定位软件渲染路径：

```bash
.venv-sim/bin/python scripts/validate_sim.py \
  --output-dir outputs/server-osmesa-01 --repeats 1 --render --gl osmesa
```

OSMesa 通过不等于 EGL 通过；没有该运行库时不要把此命令当作必过步骤。EGL 失败先检查对应日志、运行库和设备权限，不要直接更换共享驱动。

## 7. 读取报告和保存基线

| 文件 | 核对内容 |
| --- | --- |
| `report.json` | `success`、错误阶段、配置及哈希、源码逐文件哈希、Git 状态、版本、GPU、每阶段命令/耗时 |
| `environment-freeze.txt` | 此次验收实际使用的完整 Python 包版本 |
| `unittest.json` | 测试总数、错误、失败、跳过明细，必须全部满足严格条件 |
| `dependencies.log` / `assets.log` | 环境冲突、资产数量及场景/清单哈希 |
| `hold.log` | 保持姿态误差、警告计数和运行结果 |
| `pick-place-000.log` 等 | 是否真正抬起方块，释放并退离后的目标距离、高度、速度及跟踪指标 |
| `present-task/task-report.json` | 脸前展示阶段、停留、回位核验和任务结果 |
| `pick-place-rgb/metrics.json` | 第一次任务的指标与截图对应关系 |
| `policy-contract/metrics.json` | 样本频率、时间误差、8 维输入、夹爪裁剪、RGB 尺寸 |
| `policy-contract/trajectory.npz` | 每帧动作后的仿真时刻、状态、请求动作、限速后目标 |

可直接打印摘要：

```bash
.venv-sim/bin/python - <<'PY'
import json
from pathlib import Path
for directory in ('outputs/server-baseline-01', 'outputs/server-rgb-01'):
    path = Path(directory) / 'report.json'
    if not path.exists():
        continue
    report = json.loads(path.read_text())
    print(directory, 'success=', report['success'], 'error=', report.get('error'))
    for stage in report['stages']:
        print(' ', stage['name'], stage['success'], stage['log'])
PY
```

抓放成功条件包含：物体曾被抬起；松手后退离；方块中心 XY 在目标半径内；高度接近桌面；平移速度小于 0.05 m/s；夹爪已张开。蓝色盘子是无碰撞的目标标记，不是具有边缘碰撞的容器。不能用当前结果证明装箱、脸前展示标定、扫码、动态工件或完整流水线可用。

`physics_time_factor` 是仿真时间/物理运行墙钟时间，不含渲染，也不含模型推理；不是端到端策略速度。目标变化率限幅不保证真实关节速度或碰撞安全。重复五次同一场景用于检查回归稳定性，不是跨场景策略成功率。

诊断 `trajectory.npz` 的状态在动作执行**之后**记录；它不是可直接训练的 LeRobot 数据集。训练数据还需要动作执行前观测、同步图像、语言/任务、episode 边界及明确的动作标签。保留全部报告、配置、源码和资产清单，修改参数后重新生成独立基线。

## 8. 参数修改与策略接口

默认先不要改配置。完成基线后，复制本地参数文件再试验：

```bash
cp configs/mujoco.toml configs/mujoco.local.toml
.venv-sim/bin/python scripts/validate_sim.py \
  --config configs/mujoco.local.toml \
  --output-dir outputs/custom-profile-01 --repeats 5
```

`configs/*.local.*` 被 Git 忽略。新的 `--config` 会传到物理测试和运行器；旧 Show-Harness 回归读取同一资产包的原始场景。任务和姿态测试针对当前桌面基线，移动工件超出可达区域或修改关节范围时，应先重新设计轨迹与验收条件，不能只放宽阈值。

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `simulation.model` | `models/cruzr_s2/scene.xml` | 相对路径以仓库根目录为基准 |
| `simulation.physics_dt` | 0.002 s | 物理步长 |
| `simulation.control_hz` | 50 Hz | `step()` 频率，一个周期须包含整数个物理步 |
| `simulation.max_joint_speed` | 0.6 rad/s | 目标指令变化率上限 |
| `simulation.max_gripper_speed` | 0.1 m/s | 夹爪目标变化率上限 |
| `policy.pi05.sample_hz` | 15 Hz | 参考数据采样频率，不是已确定的 DreamZero 频率 |
| `policy.pi05.action_horizon` | 50 | 参考动作块最多帧数 |
| `gripper.max_width` | 0.045 m | 仿真几何最大开口 |
| `camera.head_source/wrist_source` | `front_cam` / `R_wrist_cam` | 参考图像键对应的模型相机 |

完整 8 维绝对位置接口：七个右臂关节角 rad，最后一维夹爪目标 m，顺序见 [interfaces.md](interfaces.md)。示例只做保持姿态，无学习策略：

```bash
MUJOCO_GL=disable .venv-sim/bin/python - <<'PY'
import numpy as np
from cruzr_sim import CruzrEnvironment
env = CruzrEnvironment()
try:
    observation = env.observe_pi05(images=False)
    action = np.r_[observation['observation.state'][:7], 0.05]
    chunk = np.repeat(action[None], 50, axis=0)
    history = env.step_pi05_chunk(chunk)
    print('simulation_time:', history[-1]['simulation_time'])
    print('requested:', history[-1]['requested_action'])
    print('applied:', history[-1]['applied_action'])
finally:
    env.close()
PY
```

50 个样本约推进 3.334 秒。`step()` 是 50 Hz，`step_pi05()` 是参考 15 Hz；不要混用来表达同一个策略时钟。整个动作块先校验再执行；执行中若物理异常，已执行部分不能回滚，环境锁存故障，必须 `reset()` 后开始新 episode。输入 0.05 m 开口裁剪至 0.045 m 并记录标记；这仍不是已标定的 SDK `pos` 到几何开口映射。

## 9. DreamZero 模型侧复现：独立阶段

本节给出官方模型诊断的操作顺序与限制，**不是已经在 4090 上跑通的配方**。来源为 `third_party/dreamzero/README.md` 及源码快照 `ab790c198fbce33503358efbbd4187ce9a89adf3`，文件指纹见 [dreamzero-source.json](dreamzero-source.json)。本次保持官方快照不变。

### 9.1 先理解四卡限制

当前源码有比“总显存够不够”更明确的约束：

- `socket_test_optimized_AR.py:init_mesh()` 将所有进程放进单个 `ip` 组。
- `wan_flow_matching_action_tf.py:parallelize()` 明确要求 `ip_size` 只能为 1 或 2；这里是条件/无条件预测的并行路线，不是四卡参数切片。
- `sim_policy.py` 在常规非 lazy 路径先 `model.to(device=device)`，并捕获并行异常只打印 `Skipping parallelization`。因此“启动进程未立即退出”也不能证明分片成功。

**不要直接设置 `--nproc_per_node=4` 来尝试把 14B 权重摊到四张 24GB 卡上。** 单纯删掉上述断言也没有实现正确的四卡算法。按 14B、每参数 2 字节粗算，仅参数就约 28GB，尚未包含编码器、激活和缓存；实际显存还受 checkpoint 和加载路径影响。四张卡不能视为一张 96GB 卡。

在这台服务器上，优先顺序是：完成 A/B/C → 确认兼容的权重分片、量化或 CPU offload 路线 → 单次模型推理 → Cruzr 新本体闭环。较小 Wan2.2-5B 是另一条训练/适配路线，不能直接替换 14B checkpoint 的文件名使用；详见官方 [WAN22_BACKBONE.md](../third_party/dreamzero/docs/WAN22_BACKBONE.md)。

### 9.2 建立独立模型环境与下载

只在需要模型诊断时执行，仿真 A/B/C 不需要这一步。官方源码依赖较重，安装前检查目标驱动对所选 PyTorch/CUDA wheel 的支持；CUDA Toolkit、wheel 携带的运行库和 NVIDIA 驱动不是同一个版本概念。

```bash
conda create -n dreamzero-model python=3.11 -y
conda activate dreamzero-model
cd /data/cruzr-dreamzero/third_party/dreamzero
python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129
MAX_JOBS=8 python -m pip install --no-build-isolation flash-attn
python -m pip check
python -m pip freeze > /data/cruzr-dreamzero/outputs/dreamzero-environment-freeze.txt
```

这是上游安装入口，当前依赖包含 TensorRT 等包，不保证在目标服务器一次安装成功；保留安装日志并按实际冲突适配。4090 不照抄上游 GB200 专用 NVFP4 TensorRT engine 或 cu13 安装步骤。服务源码会设置 `ATTENTION_BACKEND=TE`，对应 attention 实现在 TE 不可用时回退 FA2；若 TE 已安装但运行不兼容，则要据实际日志处理，不能仅靠外部设置同名变量保证覆盖。

独立检查模型环境看见的 GPU：

```bash
python - <<'PY'
import torch
print('torch=', torch.__version__, 'cuda_runtime=', torch.version.cuda)
print('cuda_available=', torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, 'GiB=', round(p.total_memory/2**30, 2), 'cc=', torch.cuda.get_device_capability(i))
PY
```

官方 DROID 服务诊断使用 DROID 权重；有可用网络、磁盘和访问权限后下载：

```bash
hf download GEAR-Dreams/DreamZero-DROID \
  --local-dir /data/cruzr-dreamzero/checkpoints/DreamZero-DROID
```

将实际下载的模型 revision、配置和权重哈希与测试记录一起保存。Cruzr 新本体微调可研究官方 AgiBot 路线，但 AgiBot 权重也不是现成 Cruzr 权重。

### 9.3 模型诊断命令及成功标准

下面是官方两进程调用形式，只应在已经解决每卡加载显存问题、兼容依赖与权重后使用；原样在两张 4090 上可能 OOM。命令用于明确接口，不能把它当作通过四卡部署验收的步骤：

```bash
cd /data/cruzr-dreamzero/third_party/dreamzero
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 socket_test_optimized_AR.py \
  --port 5000 --enable-dit-cache \
  --model-path /data/cruzr-dreamzero/checkpoints/DreamZero-DROID
```

另开终端进入同一模型环境和目录：

```bash
conda activate dreamzero-model
cd /data/cruzr-dreamzero/third_party/dreamzero
python test_client_AR.py --host 127.0.0.1 --port 5000 --num-chunks 1
```

测试客户端使用官方三路视频和构造状态，不连接机器人，也不使用 Cruzr 仿真。服务入口默认监听 `0.0.0.0`，按服务器现有访问控制使用测试端口；本机客户端访问 `127.0.0.1` 不会改变服务监听地址。

应检查全部 rank 无异常、动作形状/数值有限且与权重契约一致，记录首次预测/热预测耗时和每张卡的显存。`nvidia-smi` 的采样只能近似观察峰值，准确峰值应在推理进程中记录。端口开启、`/healthz` 返回成功或看到 `Skipping parallelization` 都不是 D 层验收通过。若 OOM，保留日志回到显存分配/模型方案设计，不要结束其他人的任务，也不要假定增加进程数就能解决。

## 10. Cruzr 新本体闭环：仍需实现的工作

官方 [DATASET_TO_GEAR_AND_TRAIN.md](../third_party/dreamzero/docs/DATASET_TO_GEAR_AND_TRAIN.md) 描述「LeRobot v2 数据 → GEAR 元数据 → embodiment 注册 → modality/transforms → 数据配置 → 训练」。以下是本项目接下来的工作顺序，不是已有脚本的功能：

1. 准备有语言任务和同步双相机的 Cruzr 示范，核实 LeRobot 版本、15 Hz 时间戳、动作是否为绝对关节目标、关节零位和夹爪标定。Pi0.5 参考列名可复用，但不能假定其模型输出协议就是 DreamZero 协议。
2. 建立 Cruzr 数据映射：七关节 `[0,7)`，夹爪 `[7,8)`；不要照抄官方六关节示例的 `[0,6)` / `[6,7)`。确认模型内部相对动作变换及反变换，再接绝对目标执行端。原数据不可覆盖，先在独立副本做转换。
3. 注册 `cruzr_s2` embodiment，配置两路图像、采样偏移、状态与动作统计、归一化、padding/mask 和数据混合。当前官方推理入口固定 `oxe_droid`，需与新权重一起适配，不能仅更改字符串或填第三路假相机。
4. 使用匹配的新本体权重训练/微调，或建立经验证的兼容适配路线。4×4090 的训练 batch、分片、优化器、offload/LoRA/检查点策略要单独测显存，本文不提供未经验证的训练启动参数。
5. 增加仿真客户端：同一时间获取图像和状态，发送语言任务，接收并反归一化动作，校验关节顺序/单位/有限值/限位，再送入明确频率的动作执行器。维护模型 session/cache 与 episode reset 的对应关系。
6. 先做同步闭环（推理期间物理暂停），再做延迟、动作块重规划和异步执行实验。保存请求、响应、仿真时间与墙钟时间，区分任务失败、动作拒绝、物理异常和服务超时。
7. 在多个初始位置/场景上定义独立评估；oracle 抓放只能作为底座回归，不计入 DreamZero 成功率。

真机迁移属于上述 E 层之后的独立工作：头部/腕部相机内外参、关节零位、SDK 状态有效性和时间戳、夹爪映射、命令模式和执行错误处理均需核实。当前不提供自动驱动真机入口；采集主题与 SDK 聚合主题的差异见 [sdk-reference.md](sdk-reference.md)。

## 11. 常见失败的处理顺序

| 现象 | 优先检查 |
| --- | --- |
| 找不到 Python 3.11 | 指定 `SIM_PYTHON`，或使用独立 Conda；不要复用 Python 3.10 venv |
| `pip check` 失败 | 是否把 DreamZero 与仿真装在同一环境；查看实际版本冲突 |
| 模型不存在 | 是否只复制了 Git 代码，未另行导入授权资产 |
| 资产目录已存在 | 是完整旧包还是失败遗留；先校验/备份，再决定重导，不强制覆盖 |
| 哈希/路径校验失败 | 是否漏文件、改过 XML、保留绝对路径；恢复已知来源，不改 manifest 绕过 |
| `unittest` 显示 skipped | 依赖、Python 版本或资产未就绪；严格验收会判失败 |
| 配置后测试读错模型 | 使用 `run_tests.py --config ...`；直接 unittest 主要用于默认基线 |
| 保持姿态失败或 MuJoCo warning | 查增益、执行器、初始碰撞、时间步和模型来源；不要先放宽阈值 |
| IK 不收敛、抓放超时 | 查坐标系、物体位置、关节限制和抓取几何；先恢复默认配置复查 |
| EGL 报错但物理通过 | 查 EGL 库、设备权限与设备编号，独立定位图像后端 |
| RGB 纯色/视野不对 | 检查真实相机名、位置、遮挡、离屏尺寸；尺寸正确不代表已标定 |
| 验收输出目录存在 | 换一个新目录，保留旧报告 |
| DreamZero OOM / 四进程失败 | 核对 1/2 进程限制与整模型加载路径，先解决模型显存方案 |

完整留存 B/C 两轮报告及图片之后，才把对应代码和资产版本作为后续策略实验基线。当前修改的已执行检查、未执行物理验证与历史参考边界见 [validation.md](validation.md)。
