# Cruzr DreamZero

面向 Cruzr S2「抓取 → 扫码 → 分拣」的实验工程。当前版本包含从服务器已有 Show-Harness 仿真基础整理出的 MuJoCo 资产导入、按关节名的固定周期执行环境、七自由度 IK、夹爪、桌面方块抓放和相机接口。

**当前尚未接入 DreamZero、真机 SDK、扫码器或运动流水线。** 抓放脚本使用已知物体位置的确定性轨迹，仅验证仿真与控制基础，不能代表学习策略成功率或真机性能。

## 安装与运行

推荐 Linux + Python 3.11/3.12。依赖与 DreamZero 分开安装。

```bash
git clone https://github.com/Keith-814/cruzr-dreamzero.git
cd cruzr-dreamzero
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/import_assets.py --source-root /data1/oylx2
python scripts/check_mujoco.py
python -m unittest discover -s tests -v
python scripts/smoke_cruzr_pick_place.py --verbose
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
- `configs/`：供后续适配使用的示例；当前脚本不读取该配置。

开发时保持 editable 安装；默认资产路径位于仓库内，当前不支持独立 wheel 携带模型发布。

## 后续步骤

1. 验证现有抓放与相机，记录初始基线。
2. 增加静止工件、扫码目标位、料框与任务评价。
3. 增加同步关节/夹爪数据与按固定频率执行的动作块接口。
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

`run_sim.py` 的 `pick-place` 仍是 oracle 位置控制，用于验证仿真控制链路，不是视觉策略，也不是 DreamZero。服务器上建议先用 `--mode hold` 检查模型、驱动和 EGL，再做渲染。

SDK 对照和仿真参数边界见 [`docs/sdk-reference.md`](docs/sdk-reference.md) 与 [`configs/mujoco.toml`](configs/mujoco.toml)。
