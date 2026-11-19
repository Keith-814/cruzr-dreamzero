# Cruzr DreamZero 微调与任务接口 SOP

适用：2026-11-19 新增入口，Cruzr 右臂 7 关节 + 右夹爪、头部左相机 + 物理右腕相机。用户数据目前在四卡服务器上，路径和实际格式尚未核验；本文提供可指定路径的入口，不声称已读取或完成该数据集训练。

## 1. 先确定要训练什么

当前任务是「抓起工件 → 脸前展示并停留 → 放回原位」。先用 [部署 SOP](deployment-sop.md) 验证固定工位的动作流程，再训练模型学习靠近、抓取、搬到头部观察位和原位放回。当前不要求二维码、标签可读性或扫码结果。

参考工程的任务文本是拿起工件、到头部相机前拍照、放回；若现有数据也包含这个过程，可以按实际演示保留任务标签。需要覆盖成功抓持、脸前停留、回到支撑面、松手和退离的完整轨迹。只有视频而没有状态/动作命令，或者没有可靠时间同步时，不能直接用于当前行为克隆入口。后续增加扫码时再接独立解码器，不把已有展示数据重标为扫码成功。

## 2. 识别服务器上的数据格式

先在服务器定位目录，查看是否具有下面结构，不在本机拷贝或上传整套数据：

```text
DATASET_ROOT/
├── meta/info.json
├── meta/tasks.jsonl
├── meta/episodes.jsonl
├── data/chunk-000/episode_000000.parquet
└── videos/chunk-000/
    ├── observation.images.head_left/episode_000000.mp4
    └── observation.images.wrist_left/episode_000000.mp4
```

当前入口支持 **LeRobot v2**，要求 `codebase_version=v2.x`、每个 parquet 一个 episode。LeRobot v3、ROS bag/MCAP、原始视频 + JSON 需要先做格式/采集主题专项转换，程序会明确拒绝，不会猜字段。师兄转换器可作为原始采集格式参考，但仓库没有复制它的外部源码；原始格式确认后再接对应转换流程。

后续命令中的 `DATASET`、`WAN`、`BASE_CKPT`、`TOKENIZER` 必须换成实际目录。所有输出选新目录，输入数据只读保留。

## 3. 验证数据契约

复制并核对：

```bash
cd /data/cruzr-dreamzero
cp configs/dataset.example.json configs/dataset.local.json
```

必须确认：

- 状态和动作都是 8 维，七关节顺序与 SDK 名称对应，最后一维 `right_gripper`。
- 关节单位 rad；夹爪是 SDK 行程 m，不是 0/1、毫米或已归一化值。
- 原始动作为绝对位置目标；训练管线内部的相对变换另行处理。
- 每个样本的状态/图像在对应动作执行前获得，采集延迟和相机同步已核实。不能把动作后的仿真 `trajectory.npz` 当作这种数据。
- `fps` 与实际数据一致。参考为 15 Hz，不是硬编码所有 DreamZero 数据都必须 15 Hz。
- 相机 `wrist_left` 是历史列名，实际必须是右腕；特征 names 明确，不仅维度碰巧为 8。

实际确认之后，在本地契约中将 `semantics_verified`、`observation_precedes_action` 设为 true。模板默认 false，避免未经确认的数据被静默送进训练。软件数值检查不能证明采集时序或物理单位正确。

使用**独立模型环境**，按 [服务器复现流程第 9 节](server-validation.md) 安装官方依赖。数据工具需要 pandas、pyarrow、tqdm 和兼容 OpenCV。不要在这个环境执行 `pip install -e .` 安装本仓库整个仿真包，否则会引入 NumPy 2.2.6，与官方 NumPy 1.26.4 冲突。

```bash
conda activate dreamzero-model
python -m pip install pandas==2.2.3 pyarrow==19.0.1 tqdm==4.67.1
python -m pip check
DATASET=/path/to/actual/lerobot-v2-dataset
python scripts/check_dataset.py \
  --dataset "$DATASET" --contract configs/dataset.local.json \
  --output outputs/dataset-check-01.json
```

脚本检查所有 episode 的有限值、SDK 角度/行程范围、连续帧索引、时间间隔、任务文本、两路视频逐帧可解码、视频与动作帧数/fps 一致，并保存文件哈希。当前至少要求 2 个 episode 且每条不少于 25 帧，这是数据拆分/加载的技术下限，**不是训练有效所需示范数量**。需要足够覆盖真实物体位置、姿态、抓取和展示方向的成功演示。

发现不规则采样、不同关节顺序、度制或缺失语言时，先单独修复/转换到新副本，不通过修改阈值绕过检查。

## 4. 生成数据副本和 Cruzr 训练配置

```bash
python scripts/prepare_finetune.py \
  --dataset "$DATASET" --contract configs/dataset.local.json \
  --output-dir outputs/cruzr-finetune-01 \
  --validation-fraction 0.2 --seed 42
```

该入口不启动 GPU 训练、不下载权重，完成：

1. 全量只读数据检查。
2. 以 episode 为单位随机拆分训练/验证，避免相邻帧跨集合泄漏。相同场景或同次连续采集被分成多个 episode 时，还需按采集 session 分组，当前随机拆分不能替代跨工件/跨场景测试。
3. 建立独立副本；从原始 `task_index/tasks.jsonl` 或真实文本生成副本中的 `annotation.task`，不会把原始任务改为模板任务。
4. 复制官方源码至输出目录中的 `dreamzero/`，只修改这个派生副本；`third_party/dreamzero` 快照和许可保留。
5. 注册 `cruzr_s2`、projector 编号 33、双相机 modality/transforms/fps、动作 `[0,7)` 与夹爪 `[7,8)`。
6. 修正新本体语言 collator 分支，沿用双视角网格：头部左上，右腕左下，右半为空。这里的空白是显式训练布局，不是第三路假相机。
7. 用上游转换器为训练/验证生成 GEAR 元数据。每个 split 会重新编号为连续的本地 episode id（上游转换器按 0..N-1 枚举文件），`split_file_mapping` 保留原始 id、帧数和视频对应关系；全局 `index`、`task_index`、视频路径和统计元数据会按副本重建，验证集合使用训练集合统计量。
8. 保存准备报告、拆分 ID、契约与副本/生成文件哈希。失败留下报告，但训练入口拒绝 `success=false` 的工作区。

```text
outputs/cruzr-finetune-01/
├── prepare-report.json
├── convert-train.log
├── convert-validation.log
├── dreamzero/
│   ├── cruzr-overlay.json
│   └── groot/vla/configs/data/dreamzero/cruzr_s2_relative.yaml
└── datasets/
    ├── train/
    └── validation/
```

准备会复制一套数据，预留相应磁盘空间；原始数据、权重和生成工作区不提交 Git。所有 schema/transform 源码补丁以精确锚点检查，上游版本变化时失败退出，不静默拼接。

## 5. 训练入口：先计划，再解析，再短训练

准备官方 DreamZero-AgiBot 等**经确认与 Wan2.1 14B 路线匹配**的全模型基础 checkpoint、Wan2.1 组件和 tokenizer。当前候选入口沿用官方 YAM LoRA 训练配置；不保证适用任意 DROID/5B/仅 LoRA checkpoint。程序不自动下载。

```bash
BASE_CKPT=/path/to/DreamZero-AgiBot
WAN=/path/to/Wan2.1-I2V-14B-480P
TOKENIZER=/path/to/umt5-xxl

python scripts/train_cruzr.py \
  --prepared outputs/cruzr-finetune-01 \
  --checkpoint "$BASE_CKPT" --wan "$WAN" --tokenizer "$TOKENIZER" \
  --output-dir outputs/cruzr-train-plan-01 --gpus 4 --max-steps 20
```

默认只打印完整命令，不训练。入口检查准备结果、生成源码/数据哈希、组件路径，设置 action loss 本体白名单为 `[33]`，防止训练看似运行却没把 Cruzr 动作纳入正确的损失范围。

先在模型环境解析 Hydra 配置：

```bash
python scripts/train_cruzr.py \
  --prepared outputs/cruzr-finetune-01 \
  --checkpoint "$BASE_CKPT" --wan "$WAN" --tokenizer "$TOKENIZER" \
  --output-dir outputs/cruzr-compose-01 --gpus 4 --max-steps 20 --compose
```

应检查 `resolved-config.json` 中数据根目录、`cruzr_s2`、2 路相机、8 维拆分、15 Hz（或你的已核实频率）、`action_loss_embodiment_ids=[33]`、相对动作设置和模型维度。`--compose` 只做 Hydra/OmegaConf 解析，不导入 Torch、不实例化训练、不加载权重，也不是模型加载或显存验收。

明确要开始训练时：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_cruzr.py \
  --prepared outputs/cruzr-finetune-01 \
  --checkpoint "$BASE_CKPT" --wan "$WAN" --tokenizer "$TOKENIZER" \
  --output-dir outputs/cruzr-train-smoke-01 --gpus 4 --max-steps 20 --launch
```

这只是 20 步接线/显存 smoke 配置，不是有效训练配方。默认每卡 batch 1、梯度累积 4、LoRA、BF16、学习率 1e-5，关闭 W&B 上传；均为待服务器验证的起点。可用 `--deepspeed /path/to/reviewed.json` 传现场评审的 ZeRO/offload 配置，程序不自动选择或保证 24GB 显存足够。

**四进程训练与官方 1/2 进程推理不是同一种并行。** DDP/LoRA 并不自动将冻结的 14B 基座分摊到四张卡。即使用 ZeRO，也要核实初始化时机、权重加载、冻结模块、激活、编码器与优化器占用；原样启动仍可能 OOM。发生 OOM 时保留日志，先解决模型分片/offload/量化或匹配的较小 backbone 路线，不能仅增加 `--gpus`。5B 路线需要配套架构/权重与新配置，当前生成器没有把 14B 参数冒充成 5B。

本轮没有在 4090 上安装模型环境、解析 Hydra 或训练，因此上述生成配置是**候选适配入口**，不是已证实可跑通的四卡配方。

## 6. 训练必须核对的结果

短训练通过后检查：

- 数据预览的关节顺序/两路视角与标签一致；不能把腕部左右写反。
- 训练实际包含 LoRA 与动作/状态相关模块；参数梯度和更新存在。
- `loss`、`action_loss_avg`/`action_loss`、`dynamics_loss_avg` 等日志有限；动作损失不能因为本体白名单错误而整段恒为零。
- 实际每卡显存、step 耗时、数据加载时间、checkpoint 是否可完整重载。
- `training-run.json` 的退出码和模式；解析配置成功不等于训练成功，20 步成功不等于学会任务。

`datasets/validation` 已隔离，但当前启动器沿用上游训练器，**未自动执行验证集评估或挑选最佳模型**。应建立使用同一训练归一化统计的离线 action/视频误差评估，再做仿真闭环。不能用训练 loss 下降替代任务成功率，也不能用验证数据重新计算归一化后当作无泄漏评估。

正式训练步数/批量/冻结策略依据示范规模与验证表现选择，不提供未经实验的“多少条必成功”承诺。LoRA 文件还依赖基座、编码器、tokenizer、配置和统计量；模型发布应一并记录它们的 revision/哈希，不能只复制一个 adapter 权重。

## 7. 将模型接到展示任务

任务层接口见 [部署 SOP 第 7 节](deployment-sop.md)。模型插件负责：

```text
观测（两路 RGB + 七关节实测角 + 夹爪实测 SDK 行程）
  → Cruzr embodiment 的 transforms / 训练相同的双视角布局
  → 模型预测
  → 反归一化、反相对动作变换
  → (N,8) 绝对目标，rad/m
  → 整块校验，短前缀执行，刷新观测
```

`configs/policy.example.json` 只是契约模板，不含已训练模型。工厂 `module:create_policy()` 必须提供 `reset()` 和 `predict(observation,instruction=...,phase=...)`；`phase` 是任务上下文，不能声称模型仅因收到字符串就已学会该阶段。

```bash
# 在能访问模型服务且有仿真渲染的环境运行；先准备自己的 Cruzr 策略插件。
MUJOCO_EGL_DEVICE_ID=0 .venv-sim/bin/python scripts/run_present_task.py \
  --policy site_policy:create_policy --policy-manifest configs/policy.local.json \
  --output-dir outputs/policy-present-motion-01
```

同一任务层继续管理抓持确认、展示停留、返回与释放；扫码仅是后续可选阶段。默认模型运动接口不允许模型在运输途中自主张开夹爪。仿真 `observation.state` 的末维暂为几何开口近似，真实数据是 SDK 行程；在跨环境适配器中依据标定转换，并匹配训练相机视角。当前外部前视不等同于真实头部相机，不应据此承诺 sim2real 效果。

真正的 Cruzr 模型服务仍需用新本体权重和元数据实现/验收；上游 `socket_test_optimized_AR.py` 固定 DROID，本生成器没有把它直接包装成正确的 Cruzr 服务。这里提供稳定插件接口，而非假定任意 DROID checkpoint 都能控制 Cruzr。

## 8. 本次已做与未做

已提供：数据契约模板、只读检查、按 episode 拆分/副本转换、独立源码派生、Cruzr 本体/双相机/损失配置、计划/解析/启动入口、任务模型插件边界与失败处置。

未实测：你的实际数据、pandas/pyarrow 全量转换、Hydra 解析、checkpoint 兼容加载、GPU 训练/验证、模型服务、学习策略闭环和真机。下一次有数据目录后，先从 `check_dataset.py` 输出确定格式和采集问题，再确定训练路线。
