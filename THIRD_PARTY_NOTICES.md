# 来源与许可

## 仿真基础代码

导入来源：用户服务器 `/data1/oylx2/show-harness`，2026-11-15 快照。
关联上游：[Show Lab / Show-Harness](https://github.com/showlab/Show-Harness)。本地目录带 Apache 2.0 LICENSE，但不含 `.git`，无法确定上游提交或本地 Cruzr 扩展作者。不得声称这些扩展已由上游验证。随仓库保留原始 LICENSE，SHA-256 与导入清单见 `docs/import-manifest.json`。

提取的文件包括 `sim/ik.py`、`sim/scene.py`、`sim/mujoco_env.py`、`tools/smoke_cruzr_pick_place.py`、`tests/test_cruzr_mujoco.py`。修改包括包名、路径可移植性、平台渲染默认值和说明修正。各文件保留修改标记。

## 机器人模型

本地 URDF、STL、MJCF 和 USD 未找到明确再分发许可，不纳入 Git，不适用本仓库代码许可证。模型可通过导入脚本本地复用，使用和分发权利仍需向供应方核实。

## DreamZero

本版本未复制 DreamZero 代码或权重。[官方仓库](https://github.com/dreamzero0/dreamzero)与[新本体适配指南](https://github.com/dreamzero0/dreamzero/blob/main/docs/DATASET_TO_GEAR_AND_TRAIN.md)仅作为后续集成参考；使用时单独遵守其许可与权重条款。
