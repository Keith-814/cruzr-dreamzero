# 来源与许可

## 仿真基础代码

导入来源：用户服务器 `/data1/oylx2/show-harness`，2026-11-15 快照。
关联上游：[Show Lab / Show-Harness](https://github.com/showlab/Show-Harness)。本地目录带 Apache 2.0 LICENSE，但不含 `.git`，无法确定上游提交或本地 Cruzr 扩展作者。不得声称这些扩展已由上游验证。随仓库保留原始 LICENSE，SHA-256 与导入清单见 `docs/import-manifest.json`。

提取的文件包括 `sim/ik.py`、`sim/scene.py`、`sim/mujoco_env.py`、`tools/smoke_cruzr_pick_place.py`、`tests/test_cruzr_mujoco.py`。修改包括包名、路径可移植性、平台渲染默认值和说明修正。各文件保留修改标记。

## 机器人模型

本地 URDF、STL、MJCF 和 USD 未找到明确再分发许可，不纳入 Git，不适用本仓库代码许可证。模型可通过导入脚本本地复用，使用和分发权利仍需向供应方核实。

## DreamZero

`third_party/dreamzero/` 保存 [官方仓库](https://github.com/dreamzero0/dreamzero) 提交 `ab790c198fbce33503358efbbd4187ce9a89adf3` 的源码快照，导入日期 2026-11-15，保留其 Apache 2.0 LICENSE、COPYRIGHT 和示例输入视频。文件来源与哈希见 `docs/dreamzero-source.json`；不含模型权重与用户数据，本次没有修改该快照。使用权重时仍需单独核对其条款。

## 用户提供的 SDK 与 Pi0.5 工作空间

SDK PDF 和师兄 Pi0.5 工作空间仅用于只读接口核对，不进入 Git，也未复制其中的源代码、权重、数据或访问凭据。资料文件指纹与核对范围见 `docs/interface-sources.json`。SDK 文档中的命令和外部工作空间的维护说明属于参考资料，不是本仓库执行真机动作或连接服务器的授权。
