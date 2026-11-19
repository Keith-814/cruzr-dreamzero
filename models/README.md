# 机器人资产

`cruzr_s2/` 是本地资产目录，已被 Git 忽略。服务器模型未附明确许可证，不将其标为 Apache 2.0，也不上传其 URDF、MJCF、STL 或 USD。

在有权使用现有资产的服务器上导入：

```bash
python scripts/import_assets.py --source-root /data1/oylx2
```

脚本只读来源，复制场景引用的 STL，并把 MJCF 的 meshdir 改为相对路径；目标存在时拒绝覆盖，全部复制和哈希完成后才整体落入目标目录。记录源场景、导入后场景与网格 SHA-256。原始 URDF 和 USD 保留在现有目录，后续再核实再分发许可。

默认模型为本目录下 `cruzr_s2/scene.xml`。可通过环境变量 `CRUZR_MODEL_PATH` 指向另一个完整 MJCF 场景。

迁移已导入的资产时应复制整个 `cruzr_s2/`，保留 `asset-manifest.json`，在目标机器运行 `python scripts/check_assets.py`。也支持 `scripts/import_assets.py --scene PATH --mesh-dir PATH` 导入其他目录中的同一独立 MJCF（不支持未展开的 include）。这些路径由用户准备，本仓库不提供虚构的 Cruzr 替代几何。

导入/校验不支持未打包的外部纹理、高度场、skin 或子模型文件；遇到这些依赖会明确失败。网格目录和文件引用必须可移植，不能保留机器绝对路径；程序化纹理和内联几何不受此限制。迁移和完整验收步骤见 [服务器复现操作流程](../docs/server-validation.md)。

`CRUZR_MODEL_PATH` 仅供旧 Show-Harness 接口使用；新 `CruzrEnvironment` / `run_sim.py` / `validate_sim.py` 使用 TOML 中的 `simulation.model`。相对模型路径以仓库根目录为基准。
