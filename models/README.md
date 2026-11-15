# 机器人资产

`cruzr_s2/` 是本地资产目录，已被 Git 忽略。服务器模型未附明确许可证，不将其标为 Apache 2.0，也不上传其 URDF、MJCF、STL 或 USD。

在有权使用现有资产的服务器上导入：

```bash
python scripts/import_assets.py --source-root /data1/oylx2
```

脚本只读来源，复制场景引用的 STL，并把 MJCF 的 meshdir 改为相对路径；目标存在时拒绝覆盖。记录源场景与网格 SHA-256。原始 URDF 和 USD 保留在现有目录，后续再核实再分发许可。

默认模型为本目录下 `cruzr_s2/scene.xml`。可通过环境变量 `CRUZR_MODEL_PATH` 指向另一个完整 MJCF 场景。
