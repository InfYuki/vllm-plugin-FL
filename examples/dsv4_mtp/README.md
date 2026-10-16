# DSv4 MTP 实验工作区

完整环境、进展、已知问题与后续规划：[工作交接文档](../../docs/dsv4_mtp/HANDOFF_zh.md)。

本目录保存从107实际环境导出的启动/客户端/AISBench脚本，以及新服务器工作区生成器。最新融合 greedy 候选**尚未测试**；准备脚本本身也尚未在新服务器运行。

`scripts/` 是生成器使用的模板，含原环境路径；不要直接在仓库的模板目录启动服务。

已有兼容的镜像、FL editable 安装与本地模型后，在**新容器内**执行：

```bash
cd /workspace/vllm-plugin-FL-InfYuki
python examples/dsv4_mtp/prepare_workspace.py \
  --experiment-dir /workspace/model_test/dsv4-mtp-infyuki-20261015 \
  --container fl-infyuki-v024-hzj
```

它只创建脚本、payload 和匹配当前已提交 Git HEAD 的 manifest，不启动服务、不评测、不安装依赖。拒绝覆盖现有实验目录；不要在107原工作区运行。将旧 `aisbench/data/` 迁移到新实验目录；源码、Python包版本和数据哈希参见交接文档。

之后在宿主机执行（示例宿主机 `/data/tiankuan/hzj` 挂载到容器 `/workspace`）：

```bash
cd /data/tiankuan/hzj/model_test/dsv4-mtp-infyuki-20261015
bash start_service.sh mtp2
bash aisbench/test_accuracy.sh mtp2
bash aisbench/test_performance.sh mtp2
```

以上命令由用户自行执行；切换组别先停止自己启动的服务。`MTP_FUSED_GREEDY=0` 关闭本轮融合候选，`MTP_FAST_METADATA=0` 关闭上一轮后续SWA元数据快路径，两者是独立开关。源码初始快照由 source_snapshot.json 固定，不要为了绕过身份检查随意改 SHA。
