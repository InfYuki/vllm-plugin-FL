# DSv4 MTP：融合 greedy 拒绝验证候选

按用户要求，本轮只修改并部署，不运行测试：未执行语法/导入检查、CPU/NPU 测试、精度评测或压测，也不启停服务。部署仅核对文件传输与覆盖前的 SHA，确认备份；这不构成正确性验证。

在 FL 自有 `vllm_fl/spec_decode/dsv4_mtp_greedy.py` 新增 Triton 内核，每请求一个 program，一次输出接受前缀、首个拒绝处的 target token 或全部接受后的 bonus，并写好其余 -1 padding。复用已有 GPU `cu_num_draft_tokens`，取消原验证函数中新上传 counts、cumsum、索引、cumprod、归约及多次输出写入等组合操作。按代码保留零 draft、可变 draft 长度、首拒绝、全接受与 bonus 语义，尚待用户实测确认。

改动仅接入 `Dsv4GreedyRejectionSampler` 的前缀验证部分。现有 bonus 采样、logits processor、词表 argmax、草稿 proposer、SWA 元数据、图模式、异步调度与03/06配置保持原样。内核归 FL 所有，不从 vllm_ascend 导入执行逻辑。

默认 `MTP_FUSED_GREEDY=1`，作用于 MTP1/2/3。NPU 上请求融合但 Triton 不可用时会报错，避免静默回退造成错误归因；关闭开关走上一版 PyTorch 验证。CPU 保留原参考路径。本轮没有执行任何分支。首次运行该内核可能发生 JIT 编译，正式性能对照应保持一致的预热。

## 用户自行启动、测试

107 宿主机：

```bash
cd /data/tiankuan/hzj/model_test/dsv4-mtp-infyuki-20261015
bash start_service.sh mtp2
# 服务就绪后自行评测
bash aisbench/test_accuracy.sh mtp2
bash aisbench/test_performance.sh mtp2
```

服务已经运行时先由用户自行停止，再启动。关闭本层融合做对照：

```bash
bash stop_service.sh mtp2
MTP_FUSED_GREEDY=0 bash start_service.sh mtp2
```

开启用 `MTP_FUSED_GREEDY=1`，MTP1/3 对应替换 variant。`runs/<variant>/launch_metadata.json` 新增 `dsv4_mtp_fused_greedy`，记录请求的开关状态；它不是运行或精度验证通过标记。baseline 不使用该 sampler。

## 状态与回退

| 项目 | 依赖、应用 | 应用/启用状态 | 验证、回退 |
|---|---|---|---|
| 融合 greedy 验证 | 当前 FL MTP 适配、Ascend Triton；本层 manage_greedy.py apply 部署 | 部署后下次 MTP 启动默认请求融合 | **未测试**；MTP_FUSED_GREEDY=0 回到原验证路径 |
| 源码、payload、manifest、启动记录 | 同步更新并备份 | deployment.json 记录具体文件及前后 SHA | 仅文件一致性保护，无推理验证 |
| 目标图、草稿 eager、异步、SWA、03/06 | 原配置 | 本轮不改动 | 无本轮独立回退需求 |

先自行停止使用本层代码的服务，再在107宿主机完整回退本层：

```bash
docker exec fl-infyuki-v024-hzj /opt/aisbench-venv/bin/python \
  /workspace/model_test/dsv4-mtp-infyuki-20261015/greedy-fusion-20261016/manage_greedy.py rollback
```

管理脚本拒绝覆盖后续更改，恢复前层源码、payload、启动脚本和 manifest，删除本层新增的两个内核文件副本，不启停服务。回退更早 opt 层之前必须先回退本层。每个候选只归因本层改动，性能和正确性尚未验证。

本次没有连接或改动105，也没有调整长期技能规则。
