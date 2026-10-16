# DSv4 MTP 批量性能候选（2026-10-16）

按本次用户要求合并多项候选，一次交付。全部属于**已修改、下次启动默认请求启用、未测试**；不能视为已经证明正确或加速。保留独立开关，便于后续归因。

| 可能导致性能差异的项目 | 原差异 / 本次处理 | 本次状态 |
|---|---|---|
| 首轮冷JIT与缓存 | vLLM bench和AISBench性能脚本每轮先预热64请求，正式256请求及1K/4K各两轮不变；精度脚本不加预热 | 已改；未执行 |
| 草稿步间metadata构建 | 增加DSv4专用prepared路径，首个forward前准备全部后续metadata；保留步间真实token/hidden依赖 | 已改，MTP_PREPARE_STEPS=1 |
| DP协调两次 | 将首步token数、后续batch数打包成一次DP collective，空闲rank dummy也使用同布局与K次forward | 已改，随prepared路径；未验证跨rank运行 |
| 步间CPU张量工作 | 接受数校验及位置/长度镜像用NumPy批量处理，保留CPU block-table边界与活动掩码 | 已改 |
| 接受数D2H等待 | 等待移到prepared路径的首步前；现有event同步保留，不读取未完成数据 | 调整位置；未删除同步 |
| 最后有效token与接受数 | 多个张量操作改为单Triton kernel，保留discard/零draft语义 | 已改，MTP_FUSED_INPUTS=1 |
| 拒绝后采样行/拒绝数 | padded indices计算改为单Triton kernel | 已改，MTP_FUSED_INPUTS=1 |
| 后续GPU位置/slot/长度更新 | 单Triton kernel写每步独立持久buffer，保留越界/负block/已失活mask | 已改，prepared + fused inputs |
| 新kernel随batch变化重编译 | batch数与block-table尺寸使用运行时参数 | 已改；未执行JIT |
| 草稿完整词表all-gather | 新增DSv4 get_top_tokens，使用已安装vLLM局部argmax实现，只聚合各TP候选；维持hc_head/norm、padding与tie规则 | 已改，MTP_LOCAL_ARGMAX=1；只作用于draft |
| SWA metadata快路径 | 保留已有后续单query快路径；prepared依赖其每步隔离buffer，关闭MTP_FAST_METADATA也关闭prepared | 原有保留；补充依赖关系 |
| 首步通用metadata builder | 仍需处理prefill/混合批；未用decode-only假设替换 | 未改 |
| 每步forward context | 新路径仍为每个forward建立正确context，未实现原生Ascend整段graph wrapper | 未改 |
| 草稿ACL graph / compile | 仍强制eager；未完整适配capture/replay、DP协调、持久地址与fallback | 未改 |
| greedy拒绝验证 | 继续使用已有融合kernel；不把新输入准备优化混称为greedy优化 | 原有保留，MTP_FUSED_GREEDY=1 |
| greedy kernel grid/K1特化 | 当前FL每请求program，Ascend有不同组织；无profile证据支持本次再改其grid | 未改 |
| 目标词表argmax、bonus采样、logits processor | 保留既有语义；不绕过采样约束，未开全局reduce-sample | 未改 |
| FlashComm hidden切分/gather | 两边组织位置不同但都需要正确布局；本次保留，未未经profile删通信 | 未改 |
| MTP head/e_proj/h_proj量化 | 当前权重实为BF16/FLOAT；保持既有权重与数学，未为匹配传参强行量化 | 未改 |
| attention/MoE/MC2/EP后端 | FL与Ascend后端存在差异；未整体替换目标/草稿decoder算子 | 未改 |
| shared-expert多流/静态kernel等配置 | 当前关闭状态保持；原版参考配置未知，不能把开关全开当作同条件修复 | 未改 |
| baseline hidden stash | 属目标分支共同开销，不能解释全部MTP2差距；本次不启用旧03/06补丁 | 未改 |
| tokenizer输入、硬件/驱动、TP/DP、K、并发、预热与吞吐口径 | 新预热明确记录；其余服务/正式压测参数保持。4000/8000为非同条件外部参考 | 预热已改；其他需对照证据 |

宿主机用法仍为：

```bash
cd /data/tiankuan/hzj/model_test/dsv4-mtp-infyuki-20261015
# 用户自行停止旧服务后，重新启动才会加载新代码
bash start_service.sh mtp2
bash test_vllm_bench.sh mtp2
# 或沿用AISBench：bash test_performance.sh mtp2
```

每轮先发送64个同长度/输出上限的预热请求，预热不进入正式吞吐计时；这也会覆盖K相关kernel的首次调用，但不保证覆盖所有可能的batch形状。WARMUP_REQUESTS可调整，做对照时所有组保持一致。

禁用本轮主要NPU/执行候选（用户先停止当前服务）：

```bash
MTP_PREPARE_STEPS=0 MTP_FUSED_INPUTS=0 MTP_LOCAL_ARGMAX=0 bash start_service.sh mtp2
```

上述开关回退保留本轮CPU NumPy镜像实现；如需逐文件恢复原代码，先停服务，切回 `codex/dsv4-mtp-ascend-handoff-20261016`，并将备份 `manifest.before.json` 恢复到实验manifest。完整备份在宿主 `/data/tiankuan/hzj/model_test/dsv4-mtp-batch-opt-20261016/`，包括before.tar.gz及旧脚本；历史runs/results未覆盖。

只做静态语法/差异和身份记录核对，不导入候选、不执行JIT/CPU或NPU测试、不启动服务或发送请求。新代码只在用户下次启动服务时生效；已有服务不作为新代码成绩。
