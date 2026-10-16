# 105 MTP 补丁与 107 FL 实现对比

日期：2026-10-16。本次只读读取 105/107 的补丁、源码、启动参数、安装映射和已有日志。没有在服务器应用补丁、修改服务配置、发推理请求或启停服务。105 没有部署任何文件，所有快照与分析写在本地。

## 先确认比较对象

105 指定目录 `/data/tiankuan/fanyc/models_test_opt/dsv4f/mtp_patches`：

| 文件 | 实际作用 | 不能从中得出的结论 |
|---|---|---|
| 0001-deepseek-v4-mtp-for-v024-ascend.patch | 为 FL 注册 DeepSeekV4MTPModel，新增 521 行 MTP 模型；复用 FL 优化的 decoder/MoE | 没有修改 proposer、拒绝验证、异步调度或元数据热路径；不是一套完整性能优化补丁 |
| 0002-enable-mtp-in-dsv4f-launch.patch | 增加 method=mtp、num_speculative_tokens=1 启动参数 | 不等于目前 105 正在运行 MTP1 |

README 明确以 FL `61ae2938e02cd3f843425139e9ef5dfe1cdd0a9f` 为基础，依赖已有 target-hidden-state bridge；补丁生成时没有跑性能测试。文档中的应用/重启命令只作为资料读取，本次没有执行。

105 当前 11223 的实际服务位于 `vllm-ascend-test-fanyc-1016` 容器。其日志明确显示 `Platform plugin ascend is activated`，Python 安装目录仅见 vLLM 与 vLLM-Ascend 分发包，Ascend editable 映射为 `/vllm-workspace/vllm-ascend/vllm_ascend`。启动参数为 **MTP2 + enforce_eager=true + async-scheduling + FULL_DECODE_ONLY**。因此它是原生 Ascend 实现的参考，不能把该服务的成绩归为这两个 FL 补丁的成绩。

107 本次实际读取 `fl-infyuki-v024-hzj` 中 `/workspace/vllm-plugin-FL-InfYuki`；HEAD 同为 61ae2938e，含未提交的 FL MTP 适配。最后一次 mtp2 launch_metadata 记录 manifest 为 `7aff57fe670b016789b8c20bdc0440ed3e8b33b8ec001f7d8ff3918da3168ce7`，fast_metadata=true，目标图开启、草稿 eager、异步开启。

## 能确认的实现差异及性能影响

### 1. MTP 模型数学结构基本相同，优先排查外围执行路径

两者均为 embedding/hyperconnection hidden 各自归一化与投影，相加后过一个 draft decoder，再由 hc_head、norm、LM head 生成 logits。权重加载主体也基本相同。`model_comparison.diff` 保存完整差异。

FL 额外处理 FlashComm token 分片、返回 hidden 展平和 all-gather、量化描述别名及 SWA-only 配置隔离。这些包含正确性适配，不能简单删掉来追求速度。105 补丁保留 `@support_torch_compile`，FL 没有该装饰器；但是两边当前草稿均强制 eager，不能据此声称装饰器就是速度差异，也不能只加装饰器就认为已支持草稿图。

FL 将 e_proj、h_proj、shared head 的 quant_config 设为 None，105 补丁传入 quant_config。传入配置不等于这些权重实际为 INT8，仍取决于量化描述和权重；此次没有证据把这个差异列为主因。FL decoder/MoE 本身仍接收量化配置。

### 2. FL 的拒绝验证固定走多段 PyTorch 操作，Ascend 提供融合内核路径

FL `spec_decode/dsv4_mtp.py:215` 的 Dsv4GreedyRejectionSampler 调用 `dsv4_mtp_utils.py:111`，每轮构造 counts 设备张量，执行 cumsum、gather、比较、cumprod、sum、where 和散写。小批量下，多个 eager 算子发起和临时张量可能比其数学工作更贵。

105 原生 `sample/rejection_sampler.py:539` 在 HAS_TRITON 条件下调用 `rejection_greedy_sample_with_triton`，all_greedy 时直接返回；其环境安装了 triton 与 triton_ascend。尚未采集该次实际内核 trace，不能从安装包存在单独证明每轮都使用融合分支，但代码路径差异明确。

这是可独立优化的具体对象：复用适配 NPU 的 greedy 验证融合内核，保留 FL 已有的零 draft、首拒绝、bonus、混合请求长度和 -1 padding 语义。不能直接替换整个 sampler，因为目标端 logits processor 等接口也要对齐。此项收益尚未做消融测量；此前 sample scope 的 47ms 包含设备等待，不能当作可节省的采样计算时间。

### 3. FL 在草稿步之间穿插元数据构建；Ascend 预先准备多步后再运行

FL Dsv4MTPProposer 继承通用 EagleProposer。通用 `llm_base_proposer.py:610` 的后续步循环先更新位置、重建元数据，再设置 forward context、运行模型与采样。FL `_advance_host_metadata:136` 还读取接受数，计算 CPU 位置/长度/slot 的镜像；设备端也执行位置/slot 运算。

105 `AscendSpecDecodeBaseProposer._propose:758` 先构建 `multi_steps_attn_metadata`，再进入 `_run_merged_draft:1088`；后续草稿步使用预建元数据。这样避免在两个草稿 forward 之间重新穿插通用 builder。它仍有 CPU 工作与 eager 算子，不能把方法名中的 merged 解释为一次融合模型调用。

107 已部署 `build_for_dsv4_mtp_decode:1144` 快路径，减少后续 SWA 元数据的冗余 RoPE/QLI 工作；但只覆盖真实 proposal 中 draft_index>0、单 query 等满足条件的后续步。首步仍走通用 builder，拒绝验证和其他位置运算也没有融合。因此这轮小修补无法等同于完整 Ascend proposer 的执行组织方式。

### 4. 接受数回传仍形成依赖，不能用“异步已开”推导主机没有等待

FL runner `_prepare_dsv4_async_host_metadata` 与 proposer `_advance_host_metadata` 通过 `_get_valid_sampled_token_count` 等待 D2H event 并生成 Python list。这个依赖会限制 CPU/NPU 重叠，MTP2/3 还要处理后续步。实际阻塞时长需 trace 细分。

但 105 也有 `_correct_optimistic_seq_lens_cpu:1625`，同样等待接受数 event，再用 CPU 镜像修正长度。因此“105 完全没有同步、FL 有同步所以慢”是错误结论。应比较等待发生的位置、是否已完成、每步重复工作，而不是删掉正确性所需的同步。

### 5. baseline 仍付出 MTP hidden-state gather/copy 成本，但不是后来适配新增的

FL `models/deepseek_v4_ascend.py:1080` 无条件分配 `_mtp_hidden_buffer`，forward `1134` 起在 FlashComm 下做完整 HC residual all-gather 并 copy，即便没有 speculative_config。baseline 可以通过按 MTP 实际启用状态设置构造期门控来避免这部分成本；需要验证图捕获与 runner 接口。

本次 `git show HEAD:...` 证实该行为已在 61ae2938e 原分支存在；105 当前 Ascend target 代码也包含相同 stash。因而它是可优化的共同开销，**不能作为这次 FL 新增 MTP 代码导致 baseline 回退、或 FL 相对 105 更慢的单独证据**。未测量可节省时间。

## 与既有 profile、吞吐证据的关系

之前 107 的 MTP2、4K、CC64 profile 中，CANN Free 平均约 34.03%，主机 preprocess/draft scope 平均 11.35/22.35ms。代表 rank 的 Free 大多与这两个 scope 重合。这支持优先检查发起与元数据链路；不构成上述某一段代码独立造成全部空闲的因果证明。该 profile 在最新 SWA 快路径部署前采集，尚无部署后的同条件 profile。

草稿 head 相关 all-gather 内核累计仅约采集窗口 0.72%，不能继续把完整词表通信认定为首要瓶颈。HC hidden-state all-gather 是另一类通信，不应混为一项。

以下统一为 **输出 token/s**，只列已找到的历史记录：

| 实现/历史记录 | 1K | 4K |
|---|---:|---:|
| 105 Ascend baseline（1K 两轮平均，4K 一轮） | 1452.92 | 1319.92 |
| 105 Ascend MTP2（1K 两轮平均，4K 一轮） | 1833.37 | 1561.71 |
| 107 AISBench baseline（两轮平均） | 1490.12 | 1348.28 |
| 107 AISBench MTP1 后一批（两轮平均） | 1654.75 | 1478.56 |
| 107 AISBench MTP2 后一批（两轮平均） | 1547.78 | 1414.64 |

105 为本机 vllm bench random，107 为 AISBench 固定输入；均 CC64、256 请求、输出 1024，但客户端、实际文本、预热、轮数及源码批次不同。107 此处性能汇总为 10 月 15 日记录，不能代表 10 月 16 日新快路径的性能。该表提示 **MTP 额外执行成本抵消收益** 的方向，不能视为严格跨机 A/B 或据此计算某个补丁的精确收益。

用户此前的 1K 约2900、4K 约6500若来自 total throughput，应分别除以约2、约5才能与 output throughput 比较；两种口径混用会制造看似巨大差距。上述已找到的 107 baseline 总吞吐为 2980.24/6741.38，其输出吞吐即1490.12/1348.28。

两边现有日志都存在较高接受率样本；107 最新 mtp2 日志摘录有平均接受长度 2.55–2.66、平均草稿接受率约77–83%。这些是不同请求/时段的样本，不能用来证明两个实现随机输入上的接受率完全相同，也没有证据支持“FL 完全没有正确使用 MTP 权重”的推断。

## 优化建议和当前状态

先做一项独立候选：**融合 greedy 拒绝验证**，同时细分其纯 CPU 发起与设备执行时间。随后对比 Ascend 的多步元数据预建方式，再推进草稿图完整适配。baseline hidden stash 门控另作独立 baseline 候选。每项都需相同固定输入、相同预热的正确性与吞吐 A/B；记录接受数、实际 tokens/iteration、target/draft/metadata 设备时间与 CPU 间隙。

两边当前都是 target FULL_DECODE_ONLY、draft eager；不能用“105 开图、FL 没开图”解释差距。105 的目标捕获档位与 FL MTP2 的12倍数档位不同，但多 query 对齐约束不同，不能盲目照搬105档位。

| 项目 | 状态 | 应用、启动、回退 |
|---|---|---|
| 两个补丁与实际运行源码对比 | 已完成；本地保存证据 | 无服务器改动，无需回退 |
| 新候选、03/06、草稿图 | 本次均未添加/启用/测试 | 本次没有应用或启动命令 |
| 性能原因归因 | 已确认结构差异，强候选为验证与元数据/发起链路开销 | 具体收益和各项占比尚待单变量 A/B，不承诺加速倍率 |

本次没有修改长期技能规则。
