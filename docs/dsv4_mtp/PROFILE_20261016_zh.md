# DSv4 MTP2：先profile后的优化优先级

2026-10-16。已按本轮授权在fl-infyuki-v024-hzj启动专用服务、采集并停止该服务，恢复采集前无模型服务的状态。没有改动FL/vLLM模型或MTP源码，没有启用03/06，没有重跑正式AISBench吞吐测试。

## 采集与证据

- 基于已部署主机元数据优化与MTP2图档位修复，FL HEAD为61ae2938e02cd3f843425139e9ef5dfe1cdd0a9f，manifest SHA为6004b14d3a1fe2876b1bca4cf2070b9f95cf89bcedb63e742c86409c90fd9f82。
- TP4、DP4、EP16、目标FULL_DECODE_ONLY、异步调度，草稿仍eager。只为诊断启用PROFILE=1与VLLM_CUSTOM_SCOPES_FOR_PROFILING=1，使用FL现有CPU/NPU Level1 profiler。
- 使用原AISBench固定4K数据前64条，seed1024、temperature0、ignore_eos=true。先CC64、每请求输出32 token预热，再CC64、每请求输出1024 token采集；全部64条流已开始输出后启用profiler，每个worker记录16轮迭代。预热和采集各64条均成功，usage核对输入4096、输出分别32/1024。没有重试。
- 16个rank均导出了trace_view.json、kernel_details.csv、op_statistic.csv、step_trace_time.csv等。每rank窗口约1.329–1.344秒。目录使用服务器时间命名，不作为用户本地时间显示。
- torch_npu无法在daemon worker中在线解析，故保留约2.147GB原始数据副本，使用独立非daemon进程离线解析，最终parse及汇总退出码均0。初始化阶段曾停止一次尚未发送请求的服务，以补齐原默认关闭的阶段标记；该次不计为采集成功或失败样本。

远端产物根目录：

`/data/tiankuan/hzj/model_test/dsv4-mtp-infyuki-20261015/profile-20261016`

原始副本：`traces-raw-before-analysis/`；完整导出：`runs/mtp2/traces/*_ascend_pt/ASCEND_PROFILER_OUTPUT/`；请求记录：`captures/20261016-015219/`；汇总：`analysis/trace_summary.json`。本地verified/保存客户端证据与导出汇总，aggregate.json为可复算的全rank统计。

## 主要结果

以下为16个rank各自比例/耗时的算术平均，不把16个rank的时间相加当作墙钟时间。

| 指标 | 平均 | rank间范围 |
|---|---:|---:|
| CANN Computing / Stage | 59.03% | 57.74%–60.70% |
| 未重叠Communication / Stage | 6.94% | 4.27%–9.97% |
| Free / Stage | 34.03% | 29.65%–37.03% |
| 主机preprocess scope／轮 | 11.35ms | 11.00–11.75ms |
| 主机draft scope／轮 | 22.35ms | 21.83–22.88ms |
| 主机sample scope／轮 | 47.08ms | 46.54–47.62ms |

Computing是CANN分类，包含DispatchFFNCombine等带EP通信的融合任务，不等于纯数学计算。6.94%也不是所有通信的总成本。Free表示CANN所统计的计算/通信区间之外的时间，不代表已证明全部可消除。

主机scope包含异步设备依赖与等待。尤其sample的47.08ms不能解释为采样算法本身耗时47ms；目标forward的主机scope仅约0.37ms，是图发起时间，不能解释为目标模型NPU执行仅需0.37ms。

代表rank DP0/TP0的Free区间共约427.52ms，其中约217.41ms落在host draft scope内，153.02ms落在host preprocess scope内，两者约占该rank Free的86.6%。这是时间上的重合，支持优先排查这些路径，但不是单独证明某段CPU代码造成NPU空闲的因果实验。

最大单项算子类型是编译后的DispatchFFNCombine，平均累计约221.54ms/rank/window；eager版本另约29.93ms。融合算子无法仅由此区分FFN算力、路由及等待EP对端的贡献。可作为草稿发起改造后的下一层排查对象。

## 为什么调整优化顺序

CPU→NPU flow成功关联了每rank的32次草稿LM-head matmul及32次相应allgatherAicpuKernel，另有16次目标head。匹配使用CANN官方kernel CSV与trace flow，不根据kernel发生时间硬猜所属阶段。

草稿head matmul平均累计8.24ms/rank/window，相关all-gather约9.63ms；后者约为整个采样窗口的0.72%。这是kernel累计时长与窗口的比值，不是可直接回收的墙钟时间，也不是局部top-1优化的总收益上限：CPU发起、transpose/copy、argmax等仍会变化。但它不足以支持“完整词表通信就是主要瓶颈”的初步假设，应降低该候选优先级。

约17%–18%的kernel行有可用的直接阶段flow关联，主要是eager路径；目标图内部大部分kernel没有这种直接关联。不能据此编造完整target/draft NPU耗时比例。全窗口CANN分类和head关联是本轮较可靠的证据。

建议下一轮按以下顺序，各自单独验证：

1. **草稿分步元数据与发起路径。** 将首步多query/partial-prefill和后续单query分开处理，复用持久slot、RoPE与SAS缓冲，减少Python小张量操作与重复准备。现成build_for_drafting有draft_index-1等约定，须适配后使用。preprocess也应单独标记接受数等待与metadata构建，避免为消除等待而读取未完成的结果。
2. **在元数据地址与语义稳定后适配Ascend草稿图。** 目标是减少eager算子发起和阶段间空隙。需要完整处理capture/replay、DP协调与fallback，不能直接翻转enforce_eager。
3. **检查融合MC2的小批量成本及EP等待。** 特别比较草稿小批量与目标图内的融合任务，先区分设备计算与rank等待，再决定通信或调度策略；本轮不据算子名直接归责。
4. **draft-only局部top-1。** 继续保留为独立候选，优先级低于前述发起/元数据改造。全局enable_reduce_sample仍禁止与speculative decoding同用，不能直接打开。

这轮没有baseline的同条件profile，也没有1K或其他K值的profile，因此不能用它解释所有1K/4K轮间波动，不能推出MTP相对baseline的精确收益。下一轮候选的收益须回到不带profiler的原AISBench协议验证，并保持相同预热条件；profile数据不计作正式吞吐成绩。

## 交付状态

| 项目 | 应用/启用状态 | 入口 | 验证与回退 |
|---|---|---|---|
| 专用profile服务及阶段标记 | 已运行采集，现已停止 | 本目录start_profile.sh mtp2 | 身份校验停止自身服务；原启动脚本未修改 |
| 固定CC64稳态采集客户端 | 已运行，预热/采集各64成功 | run_profile.sh mtp2 | usage和HTTP状态均核对；结果独立保存 |
| 离线解析与统计 | 已完成，16/16 rank | parse_offline.py、analyze_trace.py、aggregate.py | 原始副本及trace SHA保存；无源码回退需求 |
| 新优化代码、03/06 | 未添加/应用/启用 | 无 | 无需回退 |

复用本次目录重采前需新建唯一输出位置，或另建新日期/批次目录，避免覆盖本次trace。客户端虽会新建时间戳子目录，服务trace目录本身仍需分批隔离。

本轮未调整长期技能默认规则。
