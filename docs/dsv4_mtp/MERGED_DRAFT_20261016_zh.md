# 193：mtp2_3 合并草稿执行候选（2026-10-16）

本轮按用户要求继续批量修改；只做静态检查，没有启停服务、导入候选模块、编译算子、发起推理或压测。下一轮建议记为 mtp2_3，但启动参数仍用 mtp2。

## 已有结果

以下完全采用用户提供的总吞吐（tok/s），不把外部参考视为同机同参数测量。

| 实验 | 1k_r0 | 1k_r1 | 4k_r0 | 4k_r1 |
|---|---:|---:|---:|---:|
| mtp2_0 | 2837.06 | 3487.23 | 7287.08 | 7550.41 |
| mtp2_1 | 3386.56 | 3474.71 | 7439.29 | 7269.84 |
| mtp2_2 | 3384.46 | 3275.52 | 7367.94 | 7341.24 |

mtp2_2 相比 mtp2_1，两轮平均1K约低2.93%，4K基本相同。这个样本不足以证明某项改动导致退化，但没有支持继续默认开启全部后端候选的收益证据。

## 差异及处理状态

对照容器中干净的 vLLM-Ascend 412cda26814ff70c326f6eb6510f1b610f67bbc0；不声称已完整迁入当前 main 或所有原生能力。

| 项目 | 本轮处理 | 默认/边界 |
|---|---|---|
| 草稿整段图回放 | 新增 FL 自己的 MergedDraftGraph，将 K 次模型forward、草稿head/argmax、步间hidden/token依赖与输出stack一起capture/replay | 已改，MTP_MERGED_GRAPH=1；尚未运行验证 |
| 多步草稿执行入口 | 新增统一 runnable，metadata/context 全部在首次forward前准备，图回放时不重跑各步Python模型调度 | 已改，依赖prepared及direct路径 |
| 后续步输入三次copy | 将生成的ids、每步positions和hidden直接传入下一次forward，去掉复制回input_ids/positions/hidden_states三个buffer | 已改，MTP_DIRECT_INPUTS=1；eager也生效 |
| 每步forward context | 预建并保存各步context，执行时只切换context；纠正draft/prefill标记，每步独立MC2 mask，重置MoE层索引 | 已改；没有把不同token extent混成同一context |
| 草稿图输入地址与元数据 | 保留capture时Tensor强引用；回放前只拷贝地址变化的输入，核对结构/shape/stride/dtype/device，涵盖RoPE代理内部Tensor | 已改；SAS、slot、seq_lens、query_start、block table仍每轮刷新 |
| 图与DP一致性 | 在原来一次packed DP collective中增加eligibility行；只有所有rank均为纯decode且首步/后续精确extent相同时才一起进入图 | 已改；mixed/prefill/idle/不同DP batch统一eager，不额外pad请求 |
| 图缓存/资源 | 最多4个精确shape；其余新shape走eager，不在运行中逐出图；捕获前同一输入执行两次算子预热 | 已改；日志打印DSv4 merged MTP graph captured；首次捕获可能增加预热时间 |
| 首步SWA元数据 | 将原本仅后续单query快路径扩展到首步纯decode、最多K+1 query；首步使用预留末尾slot/SAS/RoPE缓冲，与后续步隔离 | 已改；prefill/混合/profile仍用通用builder |
| 接受数CPU快照 | 每次新D2H拷贝清空缓存；第一次等待event后缓存列表，本轮多处读取复用它 | 已改；保留首次event同步和拒绝修正，未读取未完成拷贝 |
| 目标/bonus词表搬运 | 纯greedy且没有有效惩罚、约束、bad words、logit bias、min_tokens、thinking状态时，对原logits做一次argmax，再选择target/bonus行 | 已改，MTP_TARGET_ARGMAX=1；避免两次词表宽度的行gather；未知/custom processor回原采样路径 |
| mtp2_2后端候选 | 保留源码，但默认关闭Attention输出复用、缩小MC2 workspace、shared-expert overlap和fallback GMM候选 | 已改默认DSV4_BACKEND_OPT=0；不是宣称这些算子已证明更慢 |
| 草稿局部argmax | 继续使用vLLM局部分片max及小候选聚合 | 前轮已有，MTP_LOCAL_ARGMAX=1 |
| 拒绝后输入与位置kernel | 继续使用融合next-token、padded indices、位置/slot/长度更新 | 前轮已有，MTP_FUSED_INPUTS=1 |
| greedy拒绝验证kernel | 保留当前已融合验证；没有照搬Ascend按block串行遍历请求的grid，也没有新增只适用K1的特化 | 保留；K1特化不改善本次K2，不能把不同grid一律视为原版更快 |
| Attention/MoE/EP核心 | 主干与参考已接近；保留现有FUSED_MC2路由、权重/clamp、EP通信顺序和活动mask | 不整体切换插件，不替换ABI |
| MTP head/e_proj/h_proj量化 | 当前checkpoint实际为BF16/FLOAT，保持数学与权重 | 不强行量化，表面传参差异不是INT8能力遗漏 |
| FlashComm hidden通信 | 保留完整pre-HC hidden的正确切分/gather，目标模型stash也在参考代码中存在 | 未删除必要通信或恢复旧03/06补丁 |
| hc_pre/hc_post与残差clone | 对照源码同样存在，继续使用现有原生算子 | 未当作FL独有差异修改 |
| 目标图同步 | 当前target FULL图回放屏障保留；原版也有对应顺序保障 | 未因猜测删同步；新DSA草稿图无外部attention-task参数更新 |
| 预热及正式压测参数 | 保留每轮64预热请求，正式256请求、CC64、1K/4K各两轮、greedy/seed及输出长度不变 | 原有保留；预热不进入正式吞吐计时 |
| 硬件/版本/外部参考设置 | 不属于源码补丁，原版4000/8000的TP/DP/K/并发与硬件仍未知 | 不能据此保证本版达到某吞吐 |

## 使用

宿主机目录：/data/tiankuan/hzj/model_test/dsv4-mtp-infyuki-20261015

```bash
bash stop_service.sh mtp2
bash start_service.sh mtp2
bash test_vllm_bench.sh mtp2
```

本轮没有替你执行上述命令。当前已经运行的旧服务不会自动加载本轮代码，需你自行重启。

新草稿图由专用wrapper管理；speculative_config.enforce_eager仍保留true，用来禁止不兼容的通用vLLM草稿图，不能仅据此字段判断专用图是否生效。有效条件还包括GRAPH=1、MTP_PREPARE_STEPS=1、MTP_FAST_METADATA=1、MTP_DIRECT_INPUTS=1及DP一致性条件。检查服务日志中的capture消息；请求开关开启不等于每批命中图。

只关闭新草稿图，保留直接传参及其余修改：

```bash
MTP_MERGED_GRAPH=0 bash start_service.sh mtp2
```

保留本版源码但回到先前执行/采样路径：

```bash
MTP_MERGED_GRAPH=0 MTP_DIRECT_INPUTS=0 MTP_TARGET_ARGMAX=0 MTP_FAST_METADATA=0 bash start_service.sh mtp2
```

这条不是逐文件原版复现：接受数快照缓存仍保留。若要恢复mtp2_2的全部候选，还需DSV4_BACKEND_OPT=1。精确源码回退到上一提交fcb43e9，并恢复备份中的manifest及三个启动文件。

备份目录：/data/tiankuan/hzj/model_test/dsv4-merged-draft-20261016/。保留before.tar.gz、dsa_v1.before.py、manifest/脚本备份、补丁、提交号及新manifest；历史runs/results不覆盖。

代码只通过静态AST、bash -n、git diff --check与源码身份核对。图捕获、跨rank运行、正确性、接受率和收益均待用户测试，不把“已修改”写成“已验证”。
