# DSv4 Attention / MoE / MC2 / EP 后端候选（2026-10-16）

承接920c125；仅修改、静态核对，不导入候选，不执行编译、测试、推理或压测，不启停服务。

本次对照容器vLLM-Ascend 412cda26814f：DSA prefill/decode计算、TokenDispatcher主要函数与EP prepare/finalize已基本一致。不能把同样的代码换个import就视为提速；没有整体切换到vllm_ascend后端。

| 项目 | 本次处理 | 状态与边界 |
|---|---|---|
| Attention输出中转 | 纯decode/纯prefill、无padding且shape/dtype/连续布局匹配时，直接复用attention输出做逆RoPE/o_proj，免分配和复制 | 已修改；混合批与不匹配布局保留原路径 |
| 融合MC2 workspace | 小批量按DP同步后的padded行数×EP×topK的最坏路由上界，加每本地expert两个128行tile余量，向256对齐；不超过原131072上限 | 已修改；大批/prefill保留原上限，无均衡假设，不裁token，无额外D2H |
| MC2参数访问 | 在transport创建时缓存固定mode和容量上限 | 已修改 |
| shared-expert调度 | 启用FL中已经迁入的原生Ascend多流重叠路径，保留before/after事件依赖及join | 启动配置已开启；沿用原有启动期一致性校验 |
| 普通MoE W8A8回退 | 在单权重布局、int8、标准SwiGLU、无offset/动态EPLB时接通已有GMM+SwiGLU+quant算子，传递clamp_limit | 已修改；当前融合MC2主路径不经过此分支，不承诺命中 |
| EP路由/dispatcher/combine | 与参考实现已接近，保持rank/group、topK权重、通信次序与活跃mask | 未重写 |
| Attention Q/KV多流 | 两边默认已经开启 | 保留，未重复宣传为新增 |
| 原生C++/CANN算子 | 使用当前FL已构建的算子，未替换ABI或改权重 | 无需重新编译 |
| 旧03/06开关 | 保持关闭；本次是独立DSv4范围的候选，workspace边界变化单独记录 | 未启用旧补丁 |

所有新候选均未测试，收益和正确性待用户验证。workspace优化并非原生实现已有而FL漏迁的差异：双方原先均用固定上限，这是本次新增候选。

宿主机仍使用原实验目录和启动/压测脚本。DSV4_BACKEND_OPT默认1，同时作用于目标模型和MTP草稿；TP4/DP4、K、图模式及正式压测参数不变，预热仍每轮64请求。候选只在自行重启后生效。

```bash
cd /data/tiankuan/hzj/model_test/dsv4-mtp-infyuki-20261015
bash stop_service.sh mtp2
bash start_service.sh mtp2
bash test_vllm_bench.sh mtp2
```

只关闭本轮后端候选、保留上一轮MTP改动：先自行停止对应服务，再运行

```bash
DSV4_BACKEND_OPT=0 bash start_service.sh mtp2
```

完整源码回退：停止自己的服务，切回codex/dsv4-mtp-batch-opt-20261016（920c125），恢复本轮备份的manifest和start_service.sh、serve_in_container.sh、launch_config.py。备份在宿主 `/data/tiankuan/hzj/model_test/dsv4-backend-opt-20261016/`。旧runs/results未改动。
