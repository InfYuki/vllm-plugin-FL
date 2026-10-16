# 193：关闭默认静态编译，单独尝试共享 argmax

本轮基于 `6d560ea`，分支 `codex/dsv4-verify-argmax-20261016`。已直接应用到服务器源码与现有启动脚本，不需要再手工应用补丁。未重启服务、执行模型、精度测试或压测，效果待验证。

## 静态编译的处理

已检查容器内 torch-npu 的 `inference/_cache_compiler.py`、静态编译实现及固定版本 vllm-ascend 编译入口。当前 `cache_compile` 接受绑定到 `nn.Module` 的方法，而 FL 使用 vLLM 的分段 FX 后端；不能只给现有 backend 增加缓存目录就完成复用。原版保存 FX 图代码的路径会跳过包含 Triton wrapper 的图，其代码缓存也不能直接证明静态算子 `.run` 包可在重启后复用。

本轮没有完成这两套接口与静态包生命周期的可靠整合，也没有跨进程重启验证，因此选择用户允许的回退方案：**`STATIC_KERNEL` 默认改为 0**。保留显式 `STATIC_KERNEL=1`，但它仍可能产生此前的长启动开销；不宣称缓存问题已经解决。继续正常捕获 target decode 图，草稿仍为 eager。

用户提供的 mtp2_4 均值为 1K **3483.86**、4K **7401.745** total tok/s；相对 mtp2_1 均值约高 1.55% 和 0.64%，两轮样本不足以确定这是静态编译的稳定收益。它付出了约 30 分钟启动时间。

## 本轮唯一新增性能候选

贪心验证原先先复制 `logits[target_rows]` 与 `logits[bonus_rows]` 两块完整词表，再分别求 argmax。对于无约束的贪心采样，先沿词表求每行 argmax，再按 target/bonus 行索引 token ID，结果等价；可避免两块词表大小的中间复制以及两次独立 argmax 调度。

继续使用同一份完整词表 logits 和 `torch.argmax`，没有引入新设备内核、改变词表通信协议或路由逻辑。此思路曾包含在后来整批撤回的候选中，本轮只单独尝试带严格守卫的版本，不恢复那批的合并草稿图或其他执行改动。

| 项目 | 修改、依赖与状态 | 回退 / 验证 |
|---|---|---|
| 静态编译启动成本 | 启动包装、容器脚本、配置解析与 metadata 默认统一为 `STATIC_KERNEL=0`；无需新增依赖；已应用，下次启动生效 | 显式 `STATIC_KERNEL=1` 可重现旧候选，但缓存仍未接通 |
| 贪心共享 argmax | `MTP_VERIFY_ARGMAX=1` 默认请求启用，只影响 DSv4 验证采样；代码已应用，当前运行服务不会自动切换 | `MTP_VERIFY_ARGMAX=0` 使用原采样路径 |
| 等价性守卫 | 检查纯 greedy、无 logprobs、无 penalties、无 token mask/bad words、无活跃 thinking budget；仅允许已知内置 MinTokens/LogitBias 处于空状态。未知处理器及其子类保留原路径；FP64 等其他 dtype 保留原路径 | 不改变原先不支持的采样模式；约束采样继续执行已有处理逻辑 |
| 实际命中标识 | server.log 首次命中记录 `DSv4 MTP verification uses shared argmax before row selection.`；metadata 记录请求开关 | 开关开启不等于每个 batch 都满足守卫 |
| 启动脚本 | 现有 `bash start_service.sh mtp2`；参数保持 TP4、DP4、K2、target graph、draft eager | 默认静态编译关闭，新候选开启 |
| 测试脚本 | 现有 `bash test_vllm_bench.sh mtp2`；仍为每轮 64 次预热、256 正式请求、并发 64、1K/4K 各两轮 | 本轮逐字保留测试脚本，未执行 |
| 静态检查 | 实际采样方法 AST 的 list-backed tensor doubles 检查：300 组混合 K0..3、乱序索引、tie、正负无穷；12 类守卫拒绝条件；关闭候选/约束条件/其他 dtype 的回退调用；缓存日志参数；启动配置、bash、内嵌 Python AST、哈希与源身份 | 通过；模拟检查不代表真实 FP16/BF16 设备精度验证，NPU 正确性与性能仍未验证 |

## 用户测试命令

在 193 的原实验目录执行：

```bash
cd /data/tiankuan/hzj/model_test/dsv4-mtp-infyuki-20261015
bash stop_service.sh mtp2
bash start_service.sh mtp2
# READY 后：
bash test_vllm_bench.sh mtp2
```

如需同条件对照，停服务后仅关闭这项新候选，静态编译保持默认关闭：

```bash
bash stop_service.sh mtp2
MTP_VERIFY_ARGMAX=0 bash start_service.sh mtp2
```

本轮将静态编译从开改为关，因此不能把下一轮与 mtp2_4 的差值全部归因于共享 argmax；应与 `STATIC_KERNEL=0、MTP_VERIFY_ARGMAX=0` 的对照比较。候选与对照均保留原有预热、并发和输出长度。

备份、完整补丁、提交信息与检查脚本：宿主机 `/data/tiankuan/hzj/model_test/dsv4-verify-argmax-opt-20261016/`。长期习惯未修改，启动和压测仍使用直接命令与单独开关。
