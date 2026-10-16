# 193：静态算子编译候选（2026-10-16）

本轮在 `a7419be` 恢复版上只新增一个性能候选：接通 vllm-ascend 已有的静态算子编译。分支为 `codex/dsv4-static-kernel-20261016`。代码已经应用，下一次使用原启动脚本时默认请求启用；没有重启当前服务，没有执行模型、精度测试或压测，收益未知。

| 项目 | 改动与作用 | 当前状态 / 回退 |
|---|---|---|
| 编译入口 | 将 `enable_static_kernel` 传到 npugraph_ex 的 `static_kernel_compile`，解除 FL 配置层与编译层的旧拒绝门禁 | 已应用；下次启动默认请求启用；未做设备验证 |
| 多卡协调 | 根据本地 TP 和 DP 数设置并检查 `LOCAL_WORLD_SIZE`；当前 TP4×DP4 对应 16 | 已应用；由静态编译开关控制 |
| 编译形状 | 按原版方法限制为 decode 图捕获范围；当前 mtp2 为 12、24、…、192 | 已应用；避免把所有 prefill 范围交给静态编译 |
| MoE 层索引 | 将已有的静态编译索引回绕逻辑接到配置；处理编译和捕获两次 forward 导致的层索引递增 | 已应用；与原版同一条件，防止索引越界 |
| 退出清理 | 移植原版 worker 的 SIGTERM/SIGINT 静态包清理；清理方法 AST 与固定版本原版一致 | 仅在启用静态编译时注册；未执行安装或卸载 |
| 前置依赖 | 已确认 torch-npu 源码支持静态编译与形状范围参数，容器存在 `op_compiler`；未安装新依赖 | 只做文件与配置检查 |
| 启动脚本 | `bash start_service.sh mtp2`；增加一个 `STATIC_KERNEL` 环境变量并记录到启动 metadata | 默认 1；`STATIC_KERNEL=0` 回退，`GRAPH=0` 时自动不请求静态编译 |
| 压测脚本 | `bash test_vllm_bench.sh mtp2`；沿用每轮 64 次预热、256 个正式请求、并发 64、1K/4K 各两轮 | 本轮没有修改压测脚本，也未执行 |
| GMM 融合适配 | Python 空接口之外还缺少 `grouped_matmul_swiglu_quant_weight_nz` 的 FL 原生绑定，不能仅接 Python 方法 | 暂缓；`DSV4_BACKEND_OPT` 继续默认 0 |
| 验证 | Python AST（含 shell 内嵌 Python）、bash 语法、隔离配置检查、所有启动变体与回退配置等价检查、清理方法对照、文件哈希、git diff 检查 | 静态检查通过；精度与性能均待用户验证 |

参考固定版本：[vllm-ascend 编译入口](https://github.com/vllm-project/vllm-ascend/blob/412cda26814ff70c326f6eb6510f1b610f67bbc0/vllm_ascend/compilation/compiler_interface.py)、[worker 清理](https://github.com/vllm-project/vllm-ascend/blob/412cda26814ff70c326f6eb6510f1b610f67bbc0/vllm_ascend/worker/worker.py)。本轮没有修改 SwiGLU clamp、Hash 路由、MTP 数学语义或草稿 eager 执行方式。

在 193 宿主机执行：

```bash
cd /data/tiankuan/hzj/model_test/dsv4-mtp-infyuki-20261015
bash stop_service.sh mtp2
bash start_service.sh mtp2
# 显示 READY 后：
bash test_vllm_bench.sh mtp2
```

首次启动需要额外编译、安装静态算子包，可能明显慢于之前。启动 metadata 中 `enable_static_kernel=true` 表示请求启用；应结合 server.log 中实际编译、安装及是否跳过的日志判断，不能仅凭开关认定已加速。

回退这一项候选：

```bash
bash stop_service.sh mtp2
STATIC_KERNEL=0 bash start_service.sh mtp2
```

备份、补丁与静态检查脚本在宿主机 `/data/tiankuan/hzj/model_test/dsv4-static-kernel-opt-20261016/`。当前已运行的服务仍是修改前版本，只有用户重启后才会载入候选。未调整长期脚本习惯，本轮保持单开关和直接命令。
