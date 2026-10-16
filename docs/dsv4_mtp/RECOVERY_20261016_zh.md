# 193：mtp2_3 失败后的恢复记录（2026-10-16）

## 证据与结论

- 用户贴出的 ConnectionRefusedError 是客户端无法连接127.0.0.1:1112：服务已经退出。
- 服务器 runs/mtp2/server.log 显示18:53:33引擎死亡，错误507011，设备报告DDR/MTE、UB地址越界。
- /root/ascend/log/debug/plog/plog-796_20261016185333711.log 指定故障kernel：MoeGatingTopKHash_c9fe881608378d9bec1f72b77dd57c53_6。
- Python在_get_valid_sampled_token_count的event.synchronize处接收到异步设备错误，不能据此把接受数同步函数判为故障算子，也不能删同步。
- 本次日志中“DSv4 merged MTP graph captured”数量为0。没有成功捕获记录，不足以单独归因于合并图回放，也不能区分本轮其余候选中的具体触发项。
- 失败实验为vllm_bench_results/mtp2/20261016-185307-1478040，0成功/256失败的轮次不可当作性能成绩。

## 处理

本版属于恢复版，不是新的提速结论。将mtp2_3新增的执行/采样组合撤回，恢复此前用户跑通过的prepared eager路径；没有宣称修好了底层MoeGatingTopKHash内核。

四个文件已逐字节恢复至fcb43e9版本：
- vllm_fl/spec_decode/dsv4_mtp.py
- vllm_fl/spec_decode/dsv4_mtp_utils.py
- vllm_fl/worker/model_runner.py
- vllm_fl/attention/ascend/dsa_v1.py

撤回内容包括：新合并草稿图、直接传递步间输入、首步SWA metadata扩展、接受数CPU缓存、新target/bonus argmax。合并图辅助模块从当前包移除，完整源码仍在865f007提交及备份中。

启动仍默认DSV4_BACKEND_OPT=0，保留第一批MTP优化、既有greedy融合、局部argmax、后续metadata快路径和融合输入准备。实际执行配置接近mtp2_1，不代表逐文件切回920c125，因为后端候选代码仍由关闭开关隔离。

压测脚本增加两项直接检查：
1. 每轮开始前检查/health，不可用即退出。
2. 每轮结果必须completed=256且failed=0，否则退出，不继续后续轮次。

正式参数、1K/4K各两轮、CC64、每轮64预热请求均未修改。历史日志和成绩未覆盖。

## 使用与验证边界

在宿主机 /data/tiankuan/hzj/model_test/dsv4-mtp-infyuki-20261015 执行原有启动和测试命令即可，无需新增环境变量。若旧服务残留，先自行bash stop_service.sh mtp2，再bash start_service.sh mtp2和bash test_vllm_bench.sh mtp2。

本轮没有启停服务、运行推理或压测。执行文件的版本一致性、AST、bash -n、git diff --check及manifest身份检查已完成；是否恢复正常仍需用户重启复测。

备份目录：/data/tiankuan/hzj/model_test/dsv4-mtp3-recovery-20261016/，包括撤回前源码、脚本/manifest、失败服务日志、启动记录、设备错误plog，以及恢复后的commit、补丁和manifest。
