# 193 上游同步记录（2026-10-20）

- 仓库：/workspace/vllm-plugin-FL-InfYuki（宿主机 /data/tiankuan/hzj/vllm-plugin-FL-InfYuki）。
- 部署原分支：codex/dsv4-max-context-dp-fix-20261020，提交 1404e5e。
- 原 fork 分支：feature/v024-ascend-platform-runtime，提交 61ae2938e02cd3f843425139e9ef5dfe1cdd0a9f。
- 本次上游：xiemingda-1002/vllm-plugin-FL 的同名分支，50ac04eeeb164b948d8c6c40f6347a599b825420。
- 上游有重写历史，实际 Git merge-base 为 b5b7cae。为避免把同一批 Ascend runtime 文件误判为双方新增，本次使用 61ae293 的树作为显式三方基准；合并提交仍保留当前部署 HEAD 和上游 HEAD 两个父提交。
- runner / worker / forward-context / MoE 的共同修改按三方内容合并。保留所有 DSv4 MTP、eager draft、融合 greedy、批量/DP 协调、接受数缓存和 max-context 修复。
- 上游删除 compiler_interface.py、改用 FXFusionAdaptor，并拒绝 npugraph/static kernel。现有启动脚本明确指定 enable_npugraph_ex=True，因此保留旧编译模块，通过 FXFusionAdaptor 对显式 opt-in 委派；默认 False 仍走上游 FX/eager。静态编译仍默认关闭，启用时必须同时启用 npugraph。
- 原部署提交和分支保留；backup/dsv4-before-upstream-sync-20261020 指向合并前 HEAD。
- Git bundle 和合并前状态：/workspace/model_test/fl-upstream-sync-20261020/。
- 原有未跟踪的 build_out、共享库和 _cann_ops_custom 原样保留，未提交。
- 按用户要求未运行单测、模块导入/编译、模型启动、推理或性能测试；未重装包或重启现有服务。仅进行 Git 冲突、差异和空白检查。
- 这些检查不代表运行正确性、精度或性能已通过；用户自行测试。
