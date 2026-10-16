# DSv4 Flash / FL Ascend MTP 工作交接与服务器迁移

交接日期：2026-10-16（北京时间）。本文同时供接手的新会话和新服务器使用。

## 1. 接手后首先要知道的状态

目标是在 **vllm-plugin-FL 的 v0.24 Ascend runtime 分支**上，为 DeepSeek-V4-Flash 提供正确、稳定且有实际加速的 MTP，并用统一协议比较 baseline、MTP1、MTP2、MTP3。

**当前不是已完成性能验收的版本。最新“融合 greedy 拒绝验证”候选已加入107的可编辑源码，下次启动MTP默认启用；按用户要求，没有运行该候选的任何测试，也没有启停服务。新会话应从验证这一候选继续，不能将它写成已经加速或精度通过。**

发布仓库：`https://github.com/InfYuki/vllm-plugin-FL`。

发布分支：`codex/dsv4-mtp-ascend-handoff-20261016`。

源码基线：`feature/v024-ascend-platform-runtime` 的 `61ae2938e02cd3f843425139e9ef5dfe1cdd0a9f`。本分支将107实际部署的9个源码文件提交，并补充交接文档与迁移脚本。107原有 dirty checkout、Git HEAD、editable安装没有因发布而切换。

旧工作区最新 manifest SHA256：
`68f973bcd7d4ca4bdac6a1eb92537e06763cb972a78de2da19fce03f9ccd9fff2`。

本仓库的 `examples/dsv4_mtp/source_snapshot.json` 记录这份旧部署的逐文件 SHA、基础版本和各层状态。**它不是新服务器最终 manifest**；新机器需运行准备工具，生成对应新 Git HEAD 和路径的 manifest。

## 2. 用户约定和授权边界

- 默认由用户自行启动模型服务、执行精度和压测。此前仅一次明确授权助手完成 profile，任务结束后该专用服务已停止。
- 最新指令是“融合 greedy 改完后不要测试，我自行测试”。本轮发布与写文档没有解除这一限制：没有运行候选模块导入/编译、CPU/NPU测试、服务启动、推理或压测。
- 105服务器是参考环境。用户明确要求不得改动105的文件和服务；先前比较只读完成。本轮没有连接105。
- 用户已决定不启用旧补丁03和06。不要在迁移中顺手打开。
- 一轮只新增一个性能候选，保留稳定对照，分清已添加、已应用、下次启动启用与已实测通过。
- 喜欢直接的 shell 启动/测试入口，避免把脚本变成复杂框架。最早的 `correctness` 三问仅是 smoke，不能代替精度或逐token一致性验证。
- SSH密码、GitHub token等凭据不写入本文、仓库或脚本。新会话需要连接机器时，由用户通过自己的SSH配置或私下方式提供凭据。

## 3. 原服务器、容器与路径

| 项目 | 原环境 |
|---|---|
| 主工作服务器 | 107；内网 `10.55.0.170`；原SSH经 `bastion.aiops.baai.ac.cn:2224` |
| 当前FL容器 | `fl-infyuki-v024-hzj` |
| 旧容器 | `fl-test-0923-hzj`，用于早期安装/对照，后续主要使用新容器 |
| FL宿主机源码 | `/data/tiankuan/hzj/vllm-plugin-FL-InfYuki` |
| FL容器内源码 | `/workspace/vllm-plugin-FL-InfYuki` |
| 宿主机实验目录 | `/data/tiankuan/hzj/model_test/dsv4-mtp-infyuki-20261015` |
| 容器实验目录 | `/workspace/model_test/dsv4-mtp-infyuki-20261015` |
| 模型权重/分词器 | `/public-flash/models/DeepSeek-V4-Flash-w8a8-mtp/` |
| 当前服务名、默认端口 | `dsv4`，107端口 `1112` |
| 原vLLM源码 | `/vllm-workspace/vllm`，属于容器文件系统 |
| AISBench解释器 | `/opt/aisbench-venv/bin/python` |
| AISBench入口 | `/opt/aisbench-venv/bin/ais_bench`；另有 `/usr/local/bin/ais_bench` 链接 |
| 对照服务 | 105内网 `10.55.0.166:11223`，服务名 `dsv4` |

宿主机 `/data/tiankuan/hzj` 同时挂到容器 `/workspace` 和同名绝对路径。脚本的宿主机入口会 `docker exec` 到容器，不要把宿主机默认Python当作推理Python。

容器配置实查：`--network=host`、`--ipc=host`、privileged，映射 `/dev/davinci0` 至15，以及 davinci_manager、devmm_svm、hisi_hdc；另挂载模型、驱动、固件、dcmi、npu-smi、hccn配置和 `/root/.cache`。设备编号0–15用于本实验；新服务器必须重新确认可用设备、芯片型号和驱动兼容性，不能假定任意Ascend服务器都等价。

## 4. 版本和安装依赖

2026-10-16实查；读取包元数据，没有初始化NPU：

| 项目 | 当前值 |
|---|---|
| 架构 | Linux aarch64 |
| 容器基础镜像标签 | `vllm-ascend:v0.24.0rc1-a3` |
| 镜像ID | `sha256:a9476d8994f6050aa59a46b6adb412e5288c692f9a58f47a0b2bc8e0ccc78201` |
| 镜像RepoDigests | 空；该ID是107本地镜像ID，不能当作远端可直接pull的仓库digest |
| Python | 3.12.13 |
| vLLM包 | 0.24.0+empty |
| vLLM Git HEAD | `ee0da84ab9e04ac7610e28580af62c365e898389`；本轮查看tracked状态为干净 |
| Torch | 2.10.0+cpu；Ascend由torch-npu提供，不要看到cpu后缀就擅自换成CUDA wheel |
| torch-npu | 2.10.0.post2 |
| CANN | 9.0.1，inner `V100R001C10SPC002B220` |
| Triton / Triton Ascend | 3.5.0 / 3.2.1 |
| Transformers | 5.13.0 |
| NumPy | 1.26.4 |
| FlagGems | 5.0.2；使用干净源码 `3e6528cf04f5f964a7b0fa6628de6f0410dbfd02` 的Python包构建 |
| FL原安装元数据版本 | 0.2.0+g61ae2938e；后续源码通过editable生效，不能靠这个版本字符串识别优化层 |
| AISBench | ais-bench-benchmark 3.1.20260630，独立venv复用系统site-packages |
| venv PyArrow | 19.0.1；默认推理Python没有PyArrow包 |

FL native构建使用 `VLLM_VENDOR=ascend`、`SOC_VERSION=ascend910_9391`、`MAX_JOBS=8`，并引用 `CATLASS_PATH=/vllm-workspace/vllm-ascend/csrc/third_party/catlass`，Catlass固定 `41bf90da655bba3c66d0acd7e00abe33960ecfd6`。核心安装命令为：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export VLLM_VENDOR=ascend VLLM_FL_VENDOR=ascend VLLM_PLUGINS=fl
export USE_FLAGGEMS=0 SOC_VERSION=ascend910_9391 MAX_JOBS=8
export CATLASS_PATH=/vllm-workspace/vllm-ascend/csrc/third_party/catlass
cd /workspace/vllm-plugin-FL-InfYuki
python -m pip install -v --no-build-isolation --no-deps -e .
```

这是已有兼容构建依赖下的命令，不是从任意空镜像一键成功的承诺。原来补齐了 setuptools/scm、scikit-build-core 0.11.0、pybind11、ninja、cmake 等构建依赖，并解决 CANN离线依赖。FlagGems用 `FLAGGEMS_BUILD_C_EXTENSIONS=OFF` 构建，`USE_FLAGGEMS=0`不等于可以删掉包依赖。

原安装记录在107 `model_test/fl-infyuki-v024-install-20261015/`：`install_in_container.sh`、`install.log`、`after.freeze.txt`、`verification.json`；AISBench安装记录在 `model_test/aisbench-install-20261015/`。这些大文件、wheelhouse没有上传Git。

原镜像/安装记录中存在一些预存的 `pip check` 告警（包括profiler与部分Web依赖版本约束）；当时安装前后没有新增告警，不代表整个环境零冲突。复现时先保留原推理环境，再在独立评测venv处理评测依赖，不要为消除所有镜像告警而批量升级推理依赖。

AISBench应继续隔离在 `/opt/aisbench-venv`，不要覆盖推理环境。它曾遇到 PyArrow26要求NumPy2、AISBench要求NumPy<2的冲突，已固定PyArrow19.0.1。还有 pyext 对 `inspect.getargspec` 的兼容修复。只安装最新AISBench和最新依赖无法保证复现；优先迁移现成环境及锁定清单。

运行时务必设置：

```bash
export VLLM_PLUGINS=fl VLLM_FL_VENDOR=ascend USE_FLAGGEMS=0
```

镜像另装有Ascend平台插件；未指定FL时可能出现多个平台插件冲突。当前启动脚本还清理历史库路径、旧 FL_DSV4_* 开关及来源不明的PYTHONPATH，避免把旧容器补丁误带进来。

## 5. 迁移方案：Git不是完整运行环境

### A. 优先迁移现有环境，减少依赖差异

以下均为**用户在迁移时自行执行**的示例，本轮助手没有执行docker commit/save/load，也没有创建新服务器容器。

旧107上，选足够空间的目录，保存容器文件系统和未进Git的依赖产物：

```bash
# 建议在自己的模型服务/评测空闲时制作；bind mount内容不会进入镜像。
docker commit --pause=false fl-infyuki-v024-hzj fl-dsv4-handoff:20261016
docker save -o fl-dsv4-handoff-20261016.tar fl-dsv4-handoff:20261016

tar -czf fl-native-20261016.tar.gz \
  -C /data/tiankuan/hzj/vllm-plugin-FL-InfYuki \
  vllm_fl/_C_ascend.cpython-312-aarch64-linux-gnu.so \
  vllm_fl/libvllm_fl_ascend_kernels.so vllm_fl/_cann_ops_custom

tar -czf dsv4-data-20261016.tar.gz \
  -C /data/tiankuan/hzj/model_test/dsv4-mtp-infyuki-20261015/aisbench data
```

还应单独复制：模型权重目录；整个旧实验目录（含结果、manifest、分层备份）；安装记录/wheelhouse；有分析需求时复制profile目录。FL源码本身从本Git分支克隆即可。`docker commit`不包含NPU进程内存、bind-mounted源码、模型、数据和挂载的root缓存。

新服务器加载镜像后，示例保持同样的容器内路径，宿主目录可由用户调整：

```bash
docker load -i fl-dsv4-handoff-20261016.tar
mkdir -p /data/tiankuan/hzj
git clone --branch codex/dsv4-mtp-ascend-handoff-20261016 \
  https://github.com/InfYuki/vllm-plugin-FL.git \
  /data/tiankuan/hzj/vllm-plugin-FL-InfYuki
tar -xzf fl-native-20261016.tar.gz \
  -C /data/tiankuan/hzj/vllm-plugin-FL-InfYuki

devices=()
for i in {0..15}; do devices+=(--device "/dev/davinci$i"); done
docker run -d --name fl-infyuki-v024-hzj \
  --privileged --network host --ipc host \
  "${devices[@]}" --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /data/tiankuan/hzj:/workspace \
  -v /data/tiankuan/hzj:/data/tiankuan/hzj \
  -v /public-flash/models:/public-flash/models \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/sbin:/usr/local/sbin:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /etc/hccn.conf:/etc/hccn.conf:ro \
  -e VLLM_PLUGINS=fl -e VLLM_FL_VENDOR=ascend -e VLLM_VENDOR=ascend -e USE_FLAGGEMS=0 \
  fl-dsv4-handoff:20261016 sleep infinity
```

这只是复刻原拓扑的命令模板。先确认目标主机路径、设备和网络文件存在；`--ipc=host`使用宿主IPC。原本的容器也挂载root缓存，若重编译需要缓存，可另挂自己的迁移缓存，不必覆盖新机器现有缓存。新机器的芯片、驱动、CANN或ABI不同，应重新编译native库，不能盲目复用.so。

镜像中FL的editable映射原本就是 `/workspace/vllm-plugin-FL-InfYuki`。保持路径并恢复兼容native库时可继续指向克隆代码；原FL版本字符串可能仍显示旧commit，这是包构建元数据，最终身份应以实际Git HEAD和manifest为准。若路径不同或映射缺失，按前述editable命令重装。编译和安装由用户在新环境执行。

### B. 纯重建环境

只有无法迁移镜像时，才以相同基础镜像、vLLM提交、CANN/Torch/torch-npu/Triton版本重建，补齐FlagGems和构建依赖，再编译FL。单纯克隆FL不能获得 `/vllm-workspace/vllm` 的匹配源码、AISBench venv或CANN OPP。没有将vLLM-Ascend 0.27.1/CANN9.1包迁入0.24环境；不要把它当升级依赖方案。

新服务器、设备资源和目录尚未由用户给出。本文准备了迁移方法，但没有在目标机器完成安装或启动验收。

## 6. 在新容器生成实验工作区

不要直接把旧manifest中的commit改成任意值。旧脚本按“旧HEAD + 未提交补丁”校验，新的Git提交是“已提交MTP源码”，其身份应重新生成。

在新容器中：

```bash
cd /workspace/vllm-plugin-FL-InfYuki
python examples/dsv4_mtp/prepare_workspace.py \
  --experiment-dir /workspace/model_test/dsv4-mtp-infyuki-20261015 \
  --container fl-infyuki-v024-hzj
```

该工具只用标准库与git，复制脚本/payload，验证源码文件和vLLM提交身份，生成匹配当前新Git HEAD的manifest；不安装包、不发API请求、不启动服务。它拒绝覆盖已经存在的实验目录。**新迁移工具尚未在新服务器运行。**若同时迁移了历史实验目录，将历史改放单独归档目录，或使用新的 `--experiment-dir`；不要覆盖历史日志。

然后在新宿主机恢复数据（对应上述默认路径）：

```bash
tar -xzf dsv4-data-20261016.tar.gz \
  -C /data/tiankuan/hzj/model_test/dsv4-mtp-infyuki-20261015/aisbench
```

GSM8K数据未上传Git。此前固定数据SHA如下，迁移时保持原文件，避免把输入变化当性能变化：

| 文件（aisbench/data下） | SHA256 |
|---|---|
| gsm8k/test.jsonl，1319题 | `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` |
| gsm8k/train.jsonl，7473题 | `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465` |
| perf/seed1024-n256-1024.jsonl | `e91cca4565066df027a884eb531a511b482bc3e30f2d5e78ec733e98b7bdcdcb` |
| perf/seed1024-n256-4096.jsonl | `3f730894646169a3e84896a473657f024848ce555ac8955963c92dfb95b0827e` |

perf的 `.meta.json` 也需一并迁移。若重新生成，必须保持同一个tokenizer与seed，并记录新的输入SHA；跨机器比较最好直接复用原数据。

## 7. 模型服务配置和各组含义

| 参数 | 默认值 |
|---|---|
| 模型 | DeepSeek-V4-Flash-w8a8-mtp，含独立MTP权重 |
| 并行 | TP4 × DP4，开启EP；使用0–15号设备 |
| max-model-len | 66560 |
| max-num-seqs / batched tokens | 64 / 16384 |
| 显存利用率 / block size | 0.90 / 32 |
| tokenizer / quantization | deepseek_v4 / ascend |
| chunked prefill / prefix cache | 开 / 关 |
| 调度 | 默认异步 |
| 目标图 | FULL_DECODE_ONLY |
| 草稿图 | **未适配，强制eager** |
| MTP2目标捕获尺寸 | 12、24、36……192，即LCM(3,TP4)的倍数 |

baseline不启用speculative decoding；MTP1/2/3分别每轮顺序提出1/2/3个草稿token，再由目标模型验证，不是三个不同权重。当前导出只有一层MTP，K>1是重复使用同一层；更长草稿会增加成本且后面位置接受率可能更低，K更大不保证更快。

当前适配约束：temperature=0、无logprobs、顺序method=mtp、K1–3、单导出MTP层、PP=PCP=DCP=1、不启用DSA CP、prefix caching、LoRA或KV拆分。不能当作通用随机采样MTP实现。

additional_config默认开启 fused_mc2、flashcomm1、shared_expert_dp、mlapo、balance_scheduling、cpu_binding和npugraph_ex；static_kernel及shared-expert多流重叠关闭。HCCL_OP_EXPANSION_MODE=AIV、TASK_QUEUE_ENABLE=1、OMP_NUM_THREADS=10、HCCL_BUFFSIZE=2048。

| 开关 | 默认/作用 |
|---|---|
| MTP_FUSED_GREEDY | 1；最新融合拒绝验证候选；**未测试** |
| MTP_FAST_METADATA | 1；后续单query SWA草稿元数据快路径 |
| GRAPH | 1；目标图；0为强制eager，不会因此获得草稿图支持 |
| ASYNC_SCHEDULING | auto，当前等同开启；可显式0/1做独立对照 |
| PORT | 1112 |
| CONTAINER | fl-infyuki-v024-hzj |

baseline仍有原分支预存的MTP hidden-state stash：FlashComm下会gather/copy，即使没启用MTP。这不是本轮新增代码，尚未做baseline门控优化；“当前baseline”不能自动等同9月最早完全无MTP桥接的源码。

## 8. 日常启动、停止与评测入口

用户在宿主机对应实验目录执行：

```bash
cd /data/tiankuan/hzj/model_test/dsv4-mtp-infyuki-20261015
bash start_service.sh baseline
# READY之后运行对应组的客户端。切换组前自行停止已运行的这一组：
bash stop_service.sh baseline
bash start_service.sh mtp2
```

服务日志为 `runs/<variant>/server.log`；`identity.json`记录PID及starttime，停止脚本只处理这份身份对应的进程及后代，不使用pkill。`launch_metadata.json`记录图、异步、metadata/fusion开关与manifest SHA。源码/manifest更新后，要由用户停止并重新启动旧服务再测，避免将旧进程成绩记到新代码。

快速冒烟：

```bash
bash run_client.sh mtp2 correctness
```

它只检查三条短请求能否返回，不是逐token正确性。正式精度及性能建议用AISBench：

```bash
bash aisbench/test_accuracy.sh mtp2
bash aisbench/test_performance.sh mtp2
```

GSM8K默认1319题、零样本CoT、并发8、max output8192、temperature0、top_p1、seed1024，正常EOS；去除reasoning标签内容后抽取数值答案。检查失败请求、完整题数及输出截断，再解释准确率；不是逐token等价性测试。`ACC_NUM=20`可由用户先小规模排障，正式对比用全量且统一输出上限。

性能固定为两轮1K输入、两轮4K输入，输出1024、CC64、每轮256请求、seed1024、ignore_eos=true、completions接口。`--num-warmups 0`意味着脚本没有额外独立预热请求；两轮不能自动解释为“预热轮+正式轮”，新候选首次Triton JIT和冷缓存需另作统一预热安排并记录。不要为某一组单独改变协议后直接比均值。

AISBench该版本配置 `retry=1`表示一次尝试；`retry=0`会零次请求，之前曾踩坑，不能再改回0。摘要只有在256请求完整、无失败、token数符合预期时才汇总。结果在 `aisbench/results/<variant>/<批次>/`，含配置、日志、launch证据、逐轮summary与总summary.json/csv。

在107测试105精度的独立入口是 `aisbench/test_accuracy_105.sh`，目标10.55.0.166:11223、模型dsv4；它不要求本机DSv4 PID身份。`--check-only`只GET models与核对数据，不发推理请求。该地址仅指原环境，换机后先确认网络，不要把跳板域名当HTTP推理端点。

## 9. 工作经过与当前代码结构

| 阶段 | 完成内容 | 状态/限制 |
|---|---|---|
| 10月14日 | 重建无额外性能补丁的baseline，补齐模型注册；曾加03/06供对照 | 早期历史，当前不启用03/06 |
| 10月15日 | 参考GLM5.2与Ascend，为DSv4加入MTP模型、权重路由、pre-HC hidden桥接、独立draft KV和greedy验证 | 功能适配；模型数学与GLM实现有差别，不能直接套用 |
| 10月15日 | 在InfYuki fork新容器editable安装；将MTP迁入新容器，配置AISBench | 主环境转为fl-infyuki-v024-hzj |
| 10月15日 | 目标图开启，MTP2图尺寸按12对齐；异步调度和接受数/CPU元数据校正 | 避免3-token验证与TP4 padding冲突、异步长度漂移 |
| 10月15日后续 | 移除真实草稿热路径每步positions/seq_lens.cpu，复用已有接受数事件 | 27项新增+70项历史CPU/静态回归记录；不是NPU验收 |
| 10月16日profile | MTP2、4K、CC64，预热后采16轮×16rank | 已完成采集分析并停止专用服务 |
| 10月16日metadata候选 | 后续单query SWA快路径，复用RoPE/slot/SAS，免无用QLI | 部署前153项CPU/静态记录；尚无同条件profile证明收益 |
| 10月16日对比105 | 读两个FL补丁，并核对105实际Ascend MTP2运行实现 | 实际后端和补丁说明不同；只读 |
| 最新greedy候选 | 新增单Triton内核融合接受前缀/拒绝/bonus/padding，复用cu_num_draft_tokens | 已部署，下次启动默认请求融合；**没有运行任何候选测试** |

主要源码：

- `vllm_fl/models/deepseek_v4_mtp.py`：MTP投影/归一化/decoder/hc_head、独立embedding/head、权重映射与FlashComm分片。
- `vllm_fl/spec_decode/dsv4_mtp.py`：自定义proposer、CPU元数据镜像、DSv4 greedy sampler及融合开关。
- `vllm_fl/spec_decode/dsv4_mtp_utils.py`：约束检查、量化别名、位置/拒绝/接受数算法、原PyTorch验证参考路径。
- `vllm_fl/spec_decode/dsv4_mtp_greedy.py`：最新融合验证内核，每请求一个program；尚未执行JIT。
- `vllm_fl/worker/model_runner.py`：drafter/sampler接入、读取pre-HC目标hidden、异步主机镜像校正。
- `vllm_fl/attention/ascend/dsa_v1.py`：新增后续草稿单query SWA builder。
- `vllm_fl/kv_cache/ascend/deepseek_v4_kv_cache.py`：draft KV及SWA层处理。
- `vllm_fl/dispatch/backends/vendor/ascend/patches/patch_deepseek_v4.py`：目标与MTP模型注册。

目标模型 `deepseek_v4_ascend.py` 的stash桥接是源分支已有内容，不在这次9个源文件差异中。融合候选不更改词表argmax、logits processor、bonus采样或接受规则设计，也没有顺带改proposer/metadata/graph/权重。内核归FL所有，不运行时导入vllm_ascend的执行路径。

## 10. 已有结果，不能混用测试口径

107 AISBench历史输出吞吐（tok/s；每点两轮平均，10月15日记录）：

| 组别/批次 | 1K输入+1K输出 | 4K输入+1K输出 |
|---|---:|---:|
| baseline | 1490.12 | 1348.28 |
| MTP1，后一批 | 1654.75 | 1478.56 |
| MTP2，后一批 | 1547.78 | 1414.64 |

baseline对应总吞吐是2980.24/6741.38。此前提及“1K约2900、4K约6500”应先核对是否是输入+输出的 total throughput；在本协议下total约等于output的2倍/5倍，不能与输出吞吐直接比。

105已有vllm bench随机输入日志：baseline输出约1452.92/1319.92，MTP2约1833.37/1561.71；1K两轮均值，4K一轮。客户端、输入文本、预热及源码批次不同，只能作线索，不能作为跨机严格A/B或归因某个补丁的收益。

已找到的GSM8K完整结果：107 MTP1两批分别96.6641%、96.7400%；从107评105的一批96.9674%，均1319题且summary valid=true。它们不是新融合候选的结果，也不是逐token正确性证明。未找到可归属于最新融合候选的baseline/MTP2/MTP3完整精度或性能验收。

结果批次名与源码关联应查当次config.json、manifest.json、launch_metadata.json。文件名中的服务器UTC时间仅是原始标识，不当作北京时间解释。

## 11. Profile得到的可靠线索

细节见 [profile结论](PROFILE_20261016_zh.md)。当时为融合候选和最新SWA快路径之前的版本，MTP2、目标图、草稿eager、异步、4K、CC64。

16rank均导出，窗口约1.33秒；CANN Computing平均59.03%、未重叠Communication6.94%、Free34.03%。主机preprocess/draft/sample scope约11.35/22.35/47.08ms/轮。代表rank的Free多数与draft/preprocess重合。

这些主机scope包含设备等待；不能把sample47ms全部当成采样算法耗时，也不能把目标forward主机scope约0.37ms当作目标模型设备执行时间。Free不是已经证明全部可以消除的空闲。

草稿LM-head相关all-gather累计约采集窗口0.72%，不足以支持“词表通信就是主瓶颈”的原假设；这也不是任意top-1优化收益的严格上限。大项DispatchFFNCombine融合了EP等工作，不能仅凭名字将全部时间归为FFN或通信。

原profile目录在107实验下 `profile-20261016/`：原始副本约2.147GB，离线分析和aggregate在其中。因worker daemon里torch_npu解析失败，最终在独立非daemon进程离线解析成功。原始trace没有上传Git，迁移需另复制。

## 12. 105补丁比较结论

详见 [对比报告](ASCEND_COMPARISON_20261016_zh.md)。指定目录的0001只是新增521行MTP模型与注册，0002只是开启K1，README明确补丁生成时未做性能测试。两者依赖已有hidden bridge。

105当前11223实际运行 `vllm-ascend-test-fanyc-1016` 内的原生Ascend平台、K2、async、target FULL_DECODE_ONLY、draft eager。不能说它是“打了两个FL补丁后的MTP1成绩”。

模型计算结构与FL基本一致；更值得看的是外围执行：Ascend提供融合greedy拒绝验证，并提前准备multi_steps_attn_metadata后运行草稿；FL原来用多段PyTorch验证，且在后续草稿步之间穿插位置更新和builder。两边都有接受数event同步，也都没有在当前配置启用草稿图。

baseline无条件stash/gather/copy是可单独优化的共同开销，但在原FL commit和105 target里已存在，不能归咎于最新MTP适配。quant_config传入/None及compile装饰器差异不等于已证明INT8或图加速，不能简单照抄补丁覆盖FL正确性适配。

## 13. 后续完整规划与优先级

### 第一阶段：迁移和重新建立可信对照

1. 新会话先读本文、source_snapshot和最新融合内核，不要立即增加新优化。
2. 由用户确认新服务器设备、驱动/CANN、镜像/依赖、模型和tokenizer、vLLM commit、FL commit、AISBench和输入数据一致。
3. 在新环境先用 `MTP_FUSED_GREEDY=0` 建立未融合对照。保留已部署SWA快路径开关状态及其他参数，不混入03/06。
4. 再开启融合验证：覆盖K1/K2/K3、零draft/混合draft、首拒绝/中途拒绝/全接受/bonus、边界长度及并发。先确认输出语义和安全，再做GSM8K与吞吐。
5. 统一预热后，用同一套固定AISBench输入做至少原约定的两轮1K/两轮4K，记录每轮、均值、失败、接受数与实际tokens/iteration；若波动明显再增加重复，不能只选最好一轮。

只有用户授权后，新会话才可自行运行上述步骤。本轮未代执行，也没有把原层153项结果挪作新内核测试结果。

### 第二阶段：验证最新融合候选

先对比 `MTP_FUSED_GREEDY=0/1`，保持metadata开关、K、图、调度、数据和预热一致。评估首次JIT成本与稳态收益，确认不是在NPU上误走CPU参考路径。融合要求Ascend Triton；不可用会显式报错，不静默回退。若无收益或正确性不合格，默认回到0并保留原因。

### 第三阶段：逐项推进剩余性能候选

1. 参考Ascend提前组织多步草稿元数据，减少step间Python发起、重复CPU张量操作及SAS/RoPE构建；保留接受数、KV slot和边界正确性。
2. 单独细分接受数等待、metadata CPU工作、设备执行和rank等待，再决定如何增加重叠。不能直接删event同步读取未完成数据。
3. 在地址和元数据生命周期稳定后，完整适配Ascend草稿图的capture/replay、DP协调、逐步buffer和fallback；不能仅将enforce_eager翻成false。
4. 检查小批量融合MC2与EP等待。必要时profile目标与草稿分别的kernel/通信，避免算子名归因。
5. draft-only局部top-1作为独立候选，优先级低于发起/元数据；全局enable_reduce_sample与当前spec组合约束不能直接绕过。
6. baseline的MTP hidden stash门控另立候选与baseline A/B，避免与MTP性能改动混算。

### 第四阶段：最终验收

同环境比较baseline/MTP1/MTP2/MTP3的正确性、GSM8K、1K/4K吞吐与延迟；必要时扩大context和负载。确定默认K及开关，而不是预设MTP2/3必优。输出每个候选的作用、依赖、应用状态、启用方式、实测证据和回退入口，最终才考虑上游PR或默认启用策略。

## 14. 回退和实验版本管理

最快关闭最新融合，不改代码（用户自行停当前组后启动）：

```bash
bash stop_service.sh mtp2
MTP_FUSED_GREEDY=0 bash start_service.sh mtp2
```

关闭上一层metadata快路径使用 `MTP_FAST_METADATA=0`。分别单变量验证，不要同时关闭两项后把变化归给其中一项。

旧107完整分层回退路径，必须从最新层倒序；先由用户停止自己的服务/评测：

1. `greedy-fusion-20261016/manage_greedy.py rollback`
2. `opt-20261016/manage_opt.py rollback`
3. `perf-fix-20261015/manage_fix.py rollback`
4. 再考虑历史 `rollback_async.py` 等更早层。

这些管理脚本按SHA保护，防止覆盖后续修改；不能乱序强行改哈希。分层备份仅在旧实验目录，Git中没有发布它们对应的全部机器绝对路径备份。新机器从已提交快照开始，先使用开关回退；不要假设旧107的管理脚本可以直接作用于新commit。进一步代码回退应独立Git分支/提交后重新生成一致manifest，并保留结果。

## 15. 常见问题和排查顺序

- `Connection refused`：端口未监听、服务未就绪、容器网络/地址错误或引擎已退出。它不是MTP算法错误的直接证据。先看server.log中最早的rank/worker错误，而不是最后 `Engine core initialization failed`。
- HTTP500：模型服务已监听但推理失败；看同一时段server.log中真正的异常。不要把client堆栈当根因。
- `Only one platform plugin...`：设置VLLM_PLUGINS=fl、VLLM_FL_VENDOR=ascend，核对安装路径。
- source/hash/manifest mismatch：说明当前服务、源码或实验记录不一致。不要直接删检查；确认目标环境，用户停止旧服务后按正确manifest重启。新clone必须用工作区准备工具适配已提交HEAD。
- MTP2图初始化失败或padding断言：核对3-query与TP4的12倍数档位，确认没有从参考环境抄入不兼容档位。
- 精度或接受率异常：先核对pre-HC target hidden、FlashComm token行分片、独立MTP embedding/head、量化别名、拒绝后的真实位置/长度、draft KV slot；不要先用更大K掩盖问题。
- 融合Triton报错：完整保存JIT/运行错误，关闭MTP_FUSED_GREEDY先恢复原验证路径；不要默认吞掉错误继续记“融合成绩”。
- 性能轮间差异：先查真实token数、失败/重试、冷JIT/缓存、前缀缓存开关、其他任务占卡、吞吐口径、实际源码批次，再讨论算法。当前latest优化没有验收结论。

## 16. 给新会话的起始提示

可以将下面这段直接发给新会话，并附本文：

> 我继续做DSv4 Flash在vllm-plugin-FL v0.24 Ascend上的MTP优化。请先完整读取仓库docs/dsv4_mtp/HANDOFF_zh.md、examples/dsv4_mtp/source_snapshot.json和当前MTP源码。原环境在107的fl-infyuki-v024-hzj，现准备迁到新服务器；具体新SSH/路径由我另给。最新融合greedy拒绝验证已提交但从未测试，先保持代码，不要再加新优化。03/06关闭，target FULL_DECODE_ONLY、draft eager、异步，metadata快路径和fusion各有独立开关。先核对迁移环境和测试方案，服务启动/测试默认由我自行执行，未经本次授权不要代启动。不要改105参考机的文件或服务。完成测试后再按单变量A/B决定保留、回退或下一候选。

本文及引用文档记录的是截至交接的状态。新会话应把迁移后commit、环境、命令、数据SHA和新结果继续追加到独立实验记录，不能覆盖旧历史。
