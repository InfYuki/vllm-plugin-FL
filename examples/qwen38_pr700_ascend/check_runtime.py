"""Verify the actual platform, loadable model, and serve configuration."""
import inspect
import json
from importlib import metadata
from pathlib import Path
import sys

from vllm import AsyncEngineArgs, envs
from vllm.entrypoints.cli.serve import ServeSubcommand
from vllm.model_executor.models import ModelRegistry
from vllm.platforms import current_platform
from vllm.plugins import load_general_plugins
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
import vllm_fl

assert envs.VLLM_PLUGINS == ["fl"], envs.VLLM_PLUGINS
assert Path(vllm_fl.__file__).resolve() == Path("/workspace/vllm-plugin-FL/vllm_fl/__init__.py")
editable = json.loads(metadata.distribution("vllm-plugin-fl").read_text("direct_url.json"))
assert editable["dir_info"]["editable"] and editable["url"] == "file:///workspace/vllm-plugin-FL", editable
assert type(current_platform).__module__ == "vllm_ascend.platform", type(current_platform)
load_general_plugins()

command = ServeSubcommand()
parser = FlexibleArgumentParser()
command.subparser_init(parser.add_subparsers())
args = parser.parse_args(["serve", *sys.argv[1:]])
command.validate(args)
args.model = args.model_tag
config = AsyncEngineArgs.from_cli_args(args).create_engine_config(usage_context=UsageContext.OPENAI_API_SERVER)
cls, arch = ModelRegistry.resolve_model_cls(config.model_config.architectures, config.model_config)
pc, sc, mc = config.parallel_config, config.scheduler_config, config.model_config
assert cls.__module__.startswith("vllm_ascend.models.") and arch == "Qwen4ExpForConditionalGeneration", (cls, arch)
assert pc.worker_cls.startswith("vllm_ascend."), pc.worker_cls
assert (pc.tensor_parallel_size, pc.data_parallel_size, pc.enable_expert_parallel) == (8, 2, True)
assert (sc.max_num_seqs, sc.max_num_batched_tokens, mc.max_model_len) == (32, 4096, 131072)
assert config.use_v2_model_runner and config.speculative_config is None
assert str(mc.dtype) == "torch.bfloat16" and mc.quantization is None
assert config.compilation_config.cudagraph_mode.name == "FULL_DECODE_ONLY", config.compilation_config.cudagraph_mode
report = {
    "allowlist": envs.VLLM_PLUGINS, "fl_source": vllm_fl.__file__, "editable": editable,
    "platform": f"{type(current_platform).__module__}.{type(current_platform).__name__}",
    "worker": pc.worker_cls, "model": f"{cls.__module__}.{cls.__name__}", "model_source": inspect.getfile(cls),
    "versions": {package: metadata.version(package) for package in ("vllm-plugin-fl","vllm","vllm-ascend","torch-npu")},
    "v2": config.use_v2_model_runner, "dtype": str(mc.dtype), "mtp": config.speculative_config,
    "tp": pc.tensor_parallel_size, "dp": pc.data_parallel_size, "ep": pc.enable_expert_parallel,
    "sequences_per_dp": sc.max_num_seqs, "max_num_batched_tokens": sc.max_num_batched_tokens,
    "max_model_len": mc.max_model_len, "graph": config.compilation_config.cudagraph_mode.name,
}
print("PR700_RUNTIME " + json.dumps(report, ensure_ascii=False), flush=True)
