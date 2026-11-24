# Copyright (c) 2026 BAAI. All rights reserved.
"""Startup integration with the native vLLM-Ascend runtime."""

import logging
from functools import cache
from importlib.metadata import entry_points

logger = logging.getLogger("vllm.plugins.fl")

# General plugins loaded in the reference A3 image alongside vllm-ascend.
_COMPANION_DISTRIBUTIONS = {
    "msserviceprofiler": "ms-service-profiler",
    "arctic_inference": "arctic-inference",
    "lora_filesystem_resolver": "vllm",
    "lora_hf_hub_resolver": "vllm",
}
_REQUIRED_ASCEND_HOOKS = {
    "ascend_kv_connector",
    "ascend_model",
    "ascend_model_loader",
    "ascend_service_profiling",
}


@cache
def register_platform() -> str:
    from vllm_ascend import register

    platform = register()
    logger.info("FL platform entry point selected %s via vllm_ascend.register", platform)
    return platform


@cache
def register_general_plugins() -> None:
    from vllm import envs
    from vllm.plugins import DEFAULT_PLUGINS_GROUP

    selected = envs.VLLM_PLUGINS
    if selected is None:
        # With no allowlist vLLM itself calls every discovered general plugin.
        return

    plugins = {}
    ascend_hooks = set()
    for entry in entry_points(group=DEFAULT_PLUGINS_GROUP):
        provider = entry.dist.name.lower().replace("_", "-")
        if provider == "vllm-ascend":
            ascend_hooks.add(entry.name)
        elif _COMPANION_DISTRIBUTIONS.get(entry.name) != provider:
            continue
        if entry.name in selected:
            # Explicitly selected entries already have a callback in vLLM's loader.
            continue
        try:
            plugins[entry.name] = entry.load()
        except Exception:
            if provider == "vllm-ascend":
                raise
            logger.exception("Failed to load optional general plugin %s", entry.name)

    missing = _REQUIRED_ASCEND_HOOKS - ascend_hooks
    if missing:
        raise RuntimeError(f"Missing vllm-ascend general entry points: {sorted(missing)}")

    # Match vLLM's two passes. Imports can have effects before callbacks run.
    for callback in plugins.values():
        callback()
    from .cold_static_capture import install_worker_hook

    install_worker_hook()
    logger.info("FL registered native runtime general hooks in order: %s", ", ".join(plugins))
