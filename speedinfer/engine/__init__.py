"""SpeedInfer Inference Engine Abstraction & Process Manager.

Provides out-of-process vLLM supervisor, model warm-up probing,
exponential backoff recovery, and thread-safe dynamic model hot-registry.
"""

from speedinfer.engine.registry import (
    BackendHealth,
    BackendWorker,
    ModelEntry,
    ModelPricing,
    ModelRegistry,
)
from speedinfer.engine.vllm_runner import (
    VLLMConfig,
    VLLMProcessCrashError,
    VLLMProcessSupervisor,
    VLLMRunner,
    VLLMRunnerError,
    VLLMStartupTimeoutError,
    VLLMSupervisor,
    VLLMWarmupError,
    VLLMWorkerConfig,
    WorkerStatus,
    build_cli_args,
    build_vllm_cmd,
    calculate_backoff,
)

__all__ = [
    "BackendHealth",
    "BackendWorker",
    "ModelEntry",
    "ModelPricing",
    "ModelRegistry",
    "VLLMConfig",
    "VLLMProcessCrashError",
    "VLLMProcessSupervisor",
    "VLLMRunner",
    "VLLMRunnerError",
    "VLLMStartupTimeoutError",
    "VLLMSupervisor",
    "VLLMWarmupError",
    "VLLMWorkerConfig",
    "WorkerStatus",
    "build_cli_args",
    "build_vllm_cmd",
    "calculate_backoff",
]
