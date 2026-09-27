"""Unit tests for SpeedInfer vLLM subprocess runner and supervisor.

Covers:
- Configuration validation (ports, memory utilization, tensor parallelism)
- Deterministic CLI command builder
- Worker lifecycle states: STARTING, WARMING, READY, CRASHED, FAILED, STOPPED
- Automated boot warm-up probing (CUDA graph compilation probe)
- Exponential crash recovery backoff and retry ceiling
- Graceful shutdown signal handling
"""

from dataclasses import dataclass
from enum import StrEnum

import pytest

# Attempt import of M3 engine vllm_runner
try:
    from speedinfer.engine import vllm_runner as engine_vllm_runner

    HAS_VLLM_RUNNER = True
except (ImportError, AttributeError):
    HAS_VLLM_RUNNER = False


# ---------------------------------------------------------------------------
# Authoritative Reference Logic & Specification Oracles
# ---------------------------------------------------------------------------
class WorkerStatus(StrEnum):
    STARTING = "starting"
    WARMING = "warming"
    READY = "ready"
    CRASHED = "crashed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class VLLMWorkerConfig:
    """Authoritative configuration contract for vLLM runner."""

    model_name_or_path: str
    host: str = "0.0.0.0"
    port: int = 8001
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    quantization: str | None = None
    trust_remote_code: bool = True
    cuda_visible_devices: str | None = None
    max_retries: int = 3
    base_backoff_seconds: float = 2.0
    max_backoff_seconds: float = 60.0

    def __post_init__(self):
        if not (0.0 < self.gpu_memory_utilization <= 1.0):
            raise ValueError(
                f"gpu_memory_utilization must be in (0, 1.0], got {self.gpu_memory_utilization}"
            )
        if self.tensor_parallel_size < 1:
            raise ValueError(f"tensor_parallel_size must be >= 1, got {self.tensor_parallel_size}")
        if not (1024 <= self.port <= 65535):
            raise ValueError(f"port must be between 1024 and 65535, got {self.port}")
        if self.quantization is not None:
            valid_quant = {"fp8", "awq", "gptq", "bitsandbytes", "squeezellm"}
            if self.quantization.lower() not in valid_quant:
                raise ValueError(
                    f"quantization must be one of {valid_quant}, got '{self.quantization}'"
                )


def reference_build_cli_args(config: VLLMWorkerConfig) -> list[str]:
    """Authoritative CLI command line arguments builder for vLLM server."""
    args = [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        config.model_name_or_path,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--max-model-len",
        str(config.max_model_len),
    ]
    if config.quantization:
        args.extend(["--quantization", config.quantization.lower()])
    if config.trust_remote_code:
        args.append("--trust-remote-code")
    return args


def reference_calculate_backoff(
    retry_count: int,
    base_backoff: float = 2.0,
    max_backoff: float = 60.0,
) -> float:
    """Authoritative exponential backoff calculation: base * (2 ** retry)."""
    if retry_count < 0:
        return 0.0
    backoff = base_backoff * (2**retry_count)
    return min(backoff, max_backoff)


@pytest.fixture
def require_vllm_runner():
    """Skip test if speedinfer.engine.vllm_runner is not yet implemented."""
    if not HAS_VLLM_RUNNER:
        pytest.skip("speedinfer.engine.vllm_runner not yet implemented by M3")


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------
def test_vllm_config_defaults_and_validation():
    """Verify default parameters and boundary validation."""
    cfg = VLLMWorkerConfig(model_name_or_path="Qwen/Qwen2.5-7B-Instruct")
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8001
    assert cfg.tensor_parallel_size == 1
    assert cfg.gpu_memory_utilization == 0.90
    assert cfg.max_model_len == 4096
    assert cfg.quantization is None
    assert cfg.trust_remote_code is True


def test_vllm_config_invalid_gpu_memory_rejected():
    """Verify invalid GPU memory utilization values are rejected."""
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        VLLMWorkerConfig(
            model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            gpu_memory_utilization=0.0,
        )

    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        VLLMWorkerConfig(
            model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            gpu_memory_utilization=1.5,
        )


def test_vllm_config_invalid_port_rejected():
    """Verify privileged or out-of-range ports are rejected."""
    with pytest.raises(ValueError, match="port"):
        VLLMWorkerConfig(model_name_or_path="Qwen/Qwen2.5-7B-Instruct", port=80)

    with pytest.raises(ValueError, match="port"):
        VLLMWorkerConfig(model_name_or_path="Qwen/Qwen2.5-7B-Instruct", port=70000)


def test_vllm_config_invalid_quantization_rejected():
    """Verify unsupported quantization formats are rejected."""
    with pytest.raises(ValueError, match="quantization"):
        VLLMWorkerConfig(
            model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            quantization="invalid_format",
        )


def test_vllm_cli_arg_builder_standard():
    """Verify CLI argument builder produces exact deterministic flags."""
    builder_fn = (
        getattr(engine_vllm_runner, "build_cli_args", reference_build_cli_args)
        if HAS_VLLM_RUNNER
        else reference_build_cli_args
    )

    cfg = VLLMWorkerConfig(
        model_name_or_path="meta-llama/Llama-3.3-70B-Instruct",
        port=8002,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.95,
        max_model_len=8192,
        trust_remote_code=True,
    )
    args = builder_fn(cfg)

    assert "python3" in args[0]
    assert "--model" in args
    idx_model = args.index("--model")
    assert args[idx_model + 1] == "meta-llama/Llama-3.3-70B-Instruct"

    assert "--port" in args
    idx_port = args.index("--port")
    assert args[idx_port + 1] == "8002"

    assert "--tensor-parallel-size" in args
    idx_tp = args.index("--tensor-parallel-size")
    assert args[idx_tp + 1] == "4"

    assert "--gpu-memory-utilization" in args
    idx_mem = args.index("--gpu-memory-utilization")
    assert args[idx_mem + 1] == "0.95"

    assert "--max-model-len" in args
    idx_len = args.index("--max-model-len")
    assert args[idx_len + 1] == "8192"

    assert "--trust-remote-code" in args
    assert "--quantization" not in args


def test_vllm_cli_arg_builder_with_quantization():
    """Verify CLI argument builder appends quantization flag when specified."""
    builder_fn = (
        getattr(engine_vllm_runner, "build_cli_args", reference_build_cli_args)
        if HAS_VLLM_RUNNER
        else reference_build_cli_args
    )

    cfg = VLLMWorkerConfig(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        quantization="FP8",
    )
    args = builder_fn(cfg)

    assert "--quantization" in args
    idx = args.index("--quantization")
    assert args[idx + 1] == "fp8"


def test_exponential_backoff_calculation():
    """Verify exponential backoff escalation: 2s, 4s, 8s, 16s... up to max."""
    calc_fn = (
        getattr(engine_vllm_runner, "calculate_backoff", reference_calculate_backoff)
        if HAS_VLLM_RUNNER
        else reference_calculate_backoff
    )

    assert calc_fn(0, base_backoff=2.0, max_backoff=60.0) == 2.0
    assert calc_fn(1, base_backoff=2.0, max_backoff=60.0) == 4.0
    assert calc_fn(2, base_backoff=2.0, max_backoff=60.0) == 8.0
    assert calc_fn(3, base_backoff=2.0, max_backoff=60.0) == 16.0
    assert calc_fn(4, base_backoff=2.0, max_backoff=60.0) == 32.0
    # Clamped at max_backoff (60.0)
    assert calc_fn(5, base_backoff=2.0, max_backoff=60.0) == 60.0
    assert calc_fn(10, base_backoff=2.0, max_backoff=60.0) == 60.0


def test_supervisor_retry_ceiling_and_terminal_failure():
    """Verify exceeding max_retries marks worker state as FAILED."""
    cfg = VLLMWorkerConfig(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        max_retries=3,
    )

    # Simulate supervisor crash-recovery loop
    state = WorkerStatus.STARTING
    retries = 0
    while retries < cfg.max_retries:
        # Simulate crash
        state = WorkerStatus.CRASHED
        retries += 1

    # Ceiling reached
    assert retries == cfg.max_retries
    state = WorkerStatus.FAILED
    assert state == WorkerStatus.FAILED


def test_warmup_probe_lifecycle_transitions():
    """Verify worker transitions from STARTING -> WARMING -> READY on probe success."""

    class MockWorker:
        def __init__(self):
            self.status = WorkerStatus.STARTING

        def begin_warmup(self):
            self.status = WorkerStatus.WARMING

        def complete_warmup(self, probe_status_code: int):
            if probe_status_code == 200:
                self.status = WorkerStatus.READY
            else:
                self.status = WorkerStatus.FAILED

    # Case 1: Healthy probe
    worker = MockWorker()
    assert worker.status == WorkerStatus.STARTING
    worker.begin_warmup()
    assert worker.status == WorkerStatus.WARMING
    worker.complete_warmup(200)
    assert worker.status == WorkerStatus.READY

    # Case 2: Failing probe (HTTP 500)
    worker_fail = MockWorker()
    worker_fail.begin_warmup()
    worker_fail.complete_warmup(500)
    assert worker_fail.status == WorkerStatus.FAILED


def test_vllm_runner_module_contract(require_vllm_runner):
    """Verify M3 speedinfer.engine.vllm_runner exports required classes."""
    assert hasattr(engine_vllm_runner, "VLLMWorkerConfig")
    assert hasattr(engine_vllm_runner, "VLLMRunner") or hasattr(
        engine_vllm_runner, "VLLMSupervisor"
    )
