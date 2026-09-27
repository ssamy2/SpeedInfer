"""vLLM subprocess runner and process supervisor.

Provides out-of-process vLLM instance lifecycle management:
- Validated configuration model (VLLMConfig / VLLMWorkerConfig)
- Deterministic CLI command argument generation
- Process supervision with automated boot warm-up probing
- Exponential crash recovery backoff and retry limits
- Graceful process group termination (SIGTERM -> SIGKILL)
- Health check monitoring loop
"""

import asyncio
import logging
import os
import signal
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)


class WorkerStatus(StrEnum):
    """Lifecycle status enumeration for vLLM runner instances."""

    STARTING = "starting"
    WARMING = "warming"
    READY = "ready"
    CRASHED = "crashed"
    DEGRADED = "degraded"
    FAILED = "failed"
    STOPPED = "stopped"


class VLLMRunnerError(Exception):
    """Base exception for vLLM runner errors."""


class VLLMStartupTimeoutError(VLLMRunnerError):
    """Raised when vLLM worker fails to become ready within startup timeout."""


class VLLMProcessCrashError(VLLMRunnerError):
    """Raised when vLLM subprocess crashes unexpectedly."""


class VLLMWarmupError(VLLMRunnerError):
    """Raised when boot warm-up probe fails."""


@dataclass
class VLLMConfig:
    """Configuration contract for launching and supervising a vLLM worker process."""

    model_name_or_path: str = ""
    model_path: str = ""
    model: str = ""
    host: str = "0.0.0.0"
    port: int = 8001
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    quantization: str | None = None
    dtype: str = "auto"
    trust_remote_code: bool = True
    cuda_visible_devices: str | None = None
    served_model_name: str | None = None
    chat_template: str | None = None
    enforce_eager: bool = False
    max_num_seqs: int = 256
    extra_args: list[str] = field(default_factory=list)
    max_retries: int = 3
    base_backoff_seconds: float = 2.0
    max_backoff_seconds: float = 60.0
    startup_timeout_seconds: float = 300.0
    shutdown_timeout_seconds: float = 15.0
    warmup_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        """Validate configuration parameters and normalize model identifiers."""
        resolved = self.model_name_or_path or self.model_path or self.model
        if not resolved:
            raise ValueError("model_name_or_path (or model_path / model) must be provided")
        self.model_name_or_path = resolved
        self.model_path = resolved
        self.model = resolved

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


# Alias for test suite compatibility
VLLMWorkerConfig = VLLMConfig


def build_cli_args(config: Any) -> list[str]:
    """Build deterministic CLI command line arguments for launching vLLM server."""
    python_bin = sys.executable if "python3" in sys.executable else "python3"
    model_name = (
        getattr(config, "model_name_or_path", None)
        or getattr(config, "model_path", None)
        or getattr(config, "model", "")
    )
    args = [
        python_bin,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_name,
        "--host",
        getattr(config, "host", "0.0.0.0"),
        "--port",
        str(getattr(config, "port", 8001)),
        "--tensor-parallel-size",
        str(getattr(config, "tensor_parallel_size", 1)),
        "--gpu-memory-utilization",
        str(getattr(config, "gpu_memory_utilization", 0.90)),
        "--max-model-len",
        str(getattr(config, "max_model_len", 4096)),
    ]
    quantization = getattr(config, "quantization", None)
    if quantization:
        args.extend(["--quantization", quantization.lower()])
    if getattr(config, "trust_remote_code", True):
        args.append("--trust-remote-code")
    dtype = getattr(config, "dtype", "auto")
    if dtype != "auto":
        args.extend(["--dtype", dtype])
    served_model_name = getattr(config, "served_model_name", None)
    if served_model_name:
        args.extend(["--served-model-name", served_model_name])
    chat_template = getattr(config, "chat_template", None)
    if chat_template:
        args.extend(["--chat-template", chat_template])
    if getattr(config, "enforce_eager", False):
        args.append("--enforce-eager")
    extra_args = getattr(config, "extra_args", None)
    if extra_args:
        args.extend(extra_args)
    return args


# Alias for specification and dispatch requirement
build_vllm_cmd = build_cli_args


def calculate_backoff(
    retry_count: int,
    base_backoff: float = 2.0,
    max_backoff: float = 60.0,
) -> float:
    """Calculate exponential restart backoff: base * (2 ** retry_count), clamped to max."""
    if retry_count < 0:
        return 0.0
    backoff = base_backoff * (2.0**retry_count)
    return min(backoff, max_backoff)


class VLLMProcessSupervisor:
    """Supervises the execution, warm-up probing, health, and recovery of a vLLM worker."""

    def __init__(
        self,
        config: VLLMConfig,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize supervisor with worker configuration."""
        self.config = config
        self.status = WorkerStatus.STOPPED
        self.process: asyncio.subprocess.Process | None = None
        self.pid: int | None = None
        self.pgid: int | None = None
        self.retry_count: int = 0
        self.crash_count: int = 0
        self._stop_requested: bool = False
        self._monitor_task: asyncio.Task[None] | None = None
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_buffer: list[str] = []
        self._http_client = http_client
        self._logger = structlog.get_logger(__name__)

    @property
    def endpoint_url(self) -> str:
        """Return base HTTP URL for this worker instance."""
        host = "127.0.0.1" if self.config.host in {"0.0.0.0", ""} else self.config.host
        return f"http://{host}:{self.config.port}"

    @property
    def is_alive(self) -> bool:
        """Return True if the subprocess is currently executing."""
        return self.process is not None and self.process.returncode is None

    @property
    def is_ready(self) -> bool:
        """Return True if worker is healthy and ready to serve inference requests."""
        return self.status == WorkerStatus.READY and self.is_alive

    def begin_warmup(self) -> None:
        """Transition worker status to WARMING."""
        self.status = WorkerStatus.WARMING
        self._logger.info(
            "worker_warmup_started",
            model=self.config.model_name_or_path,
            port=self.config.port,
        )

    def complete_warmup(self, probe_status_code: int) -> None:
        """Transition worker status based on warm-up probe HTTP status code."""
        if probe_status_code == 200:
            self.status = WorkerStatus.READY
            self._logger.info(
                "worker_warmup_succeeded",
                model=self.config.model_name_or_path,
                port=self.config.port,
            )
        else:
            self.status = WorkerStatus.FAILED
            self._logger.error(
                "worker_warmup_failed",
                model=self.config.model_name_or_path,
                port=self.config.port,
                status_code=probe_status_code,
            )

    async def _read_stream(
        self,
        stream: asyncio.StreamReader | None,
        stream_name: str,
    ) -> None:
        """Read lines from stdout/stderr, buffering stderr for diagnostic crash reporting."""
        if stream is None:
            return
        try:
            while not stream.at_eof():
                line = await stream.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip()
                if stream_name == "stderr":
                    self._stderr_buffer.append(decoded)
                    if len(self._stderr_buffer) > 200:
                        self._stderr_buffer.pop(0)
                if self._logger.isEnabledFor(logging.DEBUG):
                    self._logger.debug(
                        "vllm_process_output",
                        pid=self.pid,
                        stream=stream_name,
                        output=decoded,
                    )
        except (asyncio.CancelledError, OSError) as exc:
            self._logger.debug("stream_reader_interrupted", stream=stream_name, error=str(exc))

    async def _probe_warmup(self, client: httpx.AsyncClient) -> bool:
        """Issue dummy completion to pre-allocate CUDA graphs and KV cache."""
        self.begin_warmup()
        model_name = self.config.served_model_name or self.config.model_name_or_path
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "temperature": 0.0,
        }
        url = f"{self.endpoint_url}/v1/chat/completions"
        try:
            resp = await client.post(
                url,
                json=payload,
                timeout=self.config.warmup_timeout_seconds,
            )
            self.complete_warmup(resp.status_code)
            return resp.status_code == 200
        except (httpx.HTTPError, OSError) as exc:
            self._logger.warning("warmup_probe_network_error", error=str(exc))
            self.complete_warmup(500)
            return False

    async def health_check(self, client: httpx.AsyncClient | None = None) -> bool:
        """Check if worker returns 200 OK from its /health endpoint."""
        url = f"{self.endpoint_url}/health"
        close_client = False
        if client is None:
            if self._http_client is not None:
                client = self._http_client
            else:
                client = httpx.AsyncClient(timeout=5.0)
                close_client = True
        try:
            resp = await client.get(url)
            return resp.status_code == 200
        except (httpx.HTTPError, OSError):
            return False
        finally:
            if close_client:
                await client.aclose()

    async def wait_until_ready(
        self,
        timeout_seconds: float | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Poll /health endpoint until worker is responsive or timeout occurs."""
        timeout = timeout_seconds or self.config.startup_timeout_seconds
        start_time = asyncio.get_running_loop().time()
        close_client = False
        if client is None:
            if self._http_client is not None:
                client = self._http_client
            else:
                client = httpx.AsyncClient(timeout=5.0)
                close_client = True

        try:
            while True:
                # Check if process crashed prematurely
                if self.process is not None and self.process.returncode is not None:
                    stderr_snippet = "\n".join(self._stderr_buffer[-30:])
                    raise VLLMProcessCrashError(
                        f"vLLM process crashed with return code {self.process.returncode}: "
                        f"{stderr_snippet}"
                    )

                if await self.health_check(client=client):
                    return

                elapsed = asyncio.get_running_loop().time() - start_time
                if elapsed >= timeout:
                    raise VLLMStartupTimeoutError(
                        f"vLLM worker failed to become ready within {timeout} seconds"
                    )

                await asyncio.sleep(1.0)
        finally:
            if close_client:
                await client.aclose()

    async def start(self) -> None:
        """Launch the vLLM subprocess, execute warm-up probe, and begin monitoring."""
        self._stop_requested = False
        self.status = WorkerStatus.STARTING
        self._stderr_buffer.clear()

        cmd = build_cli_args(self.config)
        env = os.environ.copy()
        if self.config.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.config.cuda_visible_devices

        self._logger.info(
            "starting_vllm_process",
            cmd=cmd,
            port=self.config.port,
            model=self.config.model_name_or_path,
        )

        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
        except OSError as exc:
            self.status = WorkerStatus.FAILED
            raise VLLMProcessCrashError(f"Failed to launch vLLM executable: {exc}") from exc

        self.pid = self.process.pid
        if hasattr(os, "getpgid") and self.pid is not None:
            try:
                self.pgid = os.getpgid(self.pid)
            except OSError:
                self.pgid = None

        # Spawn background stream readers
        self._stdout_task = asyncio.create_task(self._read_stream(self.process.stdout, "stdout"))
        self._stderr_task = asyncio.create_task(self._read_stream(self.process.stderr, "stderr"))

        # Wait for healthcheck readiness
        try:
            await self.wait_until_ready()
        except Exception:
            await self.stop()
            raise

        # Execute automated warm-up probe
        warmup_client = self._http_client or httpx.AsyncClient()
        close_warmup_client = self._http_client is None
        try:
            probe_ok = await self._probe_warmup(warmup_client)
            if not probe_ok:
                await self.stop()
                raise VLLMWarmupError(
                    f"Automated warm-up probe failed for model {self.config.model_name_or_path}"
                )
        finally:
            if close_warmup_client:
                await warmup_client.aclose()

        # Reset retry counter on full healthy initialization
        self.retry_count = 0
        self._monitor_task = asyncio.create_task(self._monitor_process())

    async def _monitor_process(self) -> None:
        """Background loop to monitor subprocess lifecycle and execute exponential backoff."""
        if self.process is None:
            return

        try:
            returncode = await self.process.wait()
            if self._stop_requested:
                return

            self.crash_count += 1
            self.status = WorkerStatus.CRASHED
            self._logger.warning(
                "vllm_process_unexpected_exit",
                pid=self.pid,
                returncode=returncode,
                crash_count=self.crash_count,
            )

            # Check retry ceiling
            if self.retry_count < self.config.max_retries:
                delay = calculate_backoff(
                    self.retry_count,
                    self.config.base_backoff_seconds,
                    self.config.max_backoff_seconds,
                )
                self.retry_count += 1
                self._logger.info(
                    "scheduling_worker_restart",
                    retry_count=self.retry_count,
                    delay_seconds=delay,
                )
                await asyncio.sleep(delay)
                try:
                    await self.start()
                except Exception as exc:
                    self._logger.error("restart_attempt_failed", error=str(exc))
                    if self.retry_count >= self.config.max_retries:
                        self.status = WorkerStatus.FAILED
            else:
                self.status = WorkerStatus.FAILED
                self._logger.critical(
                    "worker_max_retries_exhausted",
                    model=self.config.model_name_or_path,
                    retries=self.retry_count,
                )
        except asyncio.CancelledError:
            self._logger.debug("monitor_task_cancelled")

    async def stop(self) -> None:
        """Gracefully terminate worker process group with SIGTERM, escalating to SIGKILL."""
        self._stop_requested = True
        self.status = WorkerStatus.STOPPED

        if self._monitor_task is not None and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                self._logger.debug("monitor_task_cancelled_on_stop")

        if self.process is not None and self.process.returncode is None:
            self._logger.info("terminating_vllm_process", pid=self.pid, pgid=self.pgid)
            # Try terminating process group if available
            terminated = False
            if self.pgid is not None and hasattr(os, "killpg"):
                try:
                    os.killpg(self.pgid, signal.SIGTERM)
                    terminated = True
                except (OSError, ProcessLookupError) as exc:
                    self._logger.debug("killpg_sigterm_failed", error=str(exc))

            if not terminated:
                try:
                    self.process.terminate()
                except (OSError, ProcessLookupError) as exc:
                    self._logger.debug("terminate_failed", error=str(exc))

            # Await graceful shutdown
            try:
                await asyncio.wait_for(
                    self.process.wait(),
                    timeout=self.config.shutdown_timeout_seconds,
                )
            except TimeoutError:
                self._logger.warning("worker_sigterm_timed_out_escalating_to_sigkill", pid=self.pid)
                if self.pgid is not None and hasattr(os, "killpg"):
                    try:
                        os.killpg(self.pgid, signal.SIGKILL)
                    except (OSError, ProcessLookupError) as exc:
                        self._logger.debug("killpg_sigkill_failed", error=str(exc))
                else:
                    try:
                        self.process.kill()
                    except (OSError, ProcessLookupError) as exc:
                        self._logger.debug("kill_failed", error=str(exc))
                await self.process.wait()

        # Cancel stream tasks
        for task in (self._stdout_task, self._stderr_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    self._logger.debug("stream_task_cancelled")

        self._logger.info("vllm_process_stopped", pid=self.pid)


# Aliases for class contracts
VLLMSupervisor = VLLMProcessSupervisor
VLLMRunner = VLLMProcessSupervisor
