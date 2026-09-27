"""Dynamic hot model registry and backend load router.

Provides thread-safe model registration, catalog inspection, health filtering,
and fallback routing across multiple vLLM worker backends without restarting the gateway.
"""

import threading
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from speedinfer.engine.vllm_runner import WorkerStatus


class BackendHealth(StrEnum):
    """Health classification for worker backends."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class BackendWorker:
    """Representation of an active or standby inference worker backend."""

    url: str = ""
    endpoint_url: str = ""
    worker_id: str = ""
    status: WorkerStatus = WorkerStatus.READY
    health: BackendHealth = BackendHealth.HEALTHY
    active_requests: int = 0
    consecutive_failures: int = 0
    failure_threshold: int = 3
    weight: int = 1

    def __init__(
        self,
        url: str = "",
        endpoint_url: str = "",
        worker_id: str = "",
        status: WorkerStatus = WorkerStatus.READY,
        health: BackendHealth = BackendHealth.HEALTHY,
        active_requests: int = 0,
        consecutive_failures: int = 0,
        failure_threshold: int = 3,
        weight: int = 1,
    ) -> None:
        """Initialize worker backend with unified URL and identifier handling."""
        resolved_url = url or endpoint_url
        self.url = resolved_url
        self.endpoint_url = resolved_url
        self.worker_id = worker_id or resolved_url
        self.status = status
        self.health = health
        self.active_requests = active_requests
        self.consecutive_failures = consecutive_failures
        self.failure_threshold = failure_threshold
        self.weight = max(1, weight)

    def record_success(self) -> None:
        """Reset consecutive failure counter and restore healthy state."""
        self.consecutive_failures = 0
        self.status = WorkerStatus.READY
        self.health = BackendHealth.HEALTHY

    def record_failure(self) -> None:
        """Increment failure counter and trip circuit breaker if threshold is reached."""
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.failure_threshold:
            self.status = WorkerStatus.DEGRADED
            self.health = BackendHealth.DEGRADED

    def is_healthy(self) -> bool:
        """Return True if worker is healthy and ready to serve inference requests."""
        if self.health in {
            BackendHealth.DEGRADED,
            BackendHealth.UNHEALTHY,
            "degraded",
            "unhealthy",
        }:
            return False
        if self.status in {
            WorkerStatus.DEGRADED,
            WorkerStatus.FAILED,
            WorkerStatus.STOPPED,
            WorkerStatus.CRASHED,
            "degraded",
            "failed",
            "stopped",
            "crashed",
        }:
            return False
        return self.health in {BackendHealth.HEALTHY, "healthy"} or self.status in {
            WorkerStatus.READY,
            "ready",
        }


@dataclass
class ModelPricing:
    """Token pricing rates for billing calculation."""

    prompt_price_per_million: float = 0.20
    completion_price_per_million: float = 0.60

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Compute USD billing cost for prompt and completion token counts."""
        prompt_cost = (prompt_tokens / 1_000_000.0) * self.prompt_price_per_million
        completion_cost = (completion_tokens / 1_000_000.0) * self.completion_price_per_million
        return round(prompt_cost + completion_cost, 6)


@dataclass
class ModelEntry:
    """Catalog entry for a served LLM model."""

    name: str
    base_model_path: str
    adapter_path: str | None = None
    context_length: int = 32768
    prompt_price_per_million: float = 0.20
    completion_price_per_million: float = 0.60
    lifecycle_status: str = "active"
    fallback_model: str | None = None
    backends: list[BackendWorker] = field(default_factory=list)
    pricing: ModelPricing | None = None

    def __post_init__(self) -> None:
        """Normalize pricing models and aliases."""
        if self.pricing is not None:
            self.prompt_price_per_million = self.pricing.prompt_price_per_million
            self.completion_price_per_million = self.pricing.completion_price_per_million
        else:
            self.pricing = ModelPricing(
                prompt_price_per_million=self.prompt_price_per_million,
                completion_price_per_million=self.completion_price_per_million,
            )

    @property
    def base_model(self) -> str:
        """Alias for base_model_path."""
        return self.base_model_path

    @base_model.setter
    def base_model(self, value: str) -> None:
        """Setter for base_model alias."""
        self.base_model_path = value

    @property
    def context_window(self) -> int:
        """Alias for context_length."""
        return self.context_length

    @context_window.setter
    def context_window(self, value: int) -> None:
        """Setter for context_window alias."""
        self.context_length = value

    @property
    def fallback_model_id(self) -> str | None:
        """Alias for fallback_model."""
        return self.fallback_model

    @fallback_model_id.setter
    def fallback_model_id(self, value: str | None) -> None:
        """Setter for fallback_model alias."""
        self.fallback_model = value

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate USD inference cost using configured pricing."""
        if self.pricing is not None:
            return self.pricing.calculate_cost(prompt_tokens, completion_tokens)
        prompt_cost = (prompt_tokens / 1_000_000.0) * self.prompt_price_per_million
        completion_cost = (completion_tokens / 1_000_000.0) * self.completion_price_per_million
        return round(prompt_cost + completion_cost, 6)


class ModelRegistry:
    """Thread-safe dynamic hot model registry and routing directory."""

    def __init__(self) -> None:
        """Initialize empty model registry with re-entrant thread lock."""
        self._lock = threading.RLock()
        self._models: dict[str, ModelEntry] = {}
        self._rr_indices: dict[str, int] = {}

    def register_model(
        self,
        model_or_entry: ModelEntry | str | None = None,
        *,
        name: str | None = None,
        base_model_path: str | None = None,
        adapter_path: str | None = None,
        context_length: int = 32768,
        prompt_price_per_million: float = 0.20,
        completion_price_per_million: float = 0.60,
        lifecycle_status: str = "active",
        backends: list[BackendWorker] | None = None,
        fallback_model: str | None = None,
        pricing: ModelPricing | None = None,
    ) -> ModelEntry:
        """Register or update a model catalog entry in a thread-safe manner."""
        with self._lock:
            if isinstance(model_or_entry, ModelEntry):
                entry = model_or_entry
            elif isinstance(model_or_entry, str):
                entry = ModelEntry(
                    name=model_or_entry,
                    base_model_path=base_model_path or "",
                    adapter_path=adapter_path,
                    context_length=context_length,
                    prompt_price_per_million=prompt_price_per_million,
                    completion_price_per_million=completion_price_per_million,
                    lifecycle_status=lifecycle_status,
                    fallback_model=fallback_model,
                    backends=backends or [],
                    pricing=pricing,
                )
            elif name is not None:
                entry = ModelEntry(
                    name=name,
                    base_model_path=base_model_path or "",
                    adapter_path=adapter_path,
                    context_length=context_length,
                    prompt_price_per_million=prompt_price_per_million,
                    completion_price_per_million=completion_price_per_million,
                    lifecycle_status=lifecycle_status,
                    fallback_model=fallback_model,
                    backends=backends or [],
                    pricing=pricing,
                )
            else:
                raise ValueError("Must provide either a ModelEntry or model name")

            self._models[entry.name] = entry
            return entry

    def unregister_model(self, model_name: str) -> bool:
        """Remove a model from the active catalog. Returns True if removed."""
        with self._lock:
            removed = self._models.pop(model_name, None)
            self._rr_indices.pop(model_name, None)
            return removed is not None

    def add_backend(self, model_name: str, backend: BackendWorker) -> bool:
        """Attach a worker backend to an existing model. Returns True if attached."""
        with self._lock:
            entry = self._models.get(model_name)
            if entry is None:
                return False
            entry.backends.append(backend)
            return True

    def remove_backend(self, model_name: str, worker_id: str) -> bool:
        """Detach a worker backend matching worker_id or URL. Returns True if removed."""
        with self._lock:
            entry = self._models.get(model_name)
            if entry is None:
                return False
            initial_count = len(entry.backends)
            entry.backends = [
                b
                for b in entry.backends
                if b.worker_id != worker_id and b.url != worker_id and b.endpoint_url != worker_id
            ]
            return len(entry.backends) < initial_count

    def get_model(self, model_name: str) -> ModelEntry | None:
        """Retrieve model metadata entry by name."""
        with self._lock:
            return self._models.get(model_name)

    def list_models(self) -> list[ModelEntry]:
        """Return a snapshot list of all currently registered models."""
        with self._lock:
            return list(self._models.values())

    def _is_backend_healthy(self, backend: Any) -> bool:
        """Check whether a backend instance is considered healthy."""
        if hasattr(backend, "is_healthy") and callable(backend.is_healthy):
            return bool(backend.is_healthy())

        health = getattr(backend, "health", None)
        if health is not None:
            if health in {BackendHealth.DEGRADED, BackendHealth.UNHEALTHY, "degraded", "unhealthy"}:
                return False
            if health in {BackendHealth.HEALTHY, "healthy"}:
                return True

        status = getattr(backend, "status", None)
        if status is not None:
            if status in {
                WorkerStatus.DEGRADED,
                WorkerStatus.FAILED,
                WorkerStatus.STOPPED,
                WorkerStatus.CRASHED,
                "degraded",
                "failed",
                "stopped",
                "crashed",
            }:
                return False
            if status in {WorkerStatus.READY, "ready"}:
                return True

        return True

    def get_healthy_backend(
        self,
        model_name: str,
        strategy: str = "least_connections",
    ) -> BackendWorker | None:
        """Select a healthy worker backend using specified routing strategy or fallback."""
        with self._lock:
            entry = self._models.get(model_name)
            if entry is None:
                return None

            healthy_backends = [b for b in entry.backends if self._is_backend_healthy(b)]

            if not healthy_backends:
                # Attempt fallback model resolution if configured
                if entry.fallback_model and entry.fallback_model != model_name:
                    return self.get_healthy_backend(entry.fallback_model, strategy=strategy)
                return None

            if strategy == "least_connections":
                return min(healthy_backends, key=lambda b: (b.active_requests, -b.weight))
            elif strategy == "round_robin":
                idx = self._rr_indices.get(model_name, 0) % len(healthy_backends)
                self._rr_indices[model_name] = (idx + 1) % len(healthy_backends)
                return healthy_backends[idx]
            else:
                return healthy_backends[0]

    def report_backend_failure(self, model_name: str, worker_id: str) -> None:
        """Record a failure event for circuit-breaker tracking."""
        with self._lock:
            entry = self._models.get(model_name)
            targets = entry.backends if entry else []
            found = False
            for b in targets:
                if b.worker_id == worker_id or b.url == worker_id or b.endpoint_url == worker_id:
                    b.record_failure()
                    found = True
                    break

            # Fallback scan across all models if not found in specified model
            if not found:
                for mod in self._models.values():
                    for b in mod.backends:
                        if (
                            b.worker_id == worker_id
                            or b.url == worker_id
                            or b.endpoint_url == worker_id
                        ):
                            b.record_failure()
                            break

    def report_backend_success(self, model_name: str, worker_id: str) -> None:
        """Record a successful request and reset consecutive failures."""
        with self._lock:
            entry = self._models.get(model_name)
            targets = entry.backends if entry else []
            found = False
            for b in targets:
                if b.worker_id == worker_id or b.url == worker_id or b.endpoint_url == worker_id:
                    b.record_success()
                    found = True
                    break

            if not found:
                for mod in self._models.values():
                    for b in mod.backends:
                        if (
                            b.worker_id == worker_id
                            or b.url == worker_id
                            or b.endpoint_url == worker_id
                        ):
                            b.record_success()
                            break
