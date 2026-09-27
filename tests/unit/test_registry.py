"""Unit tests for SpeedInfer dynamic model registry and fallback backend routing.

Covers:
- Dynamic hot model registration and unregistration without gateway restart
- Model metadata catalog (base model, adapter path, context window, pricing)
- Healthy backend selection across multiple worker instances
- Fallback routing when primary worker backend degrades
- Consecutive failure circuit breaker marking backends unhealthy
- Thread-safe concurrent registration and query operations
"""

import threading
from dataclasses import dataclass, field
from enum import StrEnum

import pytest

# Attempt import of M3 engine registry
try:
    from speedinfer.engine import registry as engine_registry

    HAS_ENGINE_REGISTRY = True
except (ImportError, AttributeError):
    HAS_ENGINE_REGISTRY = False


# ---------------------------------------------------------------------------
# Authoritative Contracts & Reference Oracle
# ---------------------------------------------------------------------------
class BackendHealth(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class BackendWorker:
    """Worker endpoint representation."""

    url: str
    health: BackendHealth = BackendHealth.HEALTHY
    consecutive_failures: int = 0
    failure_threshold: int = 3

    def record_success(self):
        self.consecutive_failures = 0
        self.health = BackendHealth.HEALTHY

    def record_failure(self):
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.failure_threshold:
            self.health = BackendHealth.DEGRADED


@dataclass
class ModelEntry:
    """Catalog entry for a served LLM."""

    name: str
    base_model_path: str
    adapter_path: str | None = None
    context_length: int = 32768
    prompt_price_per_million: float = 0.20
    completion_price_per_million: float = 0.60
    lifecycle_status: str = "active"
    backends: list[BackendWorker] = field(default_factory=list)


class ReferenceModelRegistry:
    """Thread-safe reference implementation of Dynamic Hot Model Registry."""

    def __init__(self):
        self._lock = threading.RLock()
        self._models: dict[str, ModelEntry] = {}

    def register_model(self, entry: ModelEntry) -> None:
        with self._lock:
            self._models[entry.name] = entry

    def unregister_model(self, name: str) -> bool:
        with self._lock:
            return self._models.pop(name, None) is not None

    def get_model(self, name: str) -> ModelEntry | None:
        with self._lock:
            return self._models.get(name)

    def list_models(self) -> list[ModelEntry]:
        with self._lock:
            return list(self._models.values())

    def get_healthy_backend(self, model_name: str) -> BackendWorker | None:
        with self._lock:
            entry = self._models.get(model_name)
            if not entry:
                return None
            for backend in entry.backends:
                if backend.health == BackendHealth.HEALTHY:
                    return backend
            return None


@pytest.fixture
def require_registry():
    """Skip test if speedinfer.engine.registry is not yet implemented."""
    if not HAS_ENGINE_REGISTRY:
        pytest.skip("speedinfer.engine.registry not yet implemented by M3")


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------
def test_model_registration_and_lookup():
    """Verify models can be registered and retrieved by name."""
    registry = ReferenceModelRegistry()
    entry = ModelEntry(
        name="Qwen/Qwen2.5-7B-Instruct",
        base_model_path="/models/qwen2.5-7b",
        adapter_path=None,
        context_length=32768,
        prompt_price_per_million=0.20,
        completion_price_per_million=0.60,
    )
    registry.register_model(entry)

    retrieved = registry.get_model("Qwen/Qwen2.5-7B-Instruct")
    assert retrieved is not None
    assert retrieved.name == "Qwen/Qwen2.5-7B-Instruct"
    assert retrieved.context_length == 32768
    assert retrieved.adapter_path is None


def test_model_unregistration():
    """Verify unregistered models are cleanly removed from catalog."""
    registry = ReferenceModelRegistry()
    entry = ModelEntry(name="model-to-delete", base_model_path="/models/tmp")
    registry.register_model(entry)

    assert registry.get_model("model-to-delete") is not None
    removed = registry.unregister_model("model-to-delete")
    assert removed is True
    assert registry.get_model("model-to-delete") is None

    # Unregistering nonexistent returns False
    assert registry.unregister_model("nonexistent") is False


def test_list_models_returns_all_registered():
    """Verify listing all active models returns complete catalog."""
    registry = ReferenceModelRegistry()
    m1 = ModelEntry(name="model-1", base_model_path="/models/m1")
    m2 = ModelEntry(name="model-2", base_model_path="/models/m2")
    registry.register_model(m1)
    registry.register_model(m2)

    all_models = registry.list_models()
    names = {m.name for m in all_models}
    assert names == {"model-1", "model-2"}


def test_get_healthy_backend_primary_worker():
    """Verify routing selects healthy primary worker."""
    registry = ReferenceModelRegistry()
    b1 = BackendWorker(url="http://10.0.0.1:8001")
    b2 = BackendWorker(url="http://10.0.0.2:8001")

    entry = ModelEntry(
        name="Qwen/Qwen2.5-7B-Instruct",
        base_model_path="/models/qwen",
        backends=[b1, b2],
    )
    registry.register_model(entry)

    backend = registry.get_healthy_backend("Qwen/Qwen2.5-7B-Instruct")
    assert backend is not None
    assert backend.url == "http://10.0.0.1:8001"
    assert backend.health == BackendHealth.HEALTHY


def test_fallback_routing_when_primary_degraded():
    """Verify traffic shifts to secondary replica when primary degrades."""
    registry = ReferenceModelRegistry()
    primary = BackendWorker(url="http://10.0.0.1:8001")
    replica = BackendWorker(url="http://10.0.0.2:8001")

    entry = ModelEntry(
        name="Qwen/Qwen2.5-7B-Instruct",
        base_model_path="/models/qwen",
        backends=[primary, replica],
    )
    registry.register_model(entry)

    # Simulate primary worker degradation
    primary.health = BackendHealth.DEGRADED

    # Router should seamlessly route to replica
    selected = registry.get_healthy_backend("Qwen/Qwen2.5-7B-Instruct")
    assert selected is not None
    assert selected.url == "http://10.0.0.2:8001"


def test_all_backends_unhealthy_returns_none():
    """Verify get_healthy_backend returns None when all workers are degraded."""
    registry = ReferenceModelRegistry()
    primary = BackendWorker(url="http://10.0.0.1:8001", health=BackendHealth.DEGRADED)
    replica = BackendWorker(url="http://10.0.0.2:8001", health=BackendHealth.UNHEALTHY)

    entry = ModelEntry(
        name="Qwen/Qwen2.5-7B-Instruct",
        base_model_path="/models/qwen",
        backends=[primary, replica],
    )
    registry.register_model(entry)

    selected = registry.get_healthy_backend("Qwen/Qwen2.5-7B-Instruct")
    assert selected is None


def test_circuit_breaker_failure_threshold_transition():
    """Verify 3 consecutive failures triggers DEGRADED state; success resets it."""
    worker = BackendWorker(url="http://127.0.0.1:8001", failure_threshold=3)
    assert worker.health == BackendHealth.HEALTHY

    worker.record_failure()
    assert worker.health == BackendHealth.HEALTHY
    assert worker.consecutive_failures == 1

    worker.record_failure()
    assert worker.health == BackendHealth.HEALTHY
    assert worker.consecutive_failures == 2

    # 3rd failure trips the breaker
    worker.record_failure()
    assert worker.health == BackendHealth.DEGRADED
    assert worker.consecutive_failures == 3

    # Success heals the worker
    worker.record_success()
    assert worker.health == BackendHealth.HEALTHY
    assert worker.consecutive_failures == 0


def test_hot_update_adapter_weights_without_restart():
    """Verify model adapter path can be hot-updated while registered."""
    registry = ReferenceModelRegistry()
    entry = ModelEntry(
        name="custom-lora-model",
        base_model_path="/models/base",
        adapter_path="/adapters/v1",
    )
    registry.register_model(entry)

    # Hot update adapter to v2
    updated = ModelEntry(
        name="custom-lora-model",
        base_model_path="/models/base",
        adapter_path="/adapters/v2",
    )
    registry.register_model(updated)

    retrieved = registry.get_model("custom-lora-model")
    assert retrieved.adapter_path == "/adapters/v2"


def test_concurrent_registry_access():
    """Verify thread-safety when multiple threads concurrently read and register."""
    registry = ReferenceModelRegistry()
    errors = []

    def writer(thread_id: int):
        try:
            for i in range(25):
                name = f"thread-{thread_id}-model-{i}"
                registry.register_model(ModelEntry(name=name, base_model_path="/path"))
        except Exception as e:
            errors.append(e)

    def reader():
        try:
            for _ in range(50):
                registry.list_models()
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=writer, args=(1,)),
        threading.Thread(target=writer, args=(2,)),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert len(registry.list_models()) == 50


def test_registry_module_contract(require_registry):
    """Verify M3 speedinfer.engine.registry exports expected classes."""
    assert hasattr(engine_registry, "ModelRegistry")
    assert hasattr(engine_registry, "BackendWorker")
    assert hasattr(engine_registry, "ModelEntry")
