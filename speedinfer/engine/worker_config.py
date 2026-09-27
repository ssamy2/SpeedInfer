"""Environment-driven registration for connected inference workers."""

import json
from typing import Any

from speedinfer.engine.registry import BackendWorker, ModelRegistry, WorkerRuntime


def load_workers_json(raw: str) -> list[dict[str, Any]]:
    """Validate and return worker definitions from a JSON array."""
    if not raw.strip():
        return []
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("NVIDIA_WORKERS_JSON must be valid JSON.") from exc
    if not isinstance(items, list):
        raise ValueError("NVIDIA_WORKERS_JSON must contain a JSON array.")
    return items


def register_configured_workers(registry: ModelRegistry, raw: str) -> list[BackendWorker]:
    """Register NVIDIA worker definitions and attach them to supported models."""
    workers: list[BackendWorker] = []
    for item in load_workers_json(raw):
        if not isinstance(item, dict):
            raise ValueError("Each worker definition must be a JSON object.")
        worker_id = str(item.get("worker_id", "")).strip()
        endpoint = str(item.get("runtime_endpoint") or item.get("url") or "").strip()
        models = item.get("supported_models", [])
        if not worker_id or not endpoint or not isinstance(models, list) or not models:
            raise ValueError(
                "Each configured worker needs worker_id, runtime_endpoint, and supported_models."
            )
        runtime = WorkerRuntime(str(item.get("runtime", "triton")).lower())
        if runtime not in {
            WorkerRuntime.CUDA,
            WorkerRuntime.TRITON,
            WorkerRuntime.TENSORRT_LLM,
        }:
            raise ValueError("NVIDIA workers require cuda, triton, or tensorrt-llm runtime.")
        runtime_config = item.get("runtime_config", {})
        gpu_metadata = item.get("gpu_metadata", {})
        runtime_metadata = item.get("runtime_metadata", {})
        metadata_values = (runtime_config, gpu_metadata, runtime_metadata)
        if not all(isinstance(value, dict) for value in metadata_values):
            raise ValueError("Worker runtime and GPU metadata values must be JSON objects.")
        worker = BackendWorker(
            url=endpoint,
            worker_id=worker_id,
            worker_type=str(item.get("worker_type", "nvidia-gpu")),
            runtime=runtime,
            runtime_endpoint=endpoint,
            runtime_config=runtime_config,
            supported_models=[str(model) for model in models],
            gpu_metadata=gpu_metadata,
            runtime_metadata=runtime_metadata,
            weight=int(item.get("weight", 1)),
        )
        for model_name in worker.supported_models:
            entry = registry.get_model(model_name)
            if entry is None:
                registry.register_model(
                    name=model_name,
                    base_model_path=model_name,
                    backends=[worker],
                )
            elif all(existing.worker_id != worker.worker_id for existing in entry.backends):
                registry.add_backend(model_name, worker)
        workers.append(worker)
    return workers
