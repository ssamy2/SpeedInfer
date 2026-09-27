"""Contracts for NVIDIA worker registration, routing, and Triton HTTP v2."""

import json

import httpx
import pytest

from speedinfer.engine.registry import BackendWorker, ModelRegistry, WorkerRuntime
from speedinfer.engine.runtime_adapters import TritonRuntimeAdapter, probe_worker
from speedinfer.engine.worker_config import register_configured_workers
from speedinfer.gateway.schemas import ChatCompletionRequest


def triton_worker() -> BackendWorker:
    return BackendWorker(
        url="http://triton:8000",
        worker_id="nvidia-triton-01",
        worker_type="nvidia-gpu",
        runtime=WorkerRuntime.TRITON,
        supported_models=["acme/model-1"],
        runtime_config={
            "model_name": "ensemble",
            "output_name": "text_output",
            "prompt_tokens_output_name": "prompt_tokens",
            "completion_tokens_output_name": "completion_tokens",
        },
        gpu_metadata={"product": "configured-test-gpu"},
        runtime_metadata={"cuda_version": "test"},
    )


def test_configured_nvidia_worker_registration_and_compatible_routing():
    registry = ModelRegistry()
    registry.register_model(name="acme/model-1", base_model_path="acme/model-1")
    registry.register_model(name="acme/model-2", base_model_path="acme/model-2")
    config = json.dumps(
        [
            {
                "worker_id": "nvidia-01",
                "runtime": "tensorrt-llm",
                "runtime_endpoint": "http://triton:8000",
                "supported_models": ["acme/model-1"],
                "runtime_config": {"model_name": "ensemble"},
            }
        ]
    )
    workers = register_configured_workers(registry, config)
    assert workers[0].runtime == WorkerRuntime.TENSORRT_LLM
    assert registry.get_healthy_backend("acme/model-1") is workers[0]
    assert registry.get_healthy_backend("acme/model-2") is None


@pytest.mark.asyncio
async def test_triton_health_model_readiness_and_inference_contract():
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        if request.method == "GET":
            return httpx.Response(200)
        payload = json.loads(request.content)
        assert payload["inputs"][0]["name"] == "text_input"
        return httpx.Response(
            200,
            json={
                "outputs": [
                    {"name": "text_output", "data": [["worker response"]]},
                    {"name": "prompt_tokens", "data": [[4]]},
                    {"name": "completion_tokens", "data": [[2]]},
                ]
            },
        )

    worker = triton_worker()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        health = await probe_worker(worker, client)
        result = await TritonRuntimeAdapter(worker, client).chat_response(
            ChatCompletionRequest(
                model="acme/model-1",
                messages=[{"role": "user", "content": "Hello"}],
            ),
            "acme/model-1",
        )
    assert health == {"healthy": True, "server_ready": True, "model_ready": True}
    assert "/v2/health/ready" in seen_paths
    assert "/v2/models/ensemble/ready" in seen_paths
    assert "/v2/models/ensemble/infer" in seen_paths
    assert result["choices"][0]["message"]["content"] == "worker response"
    assert result["usage"] == {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}


def test_worker_rejects_undeclared_model():
    worker = triton_worker()
    assert worker.supports_model("acme/model-1")
    assert not worker.supports_model("acme/model-2")
