"""Runtime adapters for connected inference workers.

The gateway speaks OpenAI-compatible HTTP by default. NVIDIA Triton workers use
the Triton HTTP v2 protocol through the adapter below. TensorRT-LLM is supported
either behind Triton (the common deployment) or through an OpenAI-compatible
frontend selected in the worker configuration.
"""

import json
import time
import uuid
from typing import Any

import httpx
from fastapi import HTTPException

from speedinfer.engine.registry import BackendWorker, WorkerRuntime


def _runtime_error(message: str, code: str = "runtime_unavailable") -> HTTPException:
    return HTTPException(
        503,
        detail={
            "error": {
                "message": message,
                "type": "server_error",
                "code": code,
            }
        },
    )


class TritonRuntimeAdapter:
    """Small Triton HTTP v2 adapter for text-generation model endpoints."""

    def __init__(self, worker: BackendWorker, client: httpx.AsyncClient) -> None:
        self.worker = worker
        self.client = client
        self.config = worker.runtime_config
        self.base_url = worker.runtime_endpoint.rstrip("/")

    @property
    def runtime_model(self) -> str:
        model = str(self.config.get("model_name") or "").strip()
        if not model:
            raise _runtime_error(
                "Triton worker is missing runtime_config.model_name.",
                "runtime_configuration_error",
            )
        return model

    def _headers(self) -> dict[str, str]:
        headers = {str(k): str(v) for k, v in self.config.get("headers", {}).items()}
        api_key = self.config.get("api_key")
        if api_key:
            headers.setdefault("Authorization", f"Bearer {api_key}")
        return headers

    async def health(self) -> dict[str, Any]:
        """Check Triton server readiness and configured model readiness."""
        try:
            server = await self.client.get(
                f"{self.base_url}/v2/health/ready", headers=self._headers()
            )
            model = await self.client.get(
                f"{self.base_url}/v2/models/{self.runtime_model}/ready",
                headers=self._headers(),
            )
        except httpx.HTTPError:
            self.worker.record_failure()
            return {"healthy": False, "server_ready": False, "model_ready": False}
        healthy = server.status_code == 200 and model.status_code == 200
        if healthy:
            self.worker.record_success()
        else:
            self.worker.record_failure()
        return {
            "healthy": healthy,
            "server_ready": server.status_code == 200,
            "model_ready": model.status_code == 200,
        }

    def _prompt(self, messages: list[Any]) -> str:
        template = self.config.get("chat_template", "{messages_json}")
        items = [
            {
                "role": str(getattr(message, "role", "user")),
                "content": str(getattr(message, "content", "")),
            }
            for message in messages
        ]
        return str(template).format(messages_json=json.dumps(items, ensure_ascii=False))

    def _infer_payload(self, prompt: str, max_tokens: int) -> dict[str, Any]:
        input_name = str(self.config.get("input_name", "text_input"))
        max_tokens_name = str(self.config.get("max_tokens_input_name", "max_tokens"))
        payload: dict[str, Any] = {
            "inputs": [
                {
                    "name": input_name,
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [[prompt]],
                },
                {
                    "name": max_tokens_name,
                    "shape": [1, 1],
                    "datatype": "INT32",
                    "data": [[max_tokens]],
                },
            ],
            "outputs": [{"name": str(self.config.get("output_name", "text_output"))}],
        }
        for name in ("prompt_tokens_output_name", "completion_tokens_output_name"):
            if self.config.get(name):
                payload["outputs"].append({"name": str(self.config[name])})
        return payload

    @staticmethod
    def _first_value(output: dict[str, Any]) -> Any:
        data = output.get("data", [])
        while isinstance(data, list) and data:
            data = data[0]
        return data

    async def infer(self, prompt: str, max_tokens: int) -> tuple[str, dict[str, int]]:
        """Send one non-streaming inference request using Triton HTTP v2."""
        response = await self.client.post(
            f"{self.base_url}/v2/models/{self.runtime_model}/infer",
            json=self._infer_payload(prompt, max_tokens),
            headers=self._headers(),
        )
        if response.status_code != 200:
            self.worker.record_failure()
            raise _runtime_error(
                "The connected Triton worker rejected the inference request.",
                "triton_inference_failed",
            )
        outputs = {item.get("name"): item for item in response.json().get("outputs", [])}
        text_item = outputs.get(str(self.config.get("output_name", "text_output")))
        if not text_item:
            self.worker.record_failure()
            raise _runtime_error(
                "Triton response did not include the configured text output.",
                "invalid_runtime_response",
            )
        prompt_name = self.config.get("prompt_tokens_output_name")
        completion_name = self.config.get("completion_tokens_output_name")
        if (
            not prompt_name
            or not completion_name
            or prompt_name not in outputs
            or completion_name not in outputs
        ):
            raise _runtime_error(
                "Triton model must return configured prompt and completion token counts.",
                "runtime_usage_unavailable",
            )
        usage = {
            "prompt_tokens": int(self._first_value(outputs[prompt_name])),
            "completion_tokens": int(self._first_value(outputs[completion_name])),
        }
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        self.worker.record_success()
        return str(self._first_value(text_item)), usage

    async def chat_response(self, request: Any, public_model: str) -> dict[str, Any]:
        text, usage = await self.infer(self._prompt(request.messages), request.max_tokens or 128)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": public_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }

    async def completion_response(self, request: Any, public_model: str) -> dict[str, Any]:
        prompt = request.prompt if isinstance(request.prompt, str) else " ".join(request.prompt)
        text, usage = await self.infer(prompt, request.max_tokens or 16)
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": public_model,
            "choices": [{"text": text, "index": 0, "logprobs": None, "finish_reason": "stop"}],
            "usage": usage,
        }


def uses_triton_protocol(worker: BackendWorker) -> bool:
    """Return whether this worker should be called through Triton HTTP v2."""
    if worker.runtime == WorkerRuntime.TRITON:
        return True
    return worker.runtime == WorkerRuntime.TENSORRT_LLM and str(
        worker.runtime_config.get("transport", "triton")
    ).lower() == "triton"


async def probe_worker(worker: BackendWorker, client: httpx.AsyncClient) -> dict[str, Any]:
    """Probe a worker using its configured serving protocol."""
    if uses_triton_protocol(worker):
        return await TritonRuntimeAdapter(worker, client).health()
    base_url = worker.runtime_endpoint.rstrip("/")
    models_url = f"{base_url}/models" if base_url.endswith("/v1") else f"{base_url}/v1/models"
    headers = {str(k): str(v) for k, v in worker.runtime_config.get("headers", {}).items()}
    api_key = worker.runtime_config.get("api_key")
    if api_key:
        headers.setdefault("Authorization", f"Bearer {api_key}")
    try:
        response = await client.get(models_url, headers=headers)
        items = response.json().get("data", []) if response.status_code == 200 else []
        available = {str(item.get("id")) for item in items}
        models_ready = not worker.supported_models or bool(available & set(worker.supported_models))
        healthy = response.status_code == 200 and models_ready
    except (httpx.HTTPError, ValueError):
        healthy = False
        models_ready = False
    if healthy:
        worker.record_success()
    else:
        worker.record_failure()
    return {"healthy": healthy, "server_ready": healthy, "model_ready": models_ready}
