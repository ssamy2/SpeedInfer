"""Dynamic reverse proxy and inference backend dispatcher.

Routes inference requests across healthy vLLM worker instances with
automatic load balancing, circuit breaking, and mock test emulation.
"""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi import HTTPException

from speedinfer.config import get_settings
from speedinfer.engine.registry import ModelEntry, ModelRegistry
from speedinfer.engine.runtime_adapters import TritonRuntimeAdapter, uses_triton_protocol
from speedinfer.gateway.schemas import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatResponseMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    DeltaMessage,
    UsageInfo,
)


class InferenceProxy:
    """Reverse proxy dispatcher for backend inference workers."""

    def __init__(self, registry: ModelRegistry) -> None:
        """Initialize proxy with model registry.

        Args:
            registry: Active ModelRegistry instance.
        """
        self.registry = registry
        self.http_client: httpx.AsyncClient | None = None

    async def get_client(self) -> httpx.AsyncClient:
        """Get or initialize persistent HTTP async client."""
        if self.http_client is None or self.http_client.is_closed:
            settings = get_settings()
            headers = {
                "HTTP-Referer": "https://speedinfer.com",
                "X-Title": "SpeedInfer",
            }
            if settings.vllm_api_key:
                headers["Authorization"] = f"Bearer {settings.vllm_api_key.get_secret_value()}"
            self.http_client = httpx.AsyncClient(
                timeout=settings.vllm_timeout_seconds,
                headers=headers,
            )
        return self.http_client

    async def close(self) -> None:
        """Close open HTTP connections."""
        if self.http_client and not self.http_client.is_closed:
            await self.http_client.aclose()
            self.http_client = None

    def estimate_prompt_tokens(self, text_or_messages: str | list[Any]) -> int:
        """Estimate prompt token count using standard heuristic (approx 4 chars per token)."""
        if isinstance(text_or_messages, str):
            total_chars = len(text_or_messages)
        elif isinstance(text_or_messages, list):
            total_chars = 0
            for item in text_or_messages:
                if isinstance(item, dict):
                    total_chars += len(str(item.get("content", "")))
                elif hasattr(item, "content"):
                    total_chars += len(str(item.content))
                else:
                    total_chars += len(str(item))
        else:
            total_chars = len(str(text_or_messages))
        return max(1, total_chars // 4)

    async def execute_chat(
        self,
        request: ChatCompletionRequest,
        model_entry: ModelEntry,
    ) -> ChatCompletionResponse:
        """Dispatch non-streaming chat completion to worker or generate mock completion."""
        worker = self.registry.get_healthy_backend(
            request.model, strategy="round_robin", allow_fallback=False
        )
        client = await self.get_client()

        # If a live worker backend exists, attempt forwarding
        if worker is not None and not worker.url.startswith("http://mock-"):
            worker.active_requests += 1
            try:
                if uses_triton_protocol(worker):
                    adapter = TritonRuntimeAdapter(worker, client)
                    data = await adapter.chat_response(request, request.model)
                    return ChatCompletionResponse.model_validate(data)
                base_url = worker.url.rstrip("/")
                target_url = (
                    f"{base_url}/chat/completions"
                    if base_url.endswith("/v1")
                    else f"{base_url}/v1/chat/completions"
                )
                payload = request.model_dump(exclude_none=True)
                if model_entry.base_model_path:
                    payload["model"] = model_entry.base_model_path
                resp = await client.post(target_url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    data["model"] = request.model
                    result = ChatCompletionResponse.model_validate(data)
                    worker.record_success()
                    return result
                if resp.status_code >= 500:
                    worker.record_failure()
                else:
                    try:
                        err_payload = resp.json()
                        err_detail = (
                            err_payload
                            if "error" in err_payload
                            else {
                                "error": {
                                    "message": resp.text,
                                    "type": "invalid_request_error",
                                    "code": resp.status_code,
                                }
                            }
                        )
                    except Exception:
                        err_msg = resp.text or f"Upstream rejected request: {resp.status_code}"
                        err_detail = {
                            "error": {
                                "message": err_msg,
                                "type": "invalid_request_error",
                                "code": resp.status_code,
                            }
                        }
                    raise HTTPException(status_code=resp.status_code, detail=err_detail)
            except HTTPException:
                raise
            except Exception:
                worker.record_failure()
            finally:
                worker.active_requests = max(0, worker.active_requests - 1)

        if get_settings().environment != "test":
            raise HTTPException(
                503,
                detail={
                    "error": {
                        "message": (
                            "No healthy compatible worker is available for this model. "
                            "No charge applied."
                        ),
                        "type": "server_error",
                        "code": "compatible_worker_unavailable",
                    }
                },
            )

        # Fallback / Test emulation mode
        prompt_tokens = self.estimate_prompt_tokens(request.messages)
        # Generate appropriate response text
        user_msg = ""
        for m in request.messages:
            if m.role == "user":
                user_msg = m.content

        if "2+2" in user_msg or "2 + 2" in user_msg:
            reply_text = "2 + 2 = 4."
        elif "Hello" in user_msg or "Hi" in user_msg:
            reply_text = "Hello! I am SpeedInfer, your high-performance inference gateway."
        else:
            reply_text = (
                "SpeedInfer delivers ultra-low latency LLM inference via vLLM and PagedAttention."
            )

        completion_tokens = max(1, len(reply_text) // 4)
        if request.max_tokens:
            completion_tokens = min(completion_tokens, request.max_tokens)
        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        return ChatCompletionResponse(
            id=cmpl_id,
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatResponseMessage(role="assistant", content=reply_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def stream_chat(
        self,
        request: ChatCompletionRequest,
        model_entry: ModelEntry,
    ) -> AsyncGenerator[str, None]:
        """Stream SSE chunks for chat completion."""
        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        # Check for live worker backend
        worker = self.registry.get_healthy_backend(
            request.model, strategy="round_robin", allow_fallback=False
        )
        client = await self.get_client()

        if worker is not None and not worker.url.startswith("http://mock-"):
            worker.active_requests += 1
            try:
                if uses_triton_protocol(worker):
                    raise HTTPException(
                        503,
                        detail={
                            "error": {
                                "message": (
                                    "Streaming is unavailable for this Triton HTTP worker. "
                                    "Use a non-streaming request or an OpenAI-compatible "
                                    "TensorRT-LLM frontend."
                                ),
                                "type": "server_error",
                                "code": "runtime_streaming_unavailable",
                            }
                        },
                    )
                base_url = worker.url.rstrip("/")
                target_url = (
                    f"{base_url}/chat/completions"
                    if base_url.endswith("/v1")
                    else f"{base_url}/v1/chat/completions"
                )
                payload = request.model_dump(exclude_none=True)
                if model_entry.base_model_path:
                    payload["model"] = model_entry.base_model_path
                payload["stream"] = True
                payload["stream_options"] = {"include_usage": True}

                async with client.stream("POST", target_url, json=payload) as response:
                    if response.status_code == 200:
                        async for line in response.aiter_lines():
                            if not line:
                                continue
                            if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                                try:
                                    chunk_dict = json.loads(line[6:])
                                    if "model" in chunk_dict:
                                        chunk_dict["model"] = request.model
                                        yield f"data: {json.dumps(chunk_dict)}\n\n"
                                        continue
                                except Exception:
                                    pass
                            yield f"{line}\n\n"
                        worker.record_success()
                        return
                    if response.status_code >= 500:
                        worker.record_failure()
                    else:
                        err_bytes = await response.aread()
                        err_text = err_bytes.decode(errors="replace")
                        try:
                            err_payload = json.loads(err_text)
                            err_detail = (
                                err_payload
                                if "error" in err_payload
                                else {
                                    "error": {
                                        "message": err_text,
                                        "type": "invalid_request_error",
                                        "code": response.status_code,
                                    }
                                }
                            )
                        except Exception:
                            err_msg = (
                                err_text or f"Upstream rejected request: {response.status_code}"
                            )
                            err_detail = {
                                "error": {
                                    "message": err_msg,
                                    "type": "invalid_request_error",
                                    "code": response.status_code,
                                }
                            }
                        raise HTTPException(status_code=response.status_code, detail=err_detail)
            except HTTPException:
                raise
            except Exception:
                worker.record_failure()
            finally:
                worker.active_requests = max(0, worker.active_requests - 1)

        if get_settings().environment != "test":
            raise HTTPException(
                503,
                detail={
                    "error": {
                        "message": (
                            "No healthy compatible worker is available for this model. "
                            "No charge applied."
                        ),
                        "type": "server_error",
                        "code": "compatible_worker_unavailable",
                    }
                },
            )

        # Fallback / Test emulation mode
        user_msg = ""
        for m in request.messages:
            if m.role == "user":
                user_msg = m.content

        if "Count from 1 to 5" in user_msg or "Count to 3" in user_msg:
            words = ["1", ", ", "2", ", ", "3", ", ", "4", ", ", "5"]
        elif "poem" in user_msg.lower():
            words = [
                "The",
                " server",
                " hums",
                " in",
                " quiet",
                " grace,\n",
                "Tokens",
                " streaming",
                " through",
                " time",
                " and",
                " space.",
            ]
        else:
            words = ["Speed", "Infer", " delivers", " high", " throughput", " inference."]

        # 1. First chunk with role
        chunk_init = ChatCompletionChunk(
            id=cmpl_id,
            object="chat.completion.chunk",
            created=created,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant"),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {chunk_init.model_dump_json(exclude_none=True)}\n\n"

        # 2. Content chunks
        for w in words:
            await asyncio.sleep(0.005)
            chunk = ChatCompletionChunk(
                id=cmpl_id,
                object="chat.completion.chunk",
                created=created,
                model=request.model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=DeltaMessage(content=w),
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

        # 3. Final chunk with finish_reason
        chunk_final = ChatCompletionChunk(
            id=cmpl_id,
            object="chat.completion.chunk",
            created=created,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=DeltaMessage(),
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {chunk_final.model_dump_json(exclude_none=True)}\n\n"

        # Test-only usage event follows the same upstream usage contract.
        prompt_tokens = self.estimate_prompt_tokens(request.messages)
        completion_tokens = max(1, len("".join(words)) // 4)
        if request.max_tokens:
            completion_tokens = min(completion_tokens, request.max_tokens)
        yield (
            "data: "
            + json.dumps(
                {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
            )
            + "\n\n"
        )

        # 4. Termination sentinel
        yield "data: [DONE]\n\n"

    async def execute_completion(
        self,
        request: CompletionRequest,
        model_entry: ModelEntry,
    ) -> CompletionResponse:
        """Dispatch legacy text completion to worker or generate mock completion."""
        worker = self.registry.get_healthy_backend(
            request.model, strategy="round_robin", allow_fallback=False
        )
        client = await self.get_client()

        if worker is not None and not worker.url.startswith("http://mock-"):
            worker.active_requests += 1
            try:
                if uses_triton_protocol(worker):
                    adapter = TritonRuntimeAdapter(worker, client)
                    data = await adapter.completion_response(request, request.model)
                    return CompletionResponse.model_validate(data)
                base_url = worker.url.rstrip("/")
                target_url = (
                    f"{base_url}/completions"
                    if base_url.endswith("/v1")
                    else f"{base_url}/v1/completions"
                )
                payload = request.model_dump(exclude_none=True)
                if model_entry.base_model_path:
                    payload["model"] = model_entry.base_model_path
                resp = await client.post(target_url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    data["model"] = request.model
                    result = CompletionResponse.model_validate(data)
                    worker.record_success()
                    return result
                if resp.status_code >= 500:
                    worker.record_failure()
                else:
                    try:
                        err_payload = resp.json()
                        err_detail = (
                            err_payload
                            if "error" in err_payload
                            else {
                                "error": {
                                    "message": resp.text,
                                    "type": "invalid_request_error",
                                    "code": resp.status_code,
                                }
                            }
                        )
                    except Exception:
                        err_msg = resp.text or f"Upstream rejected request: {resp.status_code}"
                        err_detail = {
                            "error": {
                                "message": err_msg,
                                "type": "invalid_request_error",
                                "code": resp.status_code,
                            }
                        }
                    raise HTTPException(status_code=resp.status_code, detail=err_detail)
            except HTTPException:
                raise
            except Exception:
                worker.record_failure()
            finally:
                worker.active_requests = max(0, worker.active_requests - 1)

        if get_settings().environment != "test":
            raise HTTPException(
                503,
                detail={
                    "error": {
                        "message": (
                            "No healthy compatible worker is available for this model. "
                            "No charge applied."
                        ),
                        "type": "server_error",
                        "code": "compatible_worker_unavailable",
                    }
                },
            )

        # Fallback / Test emulation mode
        prompt_tokens = self.estimate_prompt_tokens(request.prompt)
        prompt_str = request.prompt if isinstance(request.prompt, str) else " ".join(request.prompt)

        if "Fast inference is" in prompt_str:
            reply_text = (
                " achieved through optimized kernel scheduling, "
                "KV caching, and continuous batching."
            )
        else:
            reply_text = " completed with ultra-low latency."

        completion_tokens = max(1, len(reply_text) // 4)
        if request.max_tokens:
            completion_tokens = min(completion_tokens, request.max_tokens)
        cmpl_id = f"cmpl-{uuid.uuid4().hex[:24]}"

        return CompletionResponse(
            id=cmpl_id,
            object="text_completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                CompletionChoice(
                    text=reply_text,
                    index=0,
                    logprobs=None,
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
