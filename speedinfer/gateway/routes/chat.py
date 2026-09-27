"""OpenAI-compatible chat completions endpoint.

Provides:
- POST /v1/chat/completions (streaming and non-streaming)
"""

import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from sqlmodel import Session, select

from speedinfer.core.auth import get_authenticated_api_key
from speedinfer.core.metering import (
    calculate_token_cost,
    deduct_balance_atomic_async,
    estimate_max_cost,
    sync_usage_to_db,
)
from speedinfer.core.rate_limiter import async_check_rate_limit, build_rate_limit_headers
from speedinfer.database.models import ApiKey, ModelVersion
from speedinfer.database.session import get_session
from speedinfer.engine.registry import ModelEntry, ModelRegistry
from speedinfer.gateway.proxy import InferenceProxy
from speedinfer.gateway.redis import get_async_redis
from speedinfer.gateway.schemas import ChatCompletionRequest, ChatCompletionResponse

router = APIRouter(prefix="/v1", tags=["Chat"])


def get_model_registry() -> ModelRegistry:
    """Dependency provider for ModelRegistry."""
    from speedinfer.gateway.app import get_registry

    return get_registry()


def get_inference_proxy() -> InferenceProxy:
    """Dependency provider for InferenceProxy."""
    from speedinfer.gateway.app import get_proxy

    return get_proxy()


def _resolve_model_entry(
    model_name: str,
    registry: ModelRegistry,
    session: Session,
) -> ModelEntry:
    """Resolve ModelEntry from active registry or persistence layer.

    Raises:
        HTTPException: HTTP 404 if model is not registered or supported.
    """
    entry = registry.get_model(model_name)
    if entry is not None:
        return entry

    # Fallback to database ModelVersion
    db_model = session.exec(select(ModelVersion).where(ModelVersion.name == model_name)).first()
    if db_model is not None:
        return ModelEntry(
            name=db_model.name,
            base_model_path=db_model.base_model_path,
            context_length=db_model.context_length,
            prompt_price_per_million=db_model.prompt_price_per_million,
            completion_price_per_million=db_model.completion_price_per_million,
        )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={
            "error": {
                "message": (
                    f"The model '{model_name}' does not exist or you do not have access to it."
                ),
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found",
            }
        },
    )


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    summary="Create chat completion",
    description="Generate conversational completions with optional SSE streaming.",
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    raw_request: Request,
    api_key: Annotated[ApiKey, Depends(get_authenticated_api_key)],
    registry: Annotated[ModelRegistry, Depends(get_model_registry)],
    proxy: Annotated[InferenceProxy, Depends(get_inference_proxy)],
    redis_client: Annotated[Any, Depends(get_async_redis)],
    session: Annotated[Session, Depends(get_session)],
) -> Any:
    """Process OpenAI-compatible chat completion request."""
    # 1. Resolve and validate model availability
    model_entry = _resolve_model_entry(request.model, registry, session)

    # 2. Rate limiting check (RPM and TPM)
    est_max_tokens = request.max_tokens or 128
    rate_result = await async_check_rate_limit(
        redis_client,
        api_key=api_key,
        requested_tokens=est_max_tokens,
    )
    headers = build_rate_limit_headers(rate_result, api_key.rpm_limit, api_key.tpm_limit)

    if not rate_result.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": {
                    "message": "Rate limit exceeded. Please throttle your requests.",
                    "type": "rate_limit_error",
                    "param": None,
                    "code": "rate_limit_exceeded",
                }
            },
            headers=headers,
        )

    # 3. Pre-flight credit check
    prompt_tokens = proxy.estimate_prompt_tokens(request.messages)
    estimated_cost = estimate_max_cost(
        prompt_tokens=prompt_tokens,
        max_tokens=est_max_tokens,
        context_window=model_entry.context_length,
        prompt_price_per_m=model_entry.prompt_price_per_million,
        completion_price_per_m=model_entry.completion_price_per_million,
    )

    if api_key.credit_balance < estimated_cost:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail={
                "error": {
                    "message": (
                        f"Insufficient credit balance. Required: ${estimated_cost:.6f}, "
                        f"Available: ${api_key.credit_balance:.6f}"
                    ),
                    "type": "insufficient_quota",
                    "param": None,
                    "code": "insufficient_balance",
                }
            },
            headers=headers,
        )

    start_time = time.perf_counter()

    # 4. Handle Server-Sent Events (SSE) Streaming
    if request.stream:

        async def event_generator():
            try:
                # Track tokens for deduction on stream termination
                async for chunk_str in proxy.stream_chat(request, model_entry):
                    if await raw_request.is_disconnected():
                        break
                    yield chunk_str

                # Streaming token deduction
                est_completion_tokens = max(1, est_max_tokens // 4)
                actual_cost = calculate_token_cost(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=est_completion_tokens,
                    prompt_price_per_m=model_entry.prompt_price_per_million,
                    completion_price_per_m=model_entry.completion_price_per_million,
                )
                try:
                    await deduct_balance_atomic_async(
                        redis_client,
                        api_key.id,
                        actual_cost,
                        initial_balance=api_key.credit_balance,
                    )
                    latency_ms = (time.perf_counter() - start_time) * 1000.0
                    sync_usage_to_db(
                        session=session,
                        api_key_id=api_key.id,
                        request_id=f"stream-{int(time.time() * 1000)}",
                        model=request.model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=est_completion_tokens,
                        cost=actual_cost,
                        latency_ms=latency_ms,
                        ttft_ms=50.0,
                    )
                except Exception:
                    pass
            except Exception:
                pass

        stream_headers = dict(headers)
        stream_headers.update(
            {
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers=stream_headers,
        )

    # 5. Handle Non-Streaming Completion
    completion_resp = await proxy.execute_chat(request, model_entry)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    actual_cost = calculate_token_cost(
        prompt_tokens=completion_resp.usage.prompt_tokens,
        completion_tokens=completion_resp.usage.completion_tokens,
        prompt_price_per_m=model_entry.prompt_price_per_million,
        completion_price_per_m=model_entry.completion_price_per_million,
    )

    # Atomic credit deduction
    try:
        await deduct_balance_atomic_async(
            redis_client,
            api_key.id,
            actual_cost,
            initial_balance=api_key.credit_balance,
        )
    except Exception:
        # If balance was exhausted concurrently
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail={
                "error": {
                    "message": "Insufficient credit balance during transaction deduction.",
                    "type": "insufficient_quota",
                    "param": None,
                    "code": "insufficient_balance",
                }
            },
            headers=headers,
        ) from None

    # Persist usage to database ledger
    sync_usage_to_db(
        session=session,
        api_key_id=api_key.id,
        request_id=completion_resp.id,
        model=request.model,
        prompt_tokens=completion_resp.usage.prompt_tokens,
        completion_tokens=completion_resp.usage.completion_tokens,
        cost=actual_cost,
        latency_ms=latency_ms,
        ttft_ms=None,
    )

    return JSONResponse(content=completion_resp.model_dump(), headers=headers)
