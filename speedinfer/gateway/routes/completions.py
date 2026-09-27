"""OpenAI-compatible legacy completions endpoint.

Provides:
- POST /v1/completions
"""

import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
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
from speedinfer.gateway.schemas import CompletionRequest, CompletionResponse

router = APIRouter(prefix="/v1", tags=["Completions"])


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
    """Resolve ModelEntry from registry or database."""
    entry = registry.get_model(model_name)
    if entry is not None:
        return entry

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
    "/completions",
    response_model=CompletionResponse,
    summary="Create text completion",
    description="Generate continuations for provided prompt in OpenAI legacy format.",
)
async def create_completion(
    request: CompletionRequest,
    api_key: Annotated[ApiKey, Depends(get_authenticated_api_key)],
    registry: Annotated[ModelRegistry, Depends(get_model_registry)],
    proxy: Annotated[InferenceProxy, Depends(get_inference_proxy)],
    redis_client: Annotated[Any, Depends(get_async_redis)],
    session: Annotated[Session, Depends(get_session)],
) -> Any:
    """Process OpenAI-compatible text completion request."""
    # 1. Resolve model
    model_entry = _resolve_model_entry(request.model, registry, session)

    # 2. Rate limit check
    est_max_tokens = request.max_tokens or 16
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
    prompt_tokens = proxy.estimate_prompt_tokens(request.prompt)
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

    # 4. Execute completion
    completion_resp = await proxy.execute_completion(request, model_entry)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    actual_cost = calculate_token_cost(
        prompt_tokens=completion_resp.usage.prompt_tokens,
        completion_tokens=completion_resp.usage.completion_tokens,
        prompt_price_per_m=model_entry.prompt_price_per_million,
        completion_price_per_m=model_entry.completion_price_per_million,
    )

    # 5. Atomic balance deduction
    try:
        await deduct_balance_atomic_async(
            redis_client,
            api_key.id,
            actual_cost,
            initial_balance=api_key.credit_balance,
        )
    except Exception:
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

    # 6. Record usage in database
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
