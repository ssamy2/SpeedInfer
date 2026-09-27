"""OpenAI-compatible legacy completions endpoint.

Provides:
- POST /v1/completions
"""

import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlmodel import Session, select

from speedinfer.core.auth import require_scope
from speedinfer.core.inference_billing import reserve, settle
from speedinfer.core.metering import (
    calculate_token_cost,
    estimate_max_cost,
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
    api_key: Annotated[ApiKey, Depends(require_scope("completions"))],
    registry: Annotated[ModelRegistry, Depends(get_model_registry)],
    proxy: Annotated[InferenceProxy, Depends(get_inference_proxy)],
    redis_client: Annotated[Any, Depends(get_async_redis)],
    session: Annotated[Session, Depends(get_session)],
) -> Any:
    """Process OpenAI-compatible text completion request."""
    request.max_tokens = request.max_tokens or 16
    request.n = request.n or 1
    if request.stream:
        raise HTTPException(
            400, detail="Legacy streaming is unsupported; use /v1/chat/completions."
        )
    if isinstance(request.prompt, list):
        raise HTTPException(
            400, detail="Batch prompts are unsupported; send one prompt per request."
        )
    # 1. Resolve model
    model_entry = _resolve_model_entry(request.model, registry, session)

    # 2. Rate limit check
    est_max_tokens = request.max_tokens or 16
    rate_result = await async_check_rate_limit(
        redis_client,
        api_key=api_key,
        requested_tokens=est_max_tokens + proxy.estimate_prompt_tokens(request.prompt),
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
    estimated_cost = estimate_max_cost(
        prompt_tokens=model_entry.context_length,
        max_tokens=est_max_tokens,
        context_window=model_entry.context_length + est_max_tokens,
        prompt_price_per_m=model_entry.prompt_price_per_million,
        completion_price_per_m=model_entry.completion_price_per_million,
    )

    if est_max_tokens > model_entry.context_length:
        raise HTTPException(400, detail="max_tokens exceeds the model context limit.")
    hold_id = reserve(session, api_key.id, estimated_cost)

    start_time = time.perf_counter()

    try:
        completion_resp = await proxy.execute_completion(request, model_entry)
        usage = completion_resp.usage
        actual_cost = calculate_token_cost(
            usage.prompt_tokens,
            usage.completion_tokens,
            model_entry.prompt_price_per_million,
            model_entry.completion_price_per_million,
        )
        settle(
            session,
            hold_id,
            cost=actual_cost,
            model=request.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            success=True,
        )
        return JSONResponse(content=completion_resp.model_dump(), headers=headers)
    finally:
        settle(session, hold_id)
