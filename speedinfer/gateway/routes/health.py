"""System and cluster health diagnostic endpoint.

Provides:
- GET /health
"""

from datetime import UTC, datetime
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlmodel import Session, text

from speedinfer.config import get_settings
from speedinfer.database.session import get_session
from speedinfer.engine.registry import ModelRegistry
from speedinfer.gateway.redis import get_async_redis
from speedinfer.gateway.schemas import HealthResponse

router = APIRouter(tags=["Health"])


def get_model_registry() -> ModelRegistry:
    """Dependency provider for ModelRegistry."""
    from speedinfer.gateway.app import get_registry

    return get_registry()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Gateway liveness (not GPU readiness)",
    description="Reports gateway status and the model/worker state known to the registry.",
)
async def health_check(
    registry: Annotated[ModelRegistry, Depends(get_model_registry)],
) -> HealthResponse:
    """Probe gateway operational readiness and worker connectivity."""
    registered_models = registry.list_models()
    model_names = [m.name for m in registered_models]

    worker_info: list[dict[str, Any]] = []
    for m in registered_models:
        for b in m.backends:
            worker_info.append(
                {
                    "model": m.name,
                    "worker_id": b.worker_id,
                    "status": str(b.status),
                    "health": str(b.health),
                    "active_requests": b.active_requests,
                }
            )

    return HealthResponse(
        status="healthy",
        service="SpeedInfer",
        version="0.1.0",
        timestamp=datetime.now(UTC).isoformat(),
        models=model_names,
        workers=worker_info,
        # This gateway does not query a GPU telemetry source. Returning an explicit
        # unknown value avoids presenting configured capacity as measured hardware.
        gpu={"status": "not_reported"},
    )


@router.get("/ready", summary="Live inference readiness", tags=["Health"])
async def readiness(session: Annotated[Session, Depends(get_session)]):
    """Check the actual dependencies; never claim GPU throughput or expose worker URLs."""
    checks = {"database": False, "redis": False, "inference": False}
    try:
        session.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception:
        session.rollback()
    try:
        redis = await get_async_redis()
        checks["redis"] = bool(await redis.ping())
    except Exception:
        pass
    settings = get_settings()
    try:
        headers = (
            {"Authorization": f"Bearer {settings.vllm_api_key.get_secret_value()}"}
            if settings.vllm_api_key
            else {}
        )
        base_url = settings.vllm_base_url.rstrip("/")
        models_url = f"{base_url}/models" if base_url.endswith("/v1") else f"{base_url}/v1/models"
        async with httpx.AsyncClient(timeout=5.0, headers=headers) as client:
            response = await client.get(models_url)
            response.raise_for_status()
            data_items = response.json().get("data", [])
            checks["inference"] = any(
                m.get("id") == settings.default_model
                or m.get("id")
                in {
                    "qwen/qwen-2.5-7b-instruct",
                    "meta-llama/llama-3.2-1b-instruct",
                }
                for m in data_items
            )
    except Exception:
        pass
    ready = all(checks.values())
    return JSONResponse(
        {"status": "ready" if ready else "not_ready", "checks": checks},
        status_code=200 if ready else 503,
    )
