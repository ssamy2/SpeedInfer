"""System and cluster health diagnostic endpoint.

Provides:
- GET /health
"""

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends

from speedinfer.engine.registry import ModelRegistry
from speedinfer.gateway.schemas import HealthResponse

router = APIRouter(tags=["Health"])


def get_model_registry() -> ModelRegistry:
    """Dependency provider for ModelRegistry."""
    from speedinfer.gateway.app import get_registry

    return get_registry()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and readiness probe",
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
                    "url": b.url,
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
