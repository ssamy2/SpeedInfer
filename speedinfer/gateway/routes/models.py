"""OpenAI-compatible models catalog endpoint.

Provides:
- GET /v1/models
- GET /v1/models/{model_id}
"""

import time
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlmodel import Session, select

from speedinfer.core.auth import authenticate_api_key, check_scope_permission
from speedinfer.core.security import get_current_user
from speedinfer.database.models import ApiKey, ModelVersion, User
from speedinfer.database.session import get_session
from speedinfer.engine.registry import ModelRegistry
from speedinfer.gateway.schemas import ModelListResponse, ModelObject

router = APIRouter(prefix="/v1", tags=["Models"])


def get_model_registry() -> ModelRegistry:
    """Dependency provider for the process-wide ModelRegistry singleton."""
    from speedinfer.gateway.app import get_registry

    return get_registry()


def get_caller_auth(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    session: Annotated[Session, Depends(get_session)] = None,
) -> ApiKey | User:
    """Authenticate either via API Key (Bearer sk-speedinfer-...) or JWT access token."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "Missing Authorization header.",
                    "type": "authentication_error",
                    "param": None,
                    "code": "missing_authorization",
                }
            },
        )
    parts = authorization.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "Invalid Authorization header format. Expected 'Bearer <token>'.",
                    "type": "authentication_error",
                    "param": None,
                    "code": "invalid_header",
                }
            },
        )
    token = parts[1]
    if token.startswith("sk-speedinfer-"):
        key = authenticate_api_key(authorization, session)
        if not check_scope_permission(key, "models:read"):
            raise HTTPException(403, detail="Missing models:read permission.")
        return key

    # Decode and authenticate user via JWT
    return get_current_user(authorization=authorization, session=session)


@router.get(
    "/models",
    response_model=ModelListResponse,
    summary="List available models",
    description="Returns a list of currently active and warm models available for inference.",
)
async def list_models(
    caller: Annotated[ApiKey | User, Depends(get_caller_auth)],
    registry: Annotated[ModelRegistry, Depends(get_model_registry)],
    session: Annotated[Session, Depends(get_session)],
) -> ModelListResponse:
    """Retrieve catalog of active models."""
    registered = registry.list_models()
    models_data: list[ModelObject] = []

    for m in registered:
        models_data.append(
            ModelObject(
                id=m.name,
                object="model",
                created=int(time.time()),
                owned_by="speedinfer",
                root=m.name,
                context_length=m.context_length,
                prompt_price_per_million=m.prompt_price_per_million,
                completion_price_per_million=m.completion_price_per_million,
            )
        )

    # Also query DB for any additional active models
    db_models = session.exec(
        select(ModelVersion).where(ModelVersion.lifecycle_status.in_(["active", "production"]))
    ).all()
    for db_m in db_models:
        if not any(m.id == db_m.name for m in models_data):
            models_data.append(
                ModelObject(
                    id=db_m.name,
                    object="model",
                    created=int(db_m.created_at.timestamp())
                    if db_m.created_at
                    else int(time.time()),
                    owned_by="speedinfer",
                    root=db_m.name,
                    context_length=db_m.context_length,
                    prompt_price_per_million=db_m.prompt_price_per_million,
                    completion_price_per_million=db_m.completion_price_per_million,
                )
            )

    return ModelListResponse(object="list", data=models_data)


@router.get(
    "/models/{model_id:path}",
    response_model=ModelObject,
    summary="Retrieve model details",
    description="Returns metadata about a specific model if active.",
)
async def get_model(
    model_id: str,
    caller: Annotated[ApiKey | User, Depends(get_caller_auth)],
    registry: Annotated[ModelRegistry, Depends(get_model_registry)],
    session: Annotated[Session, Depends(get_session)],
) -> ModelObject:
    """Retrieve a single model by identifier."""
    entry = registry.get_model(model_id)
    if entry is not None:
        return ModelObject(
            id=entry.name,
            object="model",
            created=int(time.time()),
            owned_by="speedinfer",
            root=entry.name,
            context_length=entry.context_length,
            prompt_price_per_million=entry.prompt_price_per_million,
            completion_price_per_million=entry.completion_price_per_million,
        )

    db_model = session.exec(select(ModelVersion).where(ModelVersion.name == model_id)).first()
    if db_model is not None:
        return ModelObject(
            id=db_model.name,
            object="model",
            created=int(db_model.created_at.timestamp())
            if db_model.created_at
            else int(time.time()),
            owned_by="speedinfer",
            root=db_model.name,
            context_length=db_model.context_length,
            prompt_price_per_million=db_model.prompt_price_per_million,
            completion_price_per_million=db_model.completion_price_per_million,
        )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={
            "error": {
                "message": (
                    f"The model '{model_id}' does not exist or you do not have access to it."
                ),
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found",
            }
        },
    )
