"""API Key management endpoints.

Provides:
- POST /v1/keys: Generate a new API key for the authenticated user.
- GET /v1/keys: List all API keys belonging to the authenticated user.
- DELETE /v1/keys/{key_id}: Revoke an API key belonging to the authenticated user.
"""

from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from speedinfer.config import get_settings
from speedinfer.core.auth import generate_api_key
from speedinfer.core.security import get_current_user
from speedinfer.database.models import ApiKey, TrialCreditGrant, User
from speedinfer.database.session import get_session
from speedinfer.gateway.schemas import (
    ApiKeyCreatedResponse,
    ApiKeyCreateRequest,
    ApiKeyDeleteResponse,
    ApiKeyItemResponse,
    ApiKeyListResponse,
)

router = APIRouter(prefix="/v1/keys", tags=["API Keys"])


def _resolve_key_status(key: ApiKey) -> str:
    """Determine the lifecycle status of an API key."""
    if not key.is_active:
        return "revoked"
    if key.is_expired():
        return "expired"
    return "active"


@router.post(
    "",
    response_model=ApiKeyCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate a new API key",
    description=(
        "Creates a new API key for the authenticated user, returning plaintext secret once."
    ),
)
@router.post(
    "/",
    response_model=ApiKeyCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,
)
async def create_key(
    payload: ApiKeyCreateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_session)],
) -> ApiKeyCreatedResponse:
    """Generate and persist a new API key for the authenticated user."""
    # Prevent privilege escalation: non-admin users cannot grant admin scope
    raw_permissions = payload.permissions or "chat:completions,completions,models:read,usage:read"
    requested_scopes = {s.strip() for s in raw_permissions.split(",") if s.strip()}
    if not current_user.is_admin and ("admin" in requested_scopes or "*" in requested_scopes):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": {
                    "message": "Only administrators can grant 'admin' permission scope.",
                    "type": "insufficient_scope",
                    "param": "permissions",
                    "code": "insufficient_scope",
                }
            },
        )

    settings = get_settings()
    pepper = (
        settings.api_key_pepper.get_secret_value()
        if hasattr(settings.api_key_pepper, "get_secret_value")
        else str(settings.api_key_pepper)
    )

    raw_key, prefix, key_hash = generate_api_key(pepper=pepper)

    trial_amount = 0.0
    if session.get(TrialCreditGrant, current_user.id) is None:
        prior_key = session.exec(select(ApiKey.id).where(ApiKey.user_id == current_user.id)).first()
        trial_amount = settings.trial_credit_balance if prior_key is None else 0.0
        session.add(TrialCreditGrant(user_id=current_user.id, amount=trial_amount))
        try:
            session.flush()
        except IntegrityError:
            session.rollback()
            raise HTTPException(409, "Another key creation is in progress. Please retry.") from None

    expires_at = None
    if payload.expires_in_days is not None:
        expires_at = datetime.now(UTC) + timedelta(days=payload.expires_in_days)

    init_paid = (
        payload.credit_balance
        if current_user.is_admin and payload.credit_balance is not None
        else 0.0
    )
    init_trial = trial_amount

    api_key = ApiKey(
        user_id=current_user.id,
        name=payload.name,
        key_hash=key_hash,
        prefix=prefix,
        permissions=",".join(sorted(requested_scopes)),
        trial_balance=init_trial,
        paid_balance=init_paid,
        credit_balance=round(init_trial + init_paid, 6),
        rpm_limit=payload.rpm_limit or 60,
        tpm_limit=payload.tpm_limit or 60_000,
        is_active=True,
        expires_at=expires_at,
    )
    session.add(api_key)
    session.commit()
    session.refresh(api_key)

    return ApiKeyCreatedResponse(
        id=api_key.id,
        name=api_key.name,
        key=raw_key,
        prefix=api_key.prefix,
        status="active",
        is_active=api_key.is_active,
        credit_balance=api_key.credit_balance,
        paid_balance=api_key.paid_balance,
        trial_balance=api_key.trial_balance,
        rpm_limit=api_key.rpm_limit,
        tpm_limit=api_key.tpm_limit,
        permissions=api_key.permissions,
        created_at=api_key.created_at,
        expires_at=api_key.expires_at,
    )


@router.get(
    "",
    response_model=ApiKeyListResponse,
    summary="List API keys",
    description="Lists all API keys belonging to the authenticated user with masked prefixes.",
)
@router.get(
    "/",
    response_model=ApiKeyListResponse,
    include_in_schema=False,
)
async def list_keys(
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_session)],
    limit: Annotated[
        int,
        Query(ge=1, le=1000, description="Maximum number of keys to return."),
    ] = 100,
    offset: Annotated[
        int,
        Query(ge=0, description="Offset for pagination."),
    ] = 0,
) -> ApiKeyListResponse:
    """Retrieve all API keys belonging to the authenticated user with pagination support."""
    total_count = session.exec(
        select(func.count(ApiKey.id)).where(ApiKey.user_id == current_user.id)
    ).one()

    statement = (
        select(ApiKey)
        .where(ApiKey.user_id == current_user.id)
        .order_by(ApiKey.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    keys = session.exec(statement).all()

    items = [
        ApiKeyItemResponse(
            id=k.id,
            name=k.name,
            prefix=k.prefix,
            masked_key=f"{k.prefix}...",
            status=_resolve_key_status(k),
            is_active=k.is_active,
            credit_balance=k.credit_balance,
            paid_balance=getattr(k, "paid_balance", 0.0),
            trial_balance=getattr(k, "trial_balance", 0.0),
            rpm_limit=k.rpm_limit,
            tpm_limit=k.tpm_limit,
            permissions=k.permissions,
            created_at=k.created_at,
            last_used_at=k.last_used_at,
            expires_at=k.expires_at,
        )
        for k in keys
    ]

    return ApiKeyListResponse(object="list", data=items, total=total_count)


@router.delete(
    "/{key_id}",
    response_model=ApiKeyDeleteResponse,
    summary="Revoke an API key",
    description="Revokes an API key belonging to the authenticated user, disabling its use.",
)
@router.delete(
    "/{key_id}/",
    response_model=ApiKeyDeleteResponse,
    include_in_schema=False,
)
async def revoke_key(
    key_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_session)],
    permanent: Annotated[
        bool,
        Query(description="If true, permanently remove record from database."),
    ] = False,
) -> ApiKeyDeleteResponse:
    """Revoke or delete an API key belonging to the authenticated user."""
    statement = select(ApiKey).where(ApiKey.id == key_id, ApiKey.user_id == current_user.id)
    api_key = session.exec(statement).first()

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "message": f"API key with ID {key_id} not found.",
                    "type": "invalid_request_error",
                    "param": "key_id",
                    "code": "key_not_found",
                }
            },
        )

    if permanent:
        session.delete(api_key)
        session.commit()
        return ApiKeyDeleteResponse(
            id=key_id,
            deleted=True,
            status="deleted",
            message=f"API key {key_id} permanently deleted.",
        )

    api_key.is_active = False
    session.add(api_key)
    session.commit()
    session.refresh(api_key)

    return ApiKeyDeleteResponse(
        id=api_key.id,
        deleted=True,
        status="revoked",
        message=f"API key '{api_key.prefix}...' successfully revoked.",
    )
