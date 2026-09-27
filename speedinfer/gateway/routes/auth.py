"""User authentication routes.

Provides:
- POST /v1/auth/register: User account registration, initial balance, and optional default API key.
- POST /v1/auth/login: User credential verification and JWT access token issuance.
- GET /v1/auth/me: Current authenticated user profile and aggregate balance inspection.
"""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from speedinfer.config import get_settings
from speedinfer.core.auth import generate_api_key
from speedinfer.core.security import (
    DUMMY_PASSWORD_HASH,
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from speedinfer.database.models import ApiKey, TrialCreditGrant, User
from speedinfer.database.session import get_session
from speedinfer.database.workspace import PolicyAcceptance
from speedinfer.gateway.schemas import (
    ApiKeyCreatedResponse,
    AuthResponse,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
)

router = APIRouter(prefix="/v1/auth", tags=["Authentication"])


def _calculate_user_balance(session: Session, user_id: int) -> float:
    """Calculate user balance across active non-expired keys via SQL aggregation."""
    now = datetime.now(UTC)
    statement = select(func.coalesce(func.sum(ApiKey.credit_balance), 0.0)).where(
        ApiKey.user_id == user_id,
        ApiKey.is_active == True,  # noqa: E712
        (ApiKey.expires_at == None) | (ApiKey.expires_at > now),  # noqa: E711
    )
    total = session.exec(statement).one()
    return round(float(total), 6)


@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
    description="Registers a new user, assigns initial trial credits, and returns JWT credentials.",
)
@router.post(
    "/register/",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,
)
async def register(
    payload: UserRegisterRequest,
    session: Annotated[Session, Depends(get_session)],
) -> AuthResponse:
    """Register a new user account with trial credit balance and optional API key."""
    # 1. Check for existing email early
    existing_user = session.exec(select(User).where(User.email == payload.email)).first()
    if existing_user is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": f"User with email '{payload.email}' already exists.",
                    "type": "invalid_request_error",
                    "param": "email",
                    "code": "email_already_exists",
                }
            },
        )

    # 2. Hash password and build User entity
    pwd_hash = hash_password(payload.password)
    user_name = payload.name.strip() if payload.name and payload.name.strip() else None
    user = User(
        email=payload.email,
        name=user_name,
        password_hash=pwd_hash,
        is_active=True,
        is_admin=False,
    )

    api_key_resp: ApiKeyCreatedResponse | None = None
    try:
        session.add(user)
        session.flush()

        if payload.accepted_policy_version:
            session.add(PolicyAcceptance(user_id=user.id, version=payload.accepted_policy_version))

        # 3. Optionally create initial default API key funded with trial credits
        if payload.create_api_key:
            settings = get_settings()
            pepper = (
                settings.api_key_pepper.get_secret_value()
                if hasattr(settings.api_key_pepper, "get_secret_value")
                else str(settings.api_key_pepper)
            )
            raw_key, prefix, key_hash = generate_api_key(pepper=pepper)

            api_key_balance = min(payload.initial_balance, settings.trial_credit_balance)
            api_key = ApiKey(
                user_id=user.id,
                name="default",
                key_hash=key_hash,
                prefix=prefix,
                permissions="chat:completions,completions,models:read,usage:read",
                credit_balance=api_key_balance,
                rpm_limit=60,
                tpm_limit=60_000,
                is_active=True,
            )
            session.add(api_key)
            session.add(TrialCreditGrant(user_id=user.id, amount=api_key_balance))
            session.flush()
            session.refresh(api_key)

            api_key_resp = ApiKeyCreatedResponse(
                id=api_key.id,
                name=api_key.name,
                key=raw_key,
                prefix=api_key.prefix,
                status="active",
                is_active=api_key.is_active,
                credit_balance=api_key.credit_balance,
                rpm_limit=api_key.rpm_limit,
                tpm_limit=api_key.tpm_limit,
                permissions=api_key.permissions,
                created_at=api_key.created_at,
                expires_at=api_key.expires_at,
            )

        session.commit()
        session.refresh(user)
    except IntegrityError:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": f"User with email '{payload.email}' already exists.",
                    "type": "invalid_request_error",
                    "param": "email",
                    "code": "email_already_exists",
                }
            },
        ) from None

    # 4. Generate JWT access token
    access_token = create_access_token(data={"sub": str(user.id), "email": user.email})
    current_balance = _calculate_user_balance(session, user.id)

    return AuthResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            is_active=user.is_active,
            is_admin=user.is_admin,
            created_at=user.created_at,
            balance=current_balance,
            credit_balance=current_balance,
        ),
        api_key=api_key_resp,
    )


@router.post(
    "/login",
    response_model=AuthResponse,
    summary="Authenticate user and obtain JWT token",
    description="Authenticates email and password credentials, returning a JWT access token.",
)
@router.post(
    "/login/",
    response_model=AuthResponse,
    summary="Authenticate user and obtain JWT token",
    include_in_schema=False,
)
async def login(
    payload: UserLoginRequest,
    session: Annotated[Session, Depends(get_session)],
) -> AuthResponse:
    """Authenticate with email and password credentials."""
    user = session.exec(select(User).where(User.email == payload.email)).first()
    if user is None or not user.password_hash:
        # Mitigate timing-based user enumeration via dummy constant-time verification
        verify_password(payload.password, DUMMY_PASSWORD_HASH)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "Invalid email or password.",
                    "type": "authentication_error",
                    "param": None,
                    "code": "invalid_credentials",
                }
            },
        )

    if not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "Invalid email or password.",
                    "type": "authentication_error",
                    "param": None,
                    "code": "invalid_credentials",
                }
            },
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "User account is deactivated.",
                    "type": "authentication_error",
                    "param": None,
                    "code": "user_inactive",
                }
            },
        )

    access_token = create_access_token(data={"sub": str(user.id), "email": user.email})
    total_balance = _calculate_user_balance(session, user.id)

    return AuthResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            is_active=user.is_active,
            is_admin=user.is_admin,
            created_at=user.created_at,
            balance=total_balance,
            credit_balance=total_balance,
        ),
        api_key=None,
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user details",
    description=(
        "Returns profile information and aggregated credit balance of the authenticated user."
    ),
)
@router.get(
    "/me/",
    response_model=UserResponse,
    include_in_schema=False,
)
async def get_me(
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_session)],
) -> UserResponse:
    """Retrieve authenticated user details and active credit balance."""
    total_balance = _calculate_user_balance(session, current_user.id)
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        name=current_user.name,
        is_active=current_user.is_active,
        is_admin=current_user.is_admin,
        created_at=current_user.created_at,
        balance=total_balance,
        credit_balance=total_balance,
    )
