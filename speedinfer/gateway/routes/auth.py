"""User authentication, verification, password management, and profile routes.

Provides:
- POST /v1/auth/register: User account registration, Turnstile check, and referral tracking.
- POST /v1/auth/login: User credential verification with Turnstile validation and JWT issuance.
- POST /v1/auth/verify-email: 6-digit OTP verification and referee credit bonus activation.
- POST /v1/auth/resend-code: Rate-limited verification OTP code resend.
- POST /v1/auth/forgot-password: Password reset security code dispatch.
- POST /v1/auth/reset-password: Password reset fulfillment with OTP validation.
- GET /v1/auth/me: Current authenticated user profile and aggregate balance inspection.
- GET /v1/auth/profile: Detailed user profile retrieval.
- PATCH /v1/auth/profile: User profile updates (display name, avatar URL, location, organization).
"""

import secrets
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from speedinfer.config import get_settings
from speedinfer.core.auth import DEFAULT_KEY_PERMISSIONS, generate_api_key
from speedinfer.core.email import send_password_reset_email, send_verification_email
from speedinfer.core.referrals import award_referee_bonus, hash_ip, process_registration_referral
from speedinfer.core.security import (
    DUMMY_PASSWORD_HASH,
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from speedinfer.core.turnstile import verify_turnstile_token
from speedinfer.database.models import (
    ApiKey,
    EmailVerificationCode,
    TrialCreditGrant,
    User,
    utc_now,
)
from speedinfer.database.session import get_session
from speedinfer.database.workspace import PolicyAcceptance
from speedinfer.gateway.schemas import (
    ApiKeyCreatedResponse,
    AuthResponse,
    ForgotPasswordRequest,
    ProfileUpdateRequest,
    ResendCodeRequest,
    ResetPasswordRequest,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
    VerifyEmailRequest,
)

router = APIRouter(prefix="/v1/auth", tags=["Authentication"])


def _calculate_user_balances(session: Session, user_id: int) -> tuple[float, float, float]:
    """Calculate user total, paid, and trial balance across active non-expired keys."""
    now = datetime.now(UTC)
    statement = select(
        func.coalesce(func.sum(ApiKey.credit_balance), 0.0),
        func.coalesce(func.sum(ApiKey.paid_balance), 0.0),
        func.coalesce(func.sum(ApiKey.trial_balance), 0.0),
    ).where(
        ApiKey.user_id == user_id,
        ApiKey.is_active == True,  # noqa: E712
        (ApiKey.expires_at == None) | (ApiKey.expires_at > now),  # noqa: E711
    )
    row = session.exec(statement).one()
    total = round(float(row[0]), 6)
    paid = round(float(row[1]), 6)
    trial = round(float(row[2]), 6)
    return total, paid, trial


def _calculate_user_balance(session: Session, user_id: int) -> float:
    """Calculate user balance across active non-expired keys via SQL aggregation."""
    total, _, _ = _calculate_user_balances(session, user_id)
    return total


def calculate_user_balance(session: Session, user_id: int) -> float:
    """Public helper to calculate user balance across active keys."""
    return _calculate_user_balance(session, user_id)


def _build_user_response(
    user: User,
    balance: float,
    paid_balance: float = 0.0,
    trial_balance: float = 0.0,
) -> UserResponse:
    """Construct canonical UserResponse from User model instance and balances."""
    return UserResponse(
        id=int(user.id) if user.id is not None else 0,
        email=user.email,
        name=user.name,
        is_active=user.is_active,
        is_admin=user.is_admin,
        is_verified=user.is_verified,
        referral_code=user.referral_code,
        avatar_url=user.avatar_url,
        location=user.location,
        address=user.location,
        organization=user.organization,
        company=user.organization,
        created_at=user.created_at,
        balance=balance,
        credit_balance=balance,
        paid_balance=paid_balance,
        trial_balance=trial_balance,
    )


def _extract_client_ip(request: Request) -> str | None:
    """Safely extract remote client IP address from request headers or transport."""
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip and cf_ip.strip():
        return cf_ip.strip()
    x_forwarded = request.headers.get("x-forwarded-for")
    if x_forwarded and x_forwarded.strip():
        return x_forwarded.split(",")[0].strip()
    x_real = request.headers.get("x-real-ip")
    if x_real and x_real.strip():
        return x_real.strip()
    if request.client:
        return request.client.host
    return None


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
    request: Request,
    session: Annotated[Session, Depends(get_session)],
) -> AuthResponse:
    """Register a new user account with trial credit balance and optional API key."""
    settings = get_settings()
    client_ip = _extract_client_ip(request)

    # 1. Turnstile security verification
    await verify_turnstile_token(
        payload.turnstile_token,
        remote_ip=client_ip,
        expected_action="register",
        settings=settings,
    )

    # 2. Check for existing email early
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

    # 3. Hash client IP with security pepper for anti-fraud device linkage
    pepper = (
        settings.api_key_pepper.get_secret_value()
        if hasattr(settings.api_key_pepper, "get_secret_value")
        else str(settings.api_key_pepper)
    )
    ip_hash = hash_ip(client_ip, pepper)

    # 4. Hash password and build User entity
    pwd_hash = hash_password(payload.password)
    user_name = payload.name.strip() if payload.name and payload.name.strip() else None
    user = User(
        email=payload.email,
        name=user_name,
        password_hash=pwd_hash,
        is_active=True,
        is_admin=False,
        is_verified=False,
        signup_ip_hash=ip_hash,
        device_fingerprint=payload.device_fingerprint,
    )

    api_key_resp: ApiKeyCreatedResponse | None = None
    try:
        session.add(user)
        session.flush()

        # 5. Process referral anti-fraud heuristics and attribution
        if payload.referral_code:
            process_registration_referral(
                session=session,
                new_user=user,
                referral_code=payload.referral_code,
                device_fingerprint=payload.device_fingerprint,
                signup_ip_hash=ip_hash,
                settings=settings,
            )

        if payload.accepted_policy_version:
            session.add(PolicyAcceptance(user_id=user.id, version=payload.accepted_policy_version))

        # 6. Optionally create initial default API key funded with trial credits
        if payload.create_api_key:
            raw_key, prefix, key_hash = generate_api_key(pepper=pepper)

            api_key_balance = min(payload.initial_balance, settings.trial_credit_balance)
            api_key = ApiKey(
                user_id=user.id,
                name="default",
                key_hash=key_hash,
                prefix=prefix,
                permissions=DEFAULT_KEY_PERMISSIONS,
                trial_balance=api_key_balance,
                paid_balance=0.0,
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
                paid_balance=api_key.paid_balance,
                trial_balance=api_key.trial_balance,
                rpm_limit=api_key.rpm_limit,
                tpm_limit=api_key.tpm_limit,
                permissions=api_key.permissions,
                created_at=api_key.created_at,
                expires_at=api_key.expires_at,
            )

        # 7. Issue 6-digit email verification OTP
        otp_code = f"{secrets.randbelow(900000) + 100000}"
        verification_code = EmailVerificationCode(
            email=user.email,
            code=otp_code,
            purpose="registration",
            expires_at=utc_now() + timedelta(minutes=15),
        )
        session.add(verification_code)
        send_verification_email(user.email, otp_code, settings=settings)

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

    # 8. Generate JWT access token
    access_token = create_access_token(data={"sub": str(user.id), "email": user.email})
    tot_bal, paid_bal, trial_bal = _calculate_user_balances(session, user.id)

    return AuthResponse(
        access_token=access_token,
        token_type="bearer",
        user=_build_user_response(user, tot_bal, paid_bal, trial_bal),
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
    request: Request,
    session: Annotated[Session, Depends(get_session)],
) -> AuthResponse:
    """Authenticate with email and password credentials."""
    settings = get_settings()
    client_ip = _extract_client_ip(request)

    # 1. Turnstile security verification
    await verify_turnstile_token(
        payload.turnstile_token,
        remote_ip=client_ip,
        expected_action="login",
        settings=settings,
    )

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
    tot_bal, paid_bal, trial_bal = _calculate_user_balances(session, user.id)

    return AuthResponse(
        access_token=access_token,
        token_type="bearer",
        user=_build_user_response(user, tot_bal, paid_bal, trial_bal),
        api_key=None,
    )


@router.post(
    "/verify-email",
    summary="Verify user email address using 6-digit OTP code",
    description=(
        "Validates OTP and upgrades account to verified status, awarding referral trial bonuses."
    ),
)
async def verify_email(
    payload: VerifyEmailRequest,
    session: Annotated[Session, Depends(get_session)],
) -> dict[str, Any]:
    """Verify email address with 6-digit code and disburse referee bonus if applicable."""
    now = utc_now()
    code_record = session.exec(
        select(EmailVerificationCode).where(
            EmailVerificationCode.email == payload.email,
            EmailVerificationCode.code == payload.code,
            EmailVerificationCode.purpose == "registration",
            EmailVerificationCode.is_used == False,  # noqa: E712
            EmailVerificationCode.expires_at > now,
        )
    ).first()

    if code_record is None:
        # Check if there is an unused code to increment attempt counter
        active_code = session.exec(
            select(EmailVerificationCode).where(
                EmailVerificationCode.email == payload.email,
                EmailVerificationCode.purpose == "registration",
                EmailVerificationCode.is_used == False,  # noqa: E712
                EmailVerificationCode.expires_at > now,
            )
        ).first()
        if active_code is not None:
            active_code.attempts += 1
            if active_code.attempts >= 5:
                active_code.is_used = True
            session.add(active_code)
            session.commit()

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": "Invalid or expired verification code.",
                    "type": "invalid_request_error",
                    "param": "code",
                    "code": "invalid_verification_code",
                }
            },
        )

    user = session.exec(select(User).where(User.email == payload.email)).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"message": "User not found.", "code": "user_not_found"}},
        )

    code_record.is_used = True
    user.is_verified = True
    user.email_verified_at = now
    session.add(code_record)
    session.add(user)

    # Award referee bonus (+ $5.00 extra credit) if user was invited by referral
    referee_bonus_granted = award_referee_bonus(session, user)

    session.commit()
    session.refresh(user)

    return {
        "message": "Email verified successfully.",
        "is_verified": True,
        "referee_bonus_granted": referee_bonus_granted,
    }


@router.post(
    "/resend-code",
    summary="Resend verification OTP code",
    description="Generates and emails a new 6-digit OTP code subject to a 60-second cooldown.",
)
async def resend_code(
    payload: ResendCodeRequest,
    session: Annotated[Session, Depends(get_session)],
) -> dict[str, str]:
    """Resend a new 6-digit OTP verification code."""
    settings = get_settings()
    now = utc_now()
    cutoff = now - timedelta(seconds=60)

    # Rate limiting cooldown check
    recent = session.exec(
        select(EmailVerificationCode).where(
            EmailVerificationCode.email == payload.email,
            EmailVerificationCode.purpose == payload.purpose,
            EmailVerificationCode.created_at > cutoff,
        )
    ).first()
    if recent is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": {
                    "message": "Please wait 60 seconds before requesting another code.",
                    "type": "rate_limit_error",
                    "param": "email",
                    "code": "cooldown_active",
                }
            },
        )

    # Invalidate old unused codes for this purpose
    old_codes = session.exec(
        select(EmailVerificationCode).where(
            EmailVerificationCode.email == payload.email,
            EmailVerificationCode.purpose == payload.purpose,
            EmailVerificationCode.is_used == False,  # noqa: E712
        )
    ).all()
    for c in old_codes:
        c.is_used = True
        session.add(c)

    new_code = f"{secrets.randbelow(900000) + 100000}"
    record = EmailVerificationCode(
        email=payload.email,
        code=new_code,
        purpose=payload.purpose,
        expires_at=now + timedelta(minutes=15),
    )
    session.add(record)
    session.commit()

    if payload.purpose == "registration":
        send_verification_email(payload.email, new_code, settings=settings)
    else:
        send_password_reset_email(payload.email, new_code, settings=settings)

    return {"message": "Verification code sent."}


@router.post(
    "/forgot-password",
    summary="Request a password reset OTP code",
    description=(
        "Sends a 6-digit reset code to the provided email without revealing account presence."
    ),
)
async def forgot_password(
    payload: ForgotPasswordRequest,
    request: Request,
    session: Annotated[Session, Depends(get_session)],
) -> dict[str, str]:
    """Dispatch password reset security code to registered user."""
    settings = get_settings()
    client_ip = _extract_client_ip(request)

    await verify_turnstile_token(
        payload.turnstile_token,
        remote_ip=client_ip,
        expected_action="forgot_password",
        settings=settings,
    )

    user = session.exec(select(User).where(User.email == payload.email)).first()
    if user is not None and user.is_active:
        now = utc_now()
        # Invalidate old reset codes
        old_codes = session.exec(
            select(EmailVerificationCode).where(
                EmailVerificationCode.email == payload.email,
                EmailVerificationCode.purpose == "password_reset",
                EmailVerificationCode.is_used == False,  # noqa: E712
            )
        ).all()
        for c in old_codes:
            c.is_used = True
            session.add(c)

        code = f"{secrets.randbelow(900000) + 100000}"
        session.add(
            EmailVerificationCode(
                email=user.email,
                code=code,
                purpose="password_reset",
                expires_at=now + timedelta(minutes=15),
            )
        )
        session.commit()
        send_password_reset_email(user.email, code, settings=settings)

    # Return identical generic message to prevent account enumeration
    return {"message": "If the email is registered, a password reset code has been sent."}


@router.post(
    "/reset-password",
    summary="Reset account password using 6-digit reset OTP",
    description=(
        "Updates password upon valid code verification and invalidates current reset codes."
    ),
)
async def reset_password(
    payload: ResetPasswordRequest,
    session: Annotated[Session, Depends(get_session)],
) -> dict[str, str]:
    """Reset user account password with validated 6-digit OTP code."""
    now = utc_now()
    code_record = session.exec(
        select(EmailVerificationCode).where(
            EmailVerificationCode.email == payload.email,
            EmailVerificationCode.code == payload.code,
            EmailVerificationCode.purpose == "password_reset",
            EmailVerificationCode.is_used == False,  # noqa: E712
            EmailVerificationCode.expires_at > now,
        )
    ).first()

    if code_record is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": "Invalid or expired reset code.",
                    "type": "invalid_request_error",
                    "param": "code",
                    "code": "invalid_reset_code",
                }
            },
        )

    user = session.exec(select(User).where(User.email == payload.email)).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"message": "User account not found.", "code": "user_not_found"}},
        )

    code_record.is_used = True
    user.password_hash = hash_password(payload.new_password)
    user.touch()
    session.add(code_record)
    session.add(user)
    session.commit()

    return {
        "message": ("Password has been reset successfully. Please sign in with your new password.")
    }


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
    total_balance, paid_balance, trial_balance = _calculate_user_balances(session, current_user.id)
    return _build_user_response(current_user, total_balance, paid_balance, trial_balance)


@router.get(
    "/profile",
    response_model=UserResponse,
    summary="Get current user profile settings",
    description=(
        "Returns profile details including display name, avatar, location, and organization."
    ),
)
async def get_profile(
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_session)],
) -> UserResponse:
    """Retrieve authenticated user profile and account details."""
    total_balance, paid_balance, trial_balance = _calculate_user_balances(session, current_user.id)
    return _build_user_response(current_user, total_balance, paid_balance, trial_balance)


@router.patch(
    "/profile",
    response_model=UserResponse,
    summary="Update authenticated user profile",
    description="Allows updating display name, avatar picture URL, address/location, and company.",
)
async def update_profile(
    payload: ProfileUpdateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_session)],
) -> UserResponse:
    """Update profile information for the authenticated user."""
    if payload.name is not None:
        current_user.name = payload.name.strip() if payload.name.strip() else None
    if payload.avatar_url is not None:
        clean_url = payload.avatar_url.strip()
        current_user.avatar_url = clean_url if clean_url else None
    if payload.location is not None:
        clean_loc = payload.location.strip()
        current_user.location = clean_loc if clean_loc else None
    if payload.organization is not None:
        clean_org = payload.organization.strip()
        current_user.organization = clean_org if clean_org else None

    current_user.touch()
    session.add(current_user)
    session.commit()
    session.refresh(current_user)

    total_balance, paid_balance, trial_balance = _calculate_user_balances(session, current_user.id)
    return _build_user_response(current_user, total_balance, paid_balance, trial_balance)
