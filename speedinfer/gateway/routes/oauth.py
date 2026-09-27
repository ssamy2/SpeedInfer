"""Google and GitHub OAuth social authentication routes and callback handlers.

Provides:
- GET /v1/auth/oauth/google: Redirect user to Google OAuth consent screen.
- GET /v1/auth/oauth/google/callback: Exchange Google auth code, link user, return JWT.
- GET /v1/auth/oauth/github: Redirect user to GitHub OAuth authorization.
- GET /v1/auth/oauth/github/callback: Exchange GitHub code, retrieve profile, return JWT.
"""

import json
import secrets
from typing import Annotated, Any
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from sqlmodel import Session, select

from speedinfer.config import Settings, get_settings
from speedinfer.core.auth import generate_api_key
from speedinfer.core.referrals import award_referee_bonus, hash_ip, process_registration_referral
from speedinfer.core.security import create_access_token
from speedinfer.database.models import ApiKey, OAuthAccount, TrialCreditGrant, User
from speedinfer.database.session import get_session
from speedinfer.gateway.routes.auth import _extract_client_ip
from speedinfer.logging import get_logger

logger = get_logger("speedinfer.oauth")

router = APIRouter(prefix="/v1/auth/oauth", tags=["OAuth"])

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

GITHUB_AUTH_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"
GITHUB_EMAILS_URL = "https://api.github.com/user/emails"


def _secret_val(val: Any) -> str:
    """Safely extract plaintext from SecretStr or return as string."""
    return val.get_secret_value() if hasattr(val, "get_secret_value") and val else str(val or "")


def _provision_oauth_user(
    session: Session,
    provider: str,
    provider_user_id: str,
    email: str,
    name: str | None,
    avatar_url: str | None,
    referral_code: str | None,
    settings: Settings,
    device_fingerprint: str | None = None,
    signup_ip_hash: str | None = None,
) -> User:
    """Find or create User and OAuthAccount for authenticated social identity."""
    # Check if OAuthAccount already exists
    oauth_acc = session.exec(
        select(OAuthAccount).where(
            OAuthAccount.provider == provider,
            OAuthAccount.provider_user_id == provider_user_id,
        )
    ).first()

    if oauth_acc is not None:
        user = session.get(User, oauth_acc.user_id)
        if user is not None:
            if avatar_url and not user.avatar_url:
                user.avatar_url = avatar_url
                session.add(user)
                session.commit()
                session.refresh(user)
            return user

    # Check if user already exists by email
    user = session.exec(select(User).where(User.email == email.lower())).first()
    if user is not None:
        # Link existing user with OAuth provider
        new_oauth = OAuthAccount(
            user_id=user.id,
            provider=provider,
            provider_user_id=provider_user_id,
            provider_email=email.lower(),
        )
        if avatar_url and not user.avatar_url:
            user.avatar_url = avatar_url
        user.is_verified = True
        session.add(new_oauth)
        session.add(user)
        session.commit()
        session.refresh(user)
        return user

    # Brand new user registering via OAuth
    user = User(
        email=email.lower(),
        name=name,
        avatar_url=avatar_url,
        is_active=True,
        is_admin=False,
        is_verified=True,
        device_fingerprint=device_fingerprint,
        signup_ip_hash=signup_ip_hash,
    )
    session.add(user)
    session.flush()

    # Link OAuth account
    new_oauth = OAuthAccount(
        user_id=user.id,
        provider=provider,
        provider_user_id=provider_user_id,
        provider_email=email.lower(),
    )
    session.add(new_oauth)

    # Process referral if supplied
    if referral_code:
        process_registration_referral(
            session=session,
            new_user=user,
            referral_code=referral_code,
            device_fingerprint=device_fingerprint,
            signup_ip_hash=signup_ip_hash,
            settings=settings,
        )

    # Automatically provision default API key funded with trial credits
    pepper = _secret_val(settings.api_key_pepper)
    raw_key, prefix, key_hash = generate_api_key(pepper=pepper)
    trial_balance = settings.trial_credit_balance
    api_key = ApiKey(
        user_id=user.id,
        name="default",
        key_hash=key_hash,
        prefix=prefix,
        permissions="chat:completions,completions,models:read,usage:read",
        credit_balance=trial_balance,
        rpm_limit=60,
        tpm_limit=60_000,
        is_active=True,
    )
    session.add(api_key)
    session.add(TrialCreditGrant(user_id=user.id, amount=trial_balance))
    session.flush()

    # Crucial: Award referee bonus (+ $5.00 extra trial credit) upon verified OAuth registration
    award_referee_bonus(session=session, user=user, settings=settings)

    session.commit()
    session.refresh(user)
    return user


# =============================================================================
# Google OAuth
# =============================================================================
@router.get("/google", summary="Initiate Google OAuth authentication")
async def google_login(
    settings: Annotated[Settings, Depends(get_settings)],
    ref: str | None = None,
    fp: str | None = None,
) -> RedirectResponse:
    """Redirect client to Google OAuth 2.0 authorization endpoint."""
    if not settings.google_client_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth is not configured.",
        )

    state_data: dict[str, str] = {"csrf": secrets.token_urlsafe(16)}
    if ref:
        state_data["ref"] = ref.strip().upper()
    if fp:
        state_data["fp"] = fp.strip()
    state = json.dumps(state_data)

    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "online",
        "prompt": "select_account",
    }
    url = f"{GOOGLE_AUTH_URL}?{urlencode(params)}"
    return RedirectResponse(url=url, status_code=status.HTTP_302_FOUND)


@router.get("/google/callback", summary="Google OAuth callback handler")
async def google_callback(
    code: str,
    request: Request,
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    state: str | None = None,
) -> RedirectResponse:
    """Exchange authorization code for tokens, retrieve profile, and issue JWT."""
    secret = _secret_val(settings.google_client_secret)
    if not settings.google_client_id or not secret:
        raise HTTPException(status_code=503, detail="Google OAuth credentials unconfigured.")

    # 1. Exchange code for access token
    token_payload = {
        "client_id": settings.google_client_id,
        "client_secret": secret,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": settings.google_redirect_uri,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            token_res = await client.post(GOOGLE_TOKEN_URL, data=token_payload)
            token_res.raise_for_status()
            tokens = token_res.json()
            access_token = tokens.get("access_token")

            # 2. Retrieve user identity
            userinfo_res = await client.get(
                GOOGLE_USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            userinfo_res.raise_for_status()
            userinfo = userinfo_res.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.error("Google OAuth token exchange failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to authenticate with Google. Please try again.",
        ) from None

    google_sub = str(userinfo.get("sub", ""))
    email = str(userinfo.get("email", ""))
    name = userinfo.get("name")
    picture = userinfo.get("picture")

    if not google_sub or not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incomplete user profile returned by Google.",
        )

    referral_code = None
    device_fp = None
    if state:
        try:
            parsed_state = json.loads(state)
            referral_code = parsed_state.get("ref")
            device_fp = parsed_state.get("fp")
        except (json.JSONDecodeError, TypeError):
            pass

    client_ip = _extract_client_ip(request)
    pepper = _secret_val(settings.api_key_pepper)
    ip_hash = hash_ip(client_ip, pepper) if client_ip else None

    user = _provision_oauth_user(
        session=session,
        provider="google",
        provider_user_id=google_sub,
        email=email,
        name=name,
        avatar_url=picture,
        referral_code=referral_code,
        settings=settings,
        device_fingerprint=device_fp,
        signup_ip_hash=ip_hash,
    )

    jwt_token = create_access_token(data={"sub": str(user.id), "email": user.email})
    target_url = f"{settings.public_base_url.rstrip('/')}/?token={jwt_token}"
    return RedirectResponse(url=target_url, status_code=status.HTTP_302_FOUND)


# =============================================================================
# GitHub OAuth
# =============================================================================
@router.get("/github", summary="Initiate GitHub OAuth authentication")
async def github_login(
    settings: Annotated[Settings, Depends(get_settings)],
    ref: str | None = None,
    fp: str | None = None,
) -> RedirectResponse:
    """Redirect client to GitHub OAuth authorization endpoint."""
    if not settings.github_client_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GitHub OAuth is not configured.",
        )

    state_data: dict[str, str] = {"csrf": secrets.token_urlsafe(16)}
    if ref:
        state_data["ref"] = ref.strip().upper()
    if fp:
        state_data["fp"] = fp.strip()
    state = json.dumps(state_data)

    params = {
        "client_id": settings.github_client_id,
        "redirect_uri": settings.github_redirect_uri,
        "scope": "read:user user:email",
        "state": state,
    }
    url = f"{GITHUB_AUTH_URL}?{urlencode(params)}"
    return RedirectResponse(url=url, status_code=status.HTTP_302_FOUND)


@router.get("/github/callback", summary="GitHub OAuth callback handler")
async def github_callback(
    code: str,
    request: Request,
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    state: str | None = None,
) -> RedirectResponse:
    """Exchange GitHub authorization code, retrieve user identity, and issue JWT."""
    secret = _secret_val(settings.github_client_secret)
    if not settings.github_client_id or not secret:
        raise HTTPException(status_code=503, detail="GitHub OAuth credentials unconfigured.")

    token_payload = {
        "client_id": settings.github_client_id,
        "client_secret": secret,
        "code": code,
        "redirect_uri": settings.github_redirect_uri,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            token_res = await client.post(
                GITHUB_TOKEN_URL,
                data=token_payload,
                headers={"Accept": "application/json"},
            )
            token_res.raise_for_status()
            tokens = token_res.json()
            access_token = tokens.get("access_token")

            # Retrieve GitHub profile
            profile_res = await client.get(
                GITHUB_USER_URL,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "User-Agent": "SpeedInfer-OAuth",
                },
            )
            profile_res.raise_for_status()
            profile = profile_res.json()

            # Retrieve email address if not public in profile
            email = profile.get("email")
            if not email:
                emails_res = await client.get(
                    GITHUB_EMAILS_URL,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "User-Agent": "SpeedInfer-OAuth",
                    },
                )
                emails_res.raise_for_status()
                emails_list = emails_res.json()
                for em in emails_list:
                    if em.get("primary") and em.get("verified"):
                        email = em.get("email")
                        break
                if not email and emails_list:
                    email = emails_list[0].get("email")
    except (httpx.HTTPError, ValueError) as exc:
        logger.error("GitHub OAuth token exchange failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to authenticate with GitHub. Please try again.",
        ) from None

    github_id = str(profile.get("id", ""))
    if not github_id or not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to obtain verified email address from GitHub account.",
        )

    name = profile.get("name") or profile.get("login")
    avatar_url = profile.get("avatar_url")

    referral_code = None
    device_fp = None
    if state:
        try:
            parsed_state = json.loads(state)
            referral_code = parsed_state.get("ref")
            device_fp = parsed_state.get("fp")
        except (json.JSONDecodeError, TypeError):
            pass

    client_ip = _extract_client_ip(request)
    pepper = _secret_val(settings.api_key_pepper)
    ip_hash = hash_ip(client_ip, pepper) if client_ip else None

    user = _provision_oauth_user(
        session=session,
        provider="github",
        provider_user_id=github_id,
        email=email,
        name=name,
        avatar_url=avatar_url,
        referral_code=referral_code,
        settings=settings,
        device_fingerprint=device_fp,
        signup_ip_hash=ip_hash,
    )

    jwt_token = create_access_token(data={"sub": str(user.id), "email": user.email})
    target_url = f"{settings.public_base_url.rstrip('/')}/?token={jwt_token}"
    return RedirectResponse(url=target_url, status_code=status.HTTP_302_FOUND)
