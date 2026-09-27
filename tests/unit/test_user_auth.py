"""Unit tests for user authentication, password hashing, and JWT security utilities.

Verifies:
- PBKDF2-HMAC-SHA256 password hashing with 100,000+ iterations.
- Constant-time password verification and resilience to malformed inputs.
- JWT access token generation, decoding, expiration handling, and signature validation.
- FastAPI get_current_user dependency under various positive and negative scenarios.
- Backward compatibility of the User model with nullable password_hash.
"""

from datetime import timedelta

import pytest
from fastapi import HTTPException
from sqlmodel import Session, select

from speedinfer.core.security import (
    PBKDF2_ITERATIONS,
    TokenError,
    TokenExpiredError,
    create_access_token,
    decode_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from speedinfer.database.models import User


def test_hash_password_format_and_entropy() -> None:
    """Verify password hash formatting, iteration count, and unique random salting."""
    password = "SuperSecretPassword123!"
    hash1 = hash_password(password)
    hash2 = hash_password(password)

    # Validate structure: pbkdf2_sha256$<iterations>$<salt>$<hash>
    parts1 = hash1.split("$")
    assert len(parts1) == 4
    assert parts1[0] == "pbkdf2_sha256"
    assert int(parts1[1]) >= PBKDF2_ITERATIONS
    assert len(parts1[2]) == 32  # 16 bytes hex-encoded = 32 chars
    assert len(parts1[3]) == 64  # SHA-256 hex = 64 chars

    # Salts must be random, so two hashes of identical password must differ
    assert hash1 != hash2
    assert parts1[2] != hash2.split("$")[2]


def test_verify_password_correct() -> None:
    """Verify correct password matches its PBKDF2 hash."""
    password = "ValidPassword_2026!"
    hashed = hash_password(password)
    assert verify_password(password, hashed) is True


def test_verify_password_incorrect() -> None:
    """Verify wrong password fails verification."""
    password = "CorrectPassword123"
    wrong_password = "WrongPassword456"
    hashed = hash_password(password)
    assert verify_password(wrong_password, hashed) is False


@pytest.mark.parametrize(
    ("plain", "stored"),
    [
        ("", "pbkdf2_sha256$100000$abcd$ef01"),
        ("password", ""),
        ("password", None),
        ("", None),
        ("password", "invalid_format_string"),
        ("password", "md5$100000$salt$hash"),
        ("password", "pbkdf2_sha256$not_an_int$salt$hash"),
        ("password", "pbkdf2_sha256$100000$invalid_hex$hash"),
        ("password", "pbkdf2_sha256$100000"),
    ],
)
def test_verify_password_edge_cases(plain: str, stored: str | None) -> None:
    """Verify verify_password safely handles malformed, empty, and invalid hashes."""
    assert verify_password(plain, stored) is False


def test_create_and_decode_access_token() -> None:
    """Verify JWT access token creation and claim preservation."""
    claims = {"sub": "123", "email": "engineer@speedinfer.local", "role": "developer"}
    token = create_access_token(claims)
    decoded = decode_access_token(token)

    assert decoded["sub"] == "123"
    assert decoded["email"] == "engineer@speedinfer.local"
    assert decoded["role"] == "developer"
    assert "exp" in decoded
    assert "iat" in decoded


def test_access_token_expiration() -> None:
    """Verify decoding an expired access token raises TokenExpiredError."""
    claims = {"sub": "456", "email": "expired@speedinfer.local"}
    expired_token = create_access_token(claims, expires_delta=timedelta(seconds=-60))

    with pytest.raises(TokenExpiredError, match="expired"):
        decode_access_token(expired_token)


def test_access_token_tampered() -> None:
    """Verify tampering with a JWT token signature or payload causes TokenError."""
    claims = {"sub": "789", "email": "valid@speedinfer.local"}
    token = create_access_token(claims)
    parts = token.split(".")
    # Mutate the payload portion
    tampered_token = f"{parts[0]}.eyJhZG1pbiI6dHJ1ZX0.{parts[2]}"

    with pytest.raises(TokenError):
        decode_access_token(tampered_token)


def test_access_token_garbage_string() -> None:
    """Verify non-JWT strings raise TokenError."""
    with pytest.raises(TokenError):
        decode_access_token("this.is.not.a.valid.jwt.token")


def test_get_current_user_success(db_session: Session) -> None:
    """Verify get_current_user extracts active user from valid JWT Bearer header."""
    user = User(
        email="auth_success@speedinfer.local",
        name="Auth Tester",
        password_hash=hash_password("mypassword123"),
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    token = create_access_token({"sub": str(user.id), "email": user.email})
    auth_header = f"Bearer {token}"

    resolved_user = get_current_user(authorization=auth_header, session=db_session)
    assert resolved_user.id == user.id
    assert resolved_user.email == user.email


def test_get_current_user_missing_header(db_session: Session) -> None:
    """Verify missing Authorization header raises HTTP 401."""
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(authorization=None, session=db_session)
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail["error"]["code"] == "missing_token"


def test_get_current_user_invalid_header_scheme(db_session: Session) -> None:
    """Verify non-Bearer Authorization header raises HTTP 401."""
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(authorization="Basic dXNlcjpwYXNz", session=db_session)
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail["error"]["code"] == "invalid_header"


def test_get_current_user_expired_token(db_session: Session) -> None:
    """Verify expired JWT in header raises HTTP 401 token_expired."""
    expired_token = create_access_token({"sub": "1"}, expires_delta=timedelta(seconds=-10))
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(authorization=f"Bearer {expired_token}", session=db_session)
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail["error"]["code"] == "token_expired"


def test_get_current_user_invalid_token_string(db_session: Session) -> None:
    """Verify malformed JWT in header raises HTTP 401 invalid_token."""
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(authorization="Bearer corrupt-token-payload", session=db_session)
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail["error"]["code"] == "invalid_token"


def test_get_current_user_nonexistent_user(db_session: Session) -> None:
    """Verify valid token referencing deleted or nonexistent user raises HTTP 401 user_not_found."""
    token = create_access_token({"sub": "999999", "email": "nonexistent@speedinfer.local"})
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(authorization=f"Bearer {token}", session=db_session)
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail["error"]["code"] == "user_not_found"


def test_get_current_user_inactive_user(db_session: Session) -> None:
    """Verify deactivated user account raises HTTP 401 user_inactive."""
    user = User(
        email="inactive@speedinfer.local",
        name="Deactivated User",
        password_hash=hash_password("pwd"),
        is_active=False,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    token = create_access_token({"sub": str(user.id), "email": user.email})
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(authorization=f"Bearer {token}", session=db_session)
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail["error"]["code"] == "user_inactive"


def test_get_current_user_invalid_sub_format(db_session: Session) -> None:
    """Verify non-integer subject claim raises HTTP 401 invalid_token."""
    token = create_access_token({"sub": "not-a-number"})
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(authorization=f"Bearer {token}", session=db_session)
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail["error"]["code"] == "invalid_token"


def test_user_model_backward_compatibility(db_session: Session) -> None:
    """Verify User model allows None for password_hash (backward compatibility)."""
    user = User(
        email="legacy_user@speedinfer.local",
        name="Legacy Seed User",
        is_active=True,
    )
    assert user.password_hash is None
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    fetched = db_session.exec(select(User).where(User.email == user.email)).first()
    assert fetched is not None
    assert fetched.password_hash is None


def test_dummy_password_hash_constant_time() -> None:
    """Verify precomputed dummy hash structure and failure upon verification."""
    from speedinfer.core.security import DUMMY_PASSWORD_HASH

    assert DUMMY_PASSWORD_HASH.startswith("pbkdf2_sha256$100000$")
    assert verify_password("arbitrary_password", DUMMY_PASSWORD_HASH) is False


def test_apikey_is_expired_with_naive_datetime() -> None:
    """Verify is_expired safely handles timezone-naive and timezone-aware datetimes."""
    from datetime import datetime

    from speedinfer.database.models import ApiKey

    # Expired naive datetime (e.g. from SQLite)
    past_naive = datetime(2020, 1, 1, 0, 0, 0)
    key_expired = ApiKey(
        user_id=1,
        key_hash="a" * 64,
        prefix="sk-speedinfer-12345678",
        expires_at=past_naive,
    )
    assert key_expired.is_expired() is True

    # Future naive datetime
    future_naive = datetime(2035, 1, 1, 0, 0, 0)
    key_future = ApiKey(
        user_id=1,
        key_hash="b" * 64,
        prefix="sk-speedinfer-12345678",
        expires_at=future_naive,
    )
    assert key_future.is_expired() is False


def test_jwt_algorithm_validation_in_settings() -> None:
    """Verify Settings rejects invalid or insecure JWT algorithms."""
    from pydantic import ValidationError

    from speedinfer.config import Settings

    with pytest.raises(ValidationError):
        Settings(jwt_algorithm="none", api_key_pepper="atleast16characterslongsecret")

    with pytest.raises(ValidationError):
        Settings(jwt_algorithm="RS256", api_key_pepper="atleast16characterslongsecret")

    valid = Settings(jwt_algorithm="HS512", api_key_pepper="atleast16characterslongsecret")
    assert valid.jwt_algorithm == "HS512"


def test_get_current_user_lowercase_bearer(db_session: Session) -> None:
    """Verify get_current_user accepts case-insensitive 'bearer <token>' scheme."""
    user = User(
        email="case_insensitive@speedinfer.local",
        name="Case Insensitive",
        password_hash=hash_password("pwd12345"),
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    token = create_access_token({"sub": str(user.id), "email": user.email})
    resolved = get_current_user(authorization=f"bearer {token}", session=db_session)
    assert resolved.id == user.id


def test_get_current_user_empty_token_string(db_session: Session) -> None:
    """Verify get_current_user rejects Bearer headers without token string."""
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(authorization="Bearer   ", session=db_session)
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail["error"]["code"] == "invalid_token"


def test_user_request_password_limits() -> None:
    """Verify UserRegisterRequest and UserLoginRequest reject oversized passwords."""
    from pydantic import ValidationError

    from speedinfer.gateway.schemas import UserLoginRequest, UserRegisterRequest

    oversized_pwd = "A" * 300
    with pytest.raises(ValidationError):
        UserRegisterRequest(email="valid@speedinfer.local", password=oversized_pwd)

    with pytest.raises(ValidationError):
        UserLoginRequest(email="valid@speedinfer.local", password=oversized_pwd)
