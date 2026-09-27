"""Password hashing, JWT token lifecycle, and user authentication security utilities.

Provides:
- Cryptographic password hashing using PBKDF2-HMAC-SHA256 with 100,000+ iterations.
- Constant-time password verification preventing timing side-channel attacks.
- Cryptographically signed JSON Web Token (JWT) issuance and validation.
- FastAPI dependency for authenticating user sessions via JWT Bearer tokens.
"""

import hashlib
import hmac
import secrets
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

import jwt
from fastapi import Depends, Header, HTTPException, status
from sqlmodel import Session, select

from speedinfer.config import get_settings
from speedinfer.database.models import User
from speedinfer.database.session import get_session

# Password hashing constants
PBKDF2_ALGORITHM = "sha256"
PBKDF2_ITERATIONS = 100_000
SALT_BYTES = 16
HASH_PREFIX = "pbkdf2_sha256"

# Precomputed 100,000-iteration PBKDF2 hash used to mitigate timing-based user enumeration
DUMMY_PASSWORD_HASH = (
    "pbkdf2_sha256$100000$0123456789abcdef0123456789abcdef$"
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)


class SecurityError(Exception):
    """Base exception for security-related failures."""


class TokenError(SecurityError):
    """Exception raised when a JWT token is invalid or malformed."""


class TokenExpiredError(SecurityError):
    """Exception raised when a JWT token has passed its expiration."""


def _resolve_jwt_secret() -> str:
    """Resolve secret key used for signing and verifying JWT tokens.

    Prioritizes explicit jwt_secret_key in settings, falling back to api_key_pepper.

    Returns:
        str: Secret key string for HMAC-SHA256 JWT signing.
    """
    settings = get_settings()
    if settings.jwt_secret_key is not None:
        raw = settings.jwt_secret_key
        return raw.get_secret_value() if hasattr(raw, "get_secret_value") else str(raw)
    raw_pepper = settings.api_key_pepper
    return (
        raw_pepper.get_secret_value()
        if hasattr(raw_pepper, "get_secret_value")
        else str(raw_pepper)
    )


def hash_password(password: str, iterations: int = PBKDF2_ITERATIONS) -> str:
    """Hash a plaintext password using salted PBKDF2-HMAC-SHA256.

    Args:
        password: Plaintext password to hash.
        iterations: Number of PBKDF2 iterations (default: 100,000).

    Returns:
        str: Encoded string in format 'pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>'.
    """
    salt = secrets.token_bytes(SALT_BYTES)
    derived = hashlib.pbkdf2_hmac(
        PBKDF2_ALGORITHM,
        password.encode("utf-8"),
        salt,
        iterations,
    )
    salt_hex = salt.hex()
    hash_hex = derived.hex()
    return f"{HASH_PREFIX}${iterations}${salt_hex}${hash_hex}"


def verify_password(plain_password: str, hashed_password: str | None) -> bool:
    """Verify a plaintext password against a stored PBKDF2-HMAC-SHA256 hash in constant time.

    Args:
        plain_password: Candidate plaintext password.
        hashed_password: Stored hash formatted as
            'pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>'.

    Returns:
        bool: True if candidate matches the stored hash, False otherwise.
    """
    if not plain_password or not hashed_password:
        return False

    parts = hashed_password.split("$")
    if len(parts) != 4 or parts[0] != HASH_PREFIX:
        return False

    try:
        iterations = int(parts[1])
        salt = bytes.fromhex(parts[2])
        expected_hash = parts[3]
    except (ValueError, IndexError):
        return False

    computed = hashlib.pbkdf2_hmac(
        PBKDF2_ALGORITHM,
        plain_password.encode("utf-8"),
        salt,
        iterations,
    ).hex()

    return hmac.compare_digest(computed, expected_hash)


def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Generate a signed JWT access token containing arbitrary payload claims.

    Args:
        data: Claims to embed into the token (e.g. sub, email).
        expires_delta: Optional custom duration until expiration.

    Returns:
        str: Compact encoded JWT string.
    """
    settings = get_settings()
    to_encode = data.copy()

    now = datetime.now(UTC)
    if expires_delta is not None:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.jwt_access_token_expire_minutes)

    to_encode.update({
        "exp": expire,
        "iat": now,
    })

    # Ensure subject is converted to string for standard JWT compliance
    if "sub" in to_encode and not isinstance(to_encode["sub"], str):
        to_encode["sub"] = str(to_encode["sub"])

    secret = _resolve_jwt_secret()
    algorithm = settings.jwt_algorithm

    return jwt.encode(to_encode, secret, algorithm=algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and cryptographically verify a JWT access token.

    Args:
        token: Compact encoded JWT string.

    Returns:
        dict[str, Any]: Decoded payload claims dictionary.

    Raises:
        TokenExpiredError: If token expiration timestamp has passed.
        TokenError: If token signature is invalid, corrupted, or unsupported.
    """
    settings = get_settings()
    secret = _resolve_jwt_secret()
    algorithm = settings.jwt_algorithm

    try:
        payload = jwt.decode(token, secret, algorithms=[algorithm])
        return payload
    except jwt.ExpiredSignatureError as exc:
        raise TokenExpiredError("Access token has expired.") from exc
    except jwt.PyJWTError as exc:
        raise TokenError("Could not validate access token credentials.") from exc


def get_current_user(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    session: Annotated[Session, Depends(get_session)] = None,
) -> User:
    """FastAPI dependency for authenticating user requests via Bearer JWT tokens.

    Extracts Bearer token from HTTP Authorization header, verifies signature and expiry,
    and looks up the active User record from the database.

    Args:
        authorization: Raw Authorization header (expected 'Bearer <token>').
        session: Injected database session.

    Returns:
        User: Authenticated database User entity.

    Raises:
        HTTPException: HTTP 401 with OpenAI-compatible error payload on failure.
    """
    if not authorization or not authorization.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "Missing Authorization header. Expected 'Bearer <token>'",
                    "type": "authentication_error",
                    "param": None,
                    "code": "missing_token",
                }
            },
        )

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "Invalid Authorization header format. Expected 'Bearer <token>'",
                    "type": "authentication_error",
                    "param": None,
                    "code": "invalid_header",
                }
            },
        )

    token = parts[1].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "Missing token string in Authorization header.",
                    "type": "authentication_error",
                    "param": None,
                    "code": "invalid_token",
                }
            },
        )
    try:
        payload = decode_access_token(token)
    except TokenExpiredError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "Authentication token has expired.",
                    "type": "authentication_error",
                    "param": None,
                    "code": "token_expired",
                }
            },
        ) from None
    except TokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "Invalid authentication token.",
                    "type": "authentication_error",
                    "param": None,
                    "code": "invalid_token",
                }
            },
        ) from None

    subject: str | None = payload.get("sub")
    if not subject:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "Malformed token subject claim.",
                    "type": "authentication_error",
                    "param": None,
                    "code": "invalid_token",
                }
            },
        )

    try:
        user_id = int(subject)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "Invalid token subject format.",
                    "type": "authentication_error",
                    "param": None,
                    "code": "invalid_token",
                }
            },
        ) from None

    if session is None:
        from speedinfer.database.session import engine

        with Session(engine) as fallback_session:
            user = fallback_session.exec(select(User).where(User.id == user_id)).first()
    else:
        user = session.exec(select(User).where(User.id == user_id)).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "User associated with token was not found.",
                    "type": "authentication_error",
                    "param": None,
                    "code": "user_not_found",
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

    return user
