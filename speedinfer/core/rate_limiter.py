"""Dual token-bucket rate limiter engine for requests (RPM) and tokens (TPM).

Provides:
- RateLimitResult dataclass representing rate limiter decision and quota states.
- Atomic Redis token-bucket execution with fractional refills over elapsed time.
- Clock skew tolerance preventing token inflation on backward clock adjustments.
- Serialization of standard x-ratelimit-* and retry-after response headers.
- RateLimiter class and check_rate_limit function supporting sync and async Redis.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from speedinfer.database.models import ApiKey

# Path to atomic token bucket Lua script
LUA_DIR = Path(__file__).parent / "lua"
TOKEN_BUCKET_LUA_PATH = LUA_DIR / "token_bucket.lua"

# Default Lua script content as resilient fallback
DEFAULT_LUA_TOKEN_BUCKET = """
-- KEYS[1]: bucket key
-- ARGV[1]: capacity
-- ARGV[2]: refill_rate (tokens per second)
-- ARGV[3]: requested_tokens
-- ARGV[4]: current_timestamp (seconds)
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local requested = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

local data = redis.call('HMGET', key, 'tokens', 'last_updated')
local tokens = tonumber(data[1])
local last_updated = tonumber(data[2])

if not tokens or not last_updated then
    tokens = capacity
    last_updated = now
else
    local delta = math.max(0, now - last_updated)
    tokens = math.min(capacity, tokens + delta * refill_rate)
    last_updated = now
end

if tokens >= requested then
    tokens = tokens - requested
    redis.call('HMSET', key, 'tokens', tostring(tokens), 'last_updated', tostring(last_updated))
    redis.call('EXPIRE', key, math.ceil(capacity / refill_rate) + 60)
    return {1, math.floor(tokens), 0}
else
    redis.call('HMSET', key, 'tokens', tostring(tokens), 'last_updated', tostring(last_updated))
    redis.call('EXPIRE', key, math.ceil(capacity / refill_rate) + 60)
    local retry_after = math.ceil((requested - tokens) / refill_rate)
    return {0, math.floor(tokens), retry_after}
end
"""


def _load_lua_script() -> str:
    """Load the Lua token bucket script from disk with fallback.

    Returns:
        str: Lua script source code.
    """
    if TOKEN_BUCKET_LUA_PATH.is_file():
        try:
            return TOKEN_BUCKET_LUA_PATH.read_text(encoding="utf-8")
        except OSError:
            return DEFAULT_LUA_TOKEN_BUCKET.strip()


LUA_TOKEN_BUCKET = _load_lua_script()


@dataclass
class RateLimitResult:
    """Result of dual token bucket rate limit evaluation.

    Attributes:
        allowed: Whether the request is permitted within limits.
        remaining_requests: Remaining request quota in the RPM bucket.
        remaining_tokens: Remaining token quota in the TPM bucket.
        reset_seconds: Seconds until current window fully resets.
        retry_after: Seconds client must wait before retrying if rejected.
    """

    allowed: bool
    remaining_requests: int
    remaining_tokens: int
    reset_seconds: int
    retry_after: int = 0


def build_rate_limit_headers(
    result: RateLimitResult,
    rpm_limit: int,
    tpm_limit: int,
) -> dict[str, str]:
    """Build standard HTTP rate-limiting response headers.

    Injects:
    - x-ratelimit-limit-requests: Total requests allowed per minute.
    - x-ratelimit-remaining-requests: Remaining requests allowed in window.
    - x-ratelimit-reset-requests: Reset interval in seconds.
    - x-ratelimit-limit-tokens: Total tokens allowed per minute.
    - x-ratelimit-remaining-tokens: Remaining tokens allowed in window.
    - x-ratelimit-reset-tokens: Reset interval in seconds.
    - retry-after: Included only when request was rejected (allowed == False).

    Args:
        result: Evaluated RateLimitResult.
        rpm_limit: Configured requests-per-minute limit.
        tpm_limit: Configured tokens-per-minute limit.

    Returns:
        dict[str, str]: Formatted HTTP headers dictionary.
    """
    headers = {
        "x-ratelimit-limit-requests": str(rpm_limit),
        "x-ratelimit-remaining-requests": str(max(0, result.remaining_requests)),
        "x-ratelimit-reset-requests": str(result.reset_seconds),
        "x-ratelimit-limit-tokens": str(tpm_limit),
        "x-ratelimit-remaining-tokens": str(max(0, result.remaining_tokens)),
        "x-ratelimit-reset-tokens": str(result.reset_seconds),
    }
    if not result.allowed and result.retry_after > 0:
        headers["retry-after"] = str(result.retry_after)
    return headers


# Alias get_rate_limit_headers to build_rate_limit_headers
get_rate_limit_headers = build_rate_limit_headers


def _extract_rate_limit_args(
    api_key: Any = None,
    rpm_limit: int | None = None,
    tpm_limit: int | None = None,
    requested_tokens: int = 1,
    **kwargs: Any,
) -> tuple[int, int, int, int]:
    """Normalize polymorphic arguments across ApiKey instances, primitives, and kwargs.

    Args:
        api_key: ApiKey model, integer key ID, or None if passed via kwargs.
        rpm_limit: rpm_limit integer, requested_tokens (if ApiKey passed), or None.
        tpm_limit: tpm_limit integer or None.
        requested_tokens: Requested tokens count.
        **kwargs: Optional keyword arguments including key_id, api_key_id.

    Returns:
        tuple[int, int, int, int]: (key_id, rpm_limit, tpm_limit, requested_tokens).

    Raises:
        ValueError: If neither api_key nor key_id was provided.
    """
    target = api_key if api_key is not None else kwargs.get("key_id", kwargs.get("api_key_id"))
    if target is None:
        raise ValueError("api_key or key_id must be provided")

    if isinstance(target, ApiKey) or hasattr(target, "rpm_limit"):
        key_id = int(target.id if target.id is not None else 0)
        resolved_rpm = int(target.rpm_limit)
        resolved_tpm = int(target.tpm_limit)
        if tpm_limit is None and rpm_limit is not None and "requested_tokens" not in kwargs:
            tokens = int(rpm_limit)
        else:
            tokens = int(kwargs.get("requested_tokens", requested_tokens))
        return key_id, resolved_rpm, resolved_tpm, tokens

    key_id = int(target)
    resolved_rpm = int(rpm_limit if rpm_limit is not None else kwargs.get("rpm_limit", 60))
    resolved_tpm = int(tpm_limit if tpm_limit is not None else kwargs.get("tpm_limit", 60_000))
    tokens = int(kwargs.get("requested_tokens", requested_tokens))
    return key_id, resolved_rpm, resolved_tpm, tokens


def check_rate_limit(
    redis_client: Any,
    api_key: Any = None,
    rpm_limit: int | None = None,
    tpm_limit: int | None = None,
    requested_tokens: int = 1,
    current_time: float | None = None,
    **kwargs: Any,
) -> RateLimitResult:
    """Evaluate dual token-bucket rate limits for an API key in Redis.

    Evaluates both RPM (1 request consumed) and TPM (requested_tokens consumed).
    Allows request if and only if BOTH buckets have sufficient tokens.
    Handles clock skew gracefully by clamping backward time deltas to zero.

    Args:
        redis_client: Synchronous Redis client instance.
        api_key: ApiKey model, integer key ID, or None if key_id in kwargs.
        rpm_limit: rpm_limit integer, or requested_tokens if api_key passed.
        tpm_limit: tpm_limit integer if primitives used.
        requested_tokens: Total tokens (prompt + estimated completion) to reserve.
        current_time: Optional explicit timestamp in seconds for deterministic testing.
        **kwargs: Flexible keyword arguments (key_id, api_key_id, etc.).

    Returns:
        RateLimitResult: Rate limiting decision and remaining quota.

    Raises:
        ValueError: If requested_tokens is negative.
    """
    key_id, rpm_lim, tpm_lim, req_tokens = _extract_rate_limit_args(
        api_key=api_key,
        rpm_limit=rpm_limit,
        tpm_limit=tpm_limit,
        requested_tokens=requested_tokens,
        **kwargs,
    )

    if req_tokens < 0:
        raise ValueError("requested_tokens cannot be negative")

    now = time.time() if current_time is None else float(current_time)
    rpm_key = f"speedinfer:ratelimit:rpm:{key_id}"
    tpm_key = f"speedinfer:ratelimit:tpm:{key_id}"

    # Evaluate RPM bucket (1 request)
    rpm_refill = rpm_lim / 60.0
    rpm_res = redis_client.eval(
        LUA_TOKEN_BUCKET,
        1,
        rpm_key,
        rpm_lim,
        rpm_refill,
        1,
        now,
    )
    rpm_allowed = bool(int(rpm_res[0]) == 1)
    rpm_remaining = int(rpm_res[1])
    rpm_retry = int(rpm_res[2])

    # Evaluate TPM bucket (req_tokens tokens)
    tpm_refill = tpm_lim / 60.0
    tpm_res = redis_client.eval(
        LUA_TOKEN_BUCKET,
        1,
        tpm_key,
        tpm_lim,
        tpm_refill,
        req_tokens,
        now,
    )
    tpm_allowed = bool(int(tpm_res[0]) == 1)
    tpm_remaining = int(tpm_res[1])
    tpm_retry = int(tpm_res[2])

    allowed = rpm_allowed and tpm_allowed
    retry_after = max(rpm_retry, tpm_retry) if not allowed else 0
    reset_seconds = 60

    return RateLimitResult(
        allowed=allowed,
        remaining_requests=rpm_remaining,
        remaining_tokens=tpm_remaining,
        reset_seconds=reset_seconds,
        retry_after=retry_after,
    )


async def async_check_rate_limit(
    redis_client: Any,
    api_key: Any = None,
    rpm_limit: int | None = None,
    tpm_limit: int | None = None,
    requested_tokens: int = 1,
    current_time: float | None = None,
    **kwargs: Any,
) -> RateLimitResult:
    """Asynchronously evaluate dual token-bucket rate limits for an API key in Redis.

    Args:
        redis_client: Asynchronous Redis client instance.
        api_key: ApiKey model, integer key ID, or None if key_id in kwargs.
        rpm_limit: rpm_limit integer, or requested_tokens if api_key passed.
        tpm_limit: tpm_limit integer if primitives used.
        requested_tokens: Total tokens to reserve.
        current_time: Optional explicit timestamp in seconds.
        **kwargs: Flexible keyword arguments.

    Returns:
        RateLimitResult: Rate limiting decision and remaining quota.

    Raises:
        ValueError: If requested_tokens is negative.
    """
    key_id, rpm_lim, tpm_lim, req_tokens = _extract_rate_limit_args(
        api_key=api_key,
        rpm_limit=rpm_limit,
        tpm_limit=tpm_limit,
        requested_tokens=requested_tokens,
        **kwargs,
    )

    if req_tokens < 0:
        raise ValueError("requested_tokens cannot be negative")

    now = time.time() if current_time is None else float(current_time)
    rpm_key = f"speedinfer:ratelimit:rpm:{key_id}"
    tpm_key = f"speedinfer:ratelimit:tpm:{key_id}"

    rpm_refill = rpm_lim / 60.0
    rpm_res = await redis_client.eval(
        LUA_TOKEN_BUCKET,
        1,
        rpm_key,
        rpm_lim,
        rpm_refill,
        1,
        now,
    )
    rpm_allowed = bool(int(rpm_res[0]) == 1)
    rpm_remaining = int(rpm_res[1])
    rpm_retry = int(rpm_res[2])

    tpm_refill = tpm_lim / 60.0
    tpm_res = await redis_client.eval(
        LUA_TOKEN_BUCKET,
        1,
        tpm_key,
        tpm_lim,
        tpm_refill,
        req_tokens,
        now,
    )
    tpm_allowed = bool(int(tpm_res[0]) == 1)
    tpm_remaining = int(tpm_res[1])
    tpm_retry = int(tpm_res[2])

    allowed = rpm_allowed and tpm_allowed
    retry_after = max(rpm_retry, tpm_retry) if not allowed else 0
    reset_seconds = 60

    return RateLimitResult(
        allowed=allowed,
        remaining_requests=rpm_remaining,
        remaining_tokens=tpm_remaining,
        reset_seconds=reset_seconds,
        retry_after=retry_after,
    )


class RateLimiter:
    """Stateful service wrapper for dual token-bucket rate limit verification."""

    def __init__(self, redis_client: Any = None) -> None:
        """Initialize RateLimiter.

        Args:
            redis_client: Optional default Redis client.
        """
        self.redis_client = redis_client

    def check_rate_limit(
        self,
        redis_client_or_key: Any = None,
        api_key: Any = None,
        rpm_limit: int | None = None,
        tpm_limit: int | None = None,
        requested_tokens: int = 1,
        current_time: float | None = None,
        **kwargs: Any,
    ) -> RateLimitResult:
        """Check rate limit using instance client or passed client.

        Args:
            redis_client_or_key: Redis client or ApiKey entity.
            api_key: ApiKey entity or requested_tokens count.
            rpm_limit: RPM limit.
            tpm_limit: TPM limit.
            requested_tokens: Number of tokens requested.
            current_time: Optional explicit timestamp.
            **kwargs: Flexible keyword arguments.

        Returns:
            RateLimitResult: Rate limiting decision.
        """
        if hasattr(redis_client_or_key, "eval"):
            client = redis_client_or_key
            target_key = api_key
        else:
            client = self.redis_client
            target_key = redis_client_or_key

        return check_rate_limit(
            redis_client=client,
            api_key=target_key,
            rpm_limit=rpm_limit,
            tpm_limit=tpm_limit,
            requested_tokens=requested_tokens,
            current_time=current_time,
            **kwargs,
        )

    async def async_check_rate_limit(
        self,
        redis_client_or_key: Any = None,
        api_key: Any = None,
        rpm_limit: int | None = None,
        tpm_limit: int | None = None,
        requested_tokens: int = 1,
        current_time: float | None = None,
        **kwargs: Any,
    ) -> RateLimitResult:
        """Asynchronously check rate limit using instance client or passed client.

        Args:
            redis_client_or_key: Redis client or ApiKey entity.
            api_key: ApiKey entity or requested_tokens count.
            rpm_limit: RPM limit.
            tpm_limit: TPM limit.
            requested_tokens: Number of tokens requested.
            current_time: Optional explicit timestamp.
            **kwargs: Flexible keyword arguments.

        Returns:
            RateLimitResult: Rate limiting decision.
        """
        if hasattr(redis_client_or_key, "eval"):
            client = redis_client_or_key
            target_key = api_key
        else:
            client = self.redis_client
            target_key = redis_client_or_key

        return await async_check_rate_limit(
            redis_client=client,
            api_key=target_key,
            rpm_limit=rpm_limit,
            tpm_limit=tpm_limit,
            requested_tokens=requested_tokens,
            current_time=current_time,
            **kwargs,
        )
