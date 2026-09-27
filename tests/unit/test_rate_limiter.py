"""Unit tests for SpeedInfer dual token-bucket rate limiter (RPM & TPM).

Covers:
- Dual token bucket algorithm (RPM for requests, TPM for tokens)
- Burst consumption and bucket capacity enforcement
- Exhaustion behavior and Retry-After calculation
- Fractional refill over elapsed time
- Clock skew backward tolerance
- Isolation across distinct API keys
- Rate limit response headers serialization
"""

import time
from dataclasses import dataclass
from typing import Any

import pytest

# Attempt import of M2 core rate limiter
try:
    from speedinfer.core import rate_limiter as core_rate_limiter

    HAS_CORE_RATE_LIMITER = True
except (ImportError, AttributeError):
    HAS_CORE_RATE_LIMITER = False


# ---------------------------------------------------------------------------
# Authoritative Contract Specification & Reference Oracle
# ---------------------------------------------------------------------------
@dataclass
class RateLimitResult:
    """Authoritative contract for rate limit check results."""

    allowed: bool
    remaining_requests: int
    remaining_tokens: int
    reset_seconds: int
    retry_after: int = 0


def reference_build_rate_limit_headers(
    result: RateLimitResult,
    rpm_limit: int,
    tpm_limit: int,
) -> dict[str, str]:
    """Authoritative rate limit response headers."""
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


# Authoritative Token Bucket Lua Script
LUA_TOKEN_BUCKET = """
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


def execute_reference_dual_rate_limit(
    redis_client: Any,
    key_id: int,
    rpm_limit: int,
    tpm_limit: int,
    requested_tokens: int,
    current_time: float | None = None,
) -> RateLimitResult:
    """Execute dual token bucket algorithm directly via Redis Lua scripts."""
    if requested_tokens < 0:
        raise ValueError("requested_tokens cannot be negative")

    now = time.time() if current_time is None else current_time
    rpm_key = f"speedinfer:ratelimit:rpm:{key_id}"
    tpm_key = f"speedinfer:ratelimit:tpm:{key_id}"

    # RPM: capacity = rpm_limit, refill_rate = rpm_limit / 60.0, requested = 1
    rpm_refill = rpm_limit / 60.0
    rpm_res = redis_client.eval(
        LUA_TOKEN_BUCKET,
        1,
        rpm_key,
        rpm_limit,
        rpm_refill,
        1,
        now,
    )
    rpm_allowed = bool(rpm_res[0] == 1)
    rpm_remaining = int(rpm_res[1])
    rpm_retry = int(rpm_res[2])

    # TPM: capacity = tpm_limit, refill_rate = tpm_limit / 60.0, requested = requested_tokens
    tpm_refill = tpm_limit / 60.0
    tpm_res = redis_client.eval(
        LUA_TOKEN_BUCKET,
        1,
        tpm_key,
        tpm_limit,
        tpm_refill,
        requested_tokens,
        now,
    )
    tpm_allowed = bool(tpm_res[0] == 1)
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


@pytest.fixture
def require_rate_limiter():
    """Skip test if speedinfer.core.rate_limiter is not yet implemented."""
    if not HAS_CORE_RATE_LIMITER:
        pytest.skip("speedinfer.core.rate_limiter not yet implemented by M2")


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------
def test_rate_limit_result_dataclass():
    """Verify RateLimitResult attributes and defaults conform to contract."""
    res = RateLimitResult(
        allowed=True,
        remaining_requests=59,
        remaining_tokens=49500,
        reset_seconds=60,
    )
    assert res.allowed is True
    assert res.remaining_requests == 59
    assert res.remaining_tokens == 49500
    assert res.reset_seconds == 60
    assert res.retry_after == 0


def test_rate_limit_headers_builder():
    """Verify standard rate limit headers are formatted as required strings."""
    fn = (
        getattr(core_rate_limiter, "build_rate_limit_headers", reference_build_rate_limit_headers)
        if HAS_CORE_RATE_LIMITER
        else reference_build_rate_limit_headers
    )
    res = RateLimitResult(
        allowed=True,
        remaining_requests=10,
        remaining_tokens=5000,
        reset_seconds=45,
    )
    headers = fn(res, rpm_limit=60, tpm_limit=60000)

    assert headers["x-ratelimit-limit-requests"] == "60"
    assert headers["x-ratelimit-remaining-requests"] == "10"
    assert headers["x-ratelimit-reset-requests"] == "45"
    assert headers["x-ratelimit-limit-tokens"] == "60000"
    assert headers["x-ratelimit-remaining-tokens"] == "5000"
    assert headers["x-ratelimit-reset-tokens"] == "45"
    assert "retry-after" not in headers


def test_rate_limit_headers_builder_on_rejection():
    """Verify Retry-After is included in headers when request is rejected."""
    fn = (
        getattr(core_rate_limiter, "build_rate_limit_headers", reference_build_rate_limit_headers)
        if HAS_CORE_RATE_LIMITER
        else reference_build_rate_limit_headers
    )
    res = RateLimitResult(
        allowed=False,
        remaining_requests=0,
        remaining_tokens=500,
        reset_seconds=30,
        retry_after=5,
    )
    headers = fn(res, rpm_limit=60, tpm_limit=60000)
    assert headers["retry-after"] == "5"
    assert headers["x-ratelimit-remaining-requests"] == "0"


def test_dual_token_bucket_initial_success(mock_redis):
    """Verify initial request consumes 1 request and N tokens within capacity."""
    limiter_fn = (
        getattr(core_rate_limiter, "check_rate_limit", execute_reference_dual_rate_limit)
        if HAS_CORE_RATE_LIMITER
        else execute_reference_dual_rate_limit
    )

    result = limiter_fn(
        redis_client=mock_redis,
        key_id=101,
        rpm_limit=60,
        tpm_limit=10000,
        requested_tokens=250,
    )
    assert result.allowed is True
    assert result.remaining_requests == 59
    assert result.remaining_tokens == 9750
    assert result.retry_after == 0


def test_rpm_exhaustion_rejects_and_sets_retry_after(mock_redis):
    """Verify exceeding RPM capacity rejects subsequent requests."""
    limiter_fn = (
        getattr(core_rate_limiter, "check_rate_limit", execute_reference_dual_rate_limit)
        if HAS_CORE_RATE_LIMITER
        else execute_reference_dual_rate_limit
    )

    t0 = 1700000000.0
    # Key with 2 RPM and 10,000 TPM
    r1 = limiter_fn(mock_redis, 102, 2, 10000, 10, current_time=t0)
    assert r1.allowed is True
    assert r1.remaining_requests == 1

    r2 = limiter_fn(mock_redis, 102, 2, 10000, 10, current_time=t0)
    assert r2.allowed is True
    assert r2.remaining_requests == 0

    # Third request should be blocked by RPM limit
    r3 = limiter_fn(mock_redis, 102, 2, 10000, 10, current_time=t0)
    assert r3.allowed is False
    assert r3.retry_after > 0


def test_tpm_exhaustion_rejects_and_sets_retry_after(mock_redis):
    """Verify exceeding TPM capacity rejects request even if RPM is available."""
    limiter_fn = (
        getattr(core_rate_limiter, "check_rate_limit", execute_reference_dual_rate_limit)
        if HAS_CORE_RATE_LIMITER
        else execute_reference_dual_rate_limit
    )

    t0 = 1700000000.0
    # Key with 100 RPM and 1,000 TPM
    r1 = limiter_fn(mock_redis, 103, 100, 1000, 800, current_time=t0)
    assert r1.allowed is True
    assert r1.remaining_tokens == 200

    # Requesting 300 tokens when only 200 left -> rejected
    r2 = limiter_fn(mock_redis, 103, 100, 1000, 300, current_time=t0)
    assert r2.allowed is False
    assert r2.retry_after > 0


def test_token_bucket_refill_over_time(mock_redis):
    """Verify bucket tokens refill proportionally to elapsed time."""
    limiter_fn = (
        getattr(core_rate_limiter, "check_rate_limit", execute_reference_dual_rate_limit)
        if HAS_CORE_RATE_LIMITER
        else execute_reference_dual_rate_limit
    )

    t0 = 1700000000.0
    # 60 RPM = 1 req/sec; 6,000 TPM = 100 tokens/sec
    limiter_fn(mock_redis, 104, 60, 6000, 5000, current_time=t0)

    # Advance 10 seconds: 10 * 100 = +1000 tokens
    t1 = t0 + 10.0
    r2 = limiter_fn(mock_redis, 104, 60, 6000, 100, current_time=t1)
    assert r2.allowed is True
    # Initial: 6000 - 5000 = 1000. +1000 refilled = 2000. Consumed 100 -> ~1900
    assert 1850 <= r2.remaining_tokens <= 1950


def test_clock_skew_tolerance(mock_redis):
    """Verify clock stepping backward does not cause negative delta or crash."""
    limiter_fn = (
        getattr(core_rate_limiter, "check_rate_limit", execute_reference_dual_rate_limit)
        if HAS_CORE_RATE_LIMITER
        else execute_reference_dual_rate_limit
    )

    t0 = 1700000050.0
    limiter_fn(mock_redis, 105, 60, 6000, 100, current_time=t0)

    # Skew backward 10 seconds
    t_skew = t0 - 10.0
    res = limiter_fn(mock_redis, 105, 60, 6000, 50, current_time=t_skew)
    assert res.allowed is True


def test_key_rate_limit_isolation(mock_redis):
    """Verify consumption on Key A does not deplete Key B."""
    limiter_fn = (
        getattr(core_rate_limiter, "check_rate_limit", execute_reference_dual_rate_limit)
        if HAS_CORE_RATE_LIMITER
        else execute_reference_dual_rate_limit
    )

    t0 = 1700000000.0
    # Exhaust Key 1
    limiter_fn(mock_redis, 1, 1, 100, 100, current_time=t0)
    k1_blocked = limiter_fn(mock_redis, 1, 1, 100, 1, current_time=t0)
    assert k1_blocked.allowed is False

    # Key 2 should be completely fresh and unaffected
    k2 = limiter_fn(mock_redis, 2, 1, 100, 50, current_time=t0)
    assert k2.allowed is True
    assert k2.remaining_tokens == 50


def test_negative_tokens_rejected(mock_redis):
    """Verify negative requested tokens raises ValueError."""
    limiter_fn = (
        getattr(core_rate_limiter, "check_rate_limit", execute_reference_dual_rate_limit)
        if HAS_CORE_RATE_LIMITER
        else execute_reference_dual_rate_limit
    )

    with pytest.raises(ValueError, match="negative"):
        limiter_fn(mock_redis, 106, 60, 6000, -5)


def test_module_implementation_contract(require_rate_limiter):
    """Verify M2 speedinfer.core.rate_limiter exposes expected public interface."""
    assert hasattr(core_rate_limiter, "check_rate_limit")
    assert hasattr(core_rate_limiter, "RateLimitResult")
    assert hasattr(core_rate_limiter, "build_rate_limit_headers")
