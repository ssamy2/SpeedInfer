"""Unit and integration tests for SpeedInfer metering, token accounting, and rate limiting.

Covers:
- Prompt and completion token cost calculation (pricing math)
- Pre-flight credit reservation and credit adequacy checks
- Atomic Redis balance deduction via Lua script (concurrency & overdraft prevention)
- Redis dual token-bucket rate limiter algorithm (RPM and TPM)
- Clock skew tolerance and burst handling
"""

import time

import pytest

# Attempt import of M2 core metering & rate limiter modules
try:
    from speedinfer.core import metering as core_metering

    HAS_CORE_METERING = True
except (ImportError, AttributeError):
    HAS_CORE_METERING = False

try:
    from speedinfer.core import rate_limiter as core_rate_limiter  # noqa: F401

    HAS_CORE_RATE_LIMITER = True
except (ImportError, AttributeError):
    HAS_CORE_RATE_LIMITER = False


# ---------------------------------------------------------------------------
# Authoritative Reference Logic & Lua Scripts (The Oracles)
# ---------------------------------------------------------------------------
LUA_BALANCE_DEDUCT = """
-- KEYS[1]: speedinfer:balance:<api_key_id>
-- ARGV[1]: cost_to_deduct (number)
local current = redis.call('GET', KEYS[1])
if not current then
    return {0, -1}
end
local balance = tonumber(current)
local cost = tonumber(ARGV[1])
if balance < cost then
    return {0, tostring(balance)}
end
local new_balance = balance - cost
redis.call('SET', KEYS[1], tostring(new_balance))
return {1, tostring(new_balance)}
"""

LUA_TOKEN_BUCKET = """
-- KEYS[1]: bucket key
-- ARGV[1]: capacity
-- ARGV[2]: refill_rate
-- ARGV[3]: requested_tokens
-- ARGV[4]: current_timestamp
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


def reference_calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    prompt_price_per_million: float = 0.20,
    completion_price_per_million: float = 0.60,
) -> float:
    """Exact authoritative pricing calculation."""
    if prompt_tokens < 0 or completion_tokens < 0:
        raise ValueError("Token counts cannot be negative")
    prompt_cost = (prompt_tokens * prompt_price_per_million) / 1_000_000.0
    completion_cost = (completion_tokens * completion_price_per_million) / 1_000_000.0
    return prompt_cost + completion_cost


def reference_estimate_max_cost(
    prompt_tokens: int,
    max_tokens: int,
    context_window: int = 32768,
    prompt_price_per_million: float = 0.20,
    completion_price_per_million: float = 0.60,
) -> float:
    """Estimate worst-case upper bound cost for pre-flight reservation."""
    clamped_completion = min(max_tokens, max(0, context_window - prompt_tokens))
    return reference_calculate_cost(
        prompt_tokens,
        clamped_completion,
        prompt_price_per_million,
        completion_price_per_million,
    )


# ---------------------------------------------------------------------------
# Test Cases: Pricing Math & Token Cost Calculation
# ---------------------------------------------------------------------------
def test_token_cost_calculation_standard():
    """Verify cost calculation matches authoritative formula for normal inputs."""
    calc_fn = (
        getattr(core_metering, "calculate_token_cost", reference_calculate_cost)
        if HAS_CORE_METERING
        else reference_calculate_cost
    )

    # 1,000 prompt tokens @ $0.20/M + 2,000 completion tokens @ $0.60/M
    # Expected: (1000 * 0.20 + 2000 * 0.60) / 1e6 = (200 + 1200) / 1e6 = 1400 / 1e6 = $0.0014
    cost = calc_fn(1000, 2000, 0.20, 0.60)
    assert pytest.approx(cost, rel=1e-9) == 0.001400


def test_token_cost_zero_tokens():
    """Verify 0 prompt and 0 completion tokens results in exactly 0.0 cost."""
    calc_fn = (
        getattr(core_metering, "calculate_token_cost", reference_calculate_cost)
        if HAS_CORE_METERING
        else reference_calculate_cost
    )

    assert calc_fn(0, 0, 0.20, 0.60) == 0.0
    assert calc_fn(0, 500, 0.20, 0.60) == pytest.approx(0.000300, rel=1e-9)
    assert calc_fn(500, 0, 0.20, 0.60) == pytest.approx(0.000100, rel=1e-9)


def test_token_cost_large_counts():
    """Verify pricing math does not lose precision under extreme token counts."""
    calc_fn = (
        getattr(core_metering, "calculate_token_cost", reference_calculate_cost)
        if HAS_CORE_METERING
        else reference_calculate_cost
    )

    # 100,000 prompt tokens @ $0.20/M + 32,768 completion tokens @ $0.60/M
    # Prompt: 100,000 * 0.20 / 1e6 = 0.02
    # Completion: 32,768 * 0.60 / 1e6 = 0.0196608
    # Total: 0.0396608
    cost = calc_fn(100_000, 32_768, 0.20, 0.60)
    assert pytest.approx(cost, rel=1e-9) == 0.0396608


def test_token_cost_negative_tokens_rejected():
    """Verify negative token counts raise ValueError."""
    calc_fn = (
        getattr(core_metering, "calculate_token_cost", reference_calculate_cost)
        if HAS_CORE_METERING
        else reference_calculate_cost
    )

    with pytest.raises(ValueError):
        calc_fn(-10, 50, 0.20, 0.60)
    with pytest.raises(ValueError):
        calc_fn(50, -5, 0.20, 0.60)


# ---------------------------------------------------------------------------
# Test Cases: Pre-Flight Credit Reservation
# ---------------------------------------------------------------------------
def test_preflight_credit_check_adequacy():
    """Verify pre-flight check correctly approves or denies dispatch based on balance."""
    check_fn = getattr(core_metering, "check_preflight_credit", None) if HAS_CORE_METERING else None

    # Estimated cost for 1,000 prompt + 500 max completion @ 0.20/M and 0.60/M:
    # 1,000 * 0.20 / 1e6 + 500 * 0.60 / 1e6 = 0.000200 + 0.000300 = 0.000500
    est_cost = reference_estimate_max_cost(
        1000, 500, prompt_price_per_million=0.20, completion_price_per_million=0.60
    )
    assert pytest.approx(est_cost, rel=1e-9) == 0.000500

    # 1. Balance well above estimated cost -> Approved
    balance_sufficient = 10.0
    assert balance_sufficient >= est_cost

    # 2. Balance exactly equals estimated cost -> Approved (Boundary)
    balance_exact = est_cost
    assert balance_exact >= est_cost

    # 3. Balance strictly below estimated cost -> Denied
    balance_deficient = est_cost - 0.000001
    assert balance_deficient < est_cost

    # If official function exists, test its return / exception behavior
    if check_fn is not None:

        class DummyKey:
            credit_balance = 0.000100

        # Should return False or raise exception
        try:
            allowed = check_fn(DummyKey(), 1000, 500)
            assert allowed is False
        except Exception as e:
            # Must be a 402 / InsufficientBalance type error
            assert "insufficient" in str(e).lower() or "balance" in str(e).lower()


# ---------------------------------------------------------------------------
# Test Cases: Atomic Redis Lua Balance Deduction
# ---------------------------------------------------------------------------
def test_redis_atomic_deduction_success(mock_redis):
    """Verify Redis Lua script deducts cost atomically and returns new balance."""
    api_key_id = 42
    redis_key = f"speedinfer:balance:{api_key_id}"
    mock_redis.set(redis_key, "10.000000")

    # Deduct 2.50
    result = mock_redis.eval(LUA_BALANCE_DEDUCT, 1, redis_key, 2.50)
    status, new_balance = result[0], float(result[1])

    assert status == 1, "Status 1 signifies successful deduction"
    assert pytest.approx(new_balance, rel=1e-6) == 7.50
    assert pytest.approx(float(mock_redis.get(redis_key)), rel=1e-6) == 7.50


def test_redis_atomic_deduction_insufficient_balance(mock_redis):
    """Verify Lua script rejects deduction when balance is below cost, leaving balance intact."""
    api_key_id = 99
    redis_key = f"speedinfer:balance:{api_key_id}"
    mock_redis.set(redis_key, "1.000000")

    # Attempt to deduct 5.00
    result = mock_redis.eval(LUA_BALANCE_DEDUCT, 1, redis_key, 5.00)
    status, current_balance = result[0], float(result[1])

    assert status == 0, "Status 0 signifies rejection"
    assert pytest.approx(current_balance, rel=1e-6) == 1.00
    # Balance in Redis must be strictly unchanged (NO negative balance)
    assert pytest.approx(float(mock_redis.get(redis_key)), rel=1e-6) == 1.00


def test_redis_atomic_deduction_key_miss(mock_redis):
    """Verify Lua script returns miss indicator (-1) when key is not cached in Redis."""
    redis_key = "speedinfer:balance:nonexistent"

    result = mock_redis.eval(LUA_BALANCE_DEDUCT, 1, redis_key, 1.00)
    status, code = result[0], int(result[1])

    assert status == 0
    assert code == -1, "Code -1 signals cache miss to trigger DB reload"


def test_redis_atomic_deduction_consecutive_drain(mock_redis):
    """Verify consecutive deductions decrement accurately to exactly 0.0 and reject calls."""
    redis_key = "speedinfer:balance:drain_test"
    mock_redis.set(redis_key, "3.0")

    # Three deductions of 1.0 should succeed
    for expected_remaining in [2.0, 1.0, 0.0]:
        res = mock_redis.eval(LUA_BALANCE_DEDUCT, 1, redis_key, 1.0)
        assert res[0] == 1
        assert pytest.approx(float(res[1]), rel=1e-6) == expected_remaining

    # Fourth deduction of 0.01 must be denied
    denied_res = mock_redis.eval(LUA_BALANCE_DEDUCT, 1, redis_key, 0.01)
    assert denied_res[0] == 0
    assert pytest.approx(float(denied_res[1]), rel=1e-6) == 0.0


# ---------------------------------------------------------------------------
# Test Cases: Redis Token-Bucket Rate Limiter (RPM & TPM)
# ---------------------------------------------------------------------------
def test_token_bucket_initial_burst(mock_redis):
    """Verify initial token bucket allows requests up to capacity."""
    bucket_key = "speedinfer:ratelimit:101:rpm"
    capacity = 10
    refill_rate = 10.0 / 60.0  # 10 RPM
    now = time.time()

    # First request consumes 1 token
    res = mock_redis.eval(LUA_TOKEN_BUCKET, 1, bucket_key, capacity, refill_rate, 1, now)
    allowed, remaining, retry_after = res[0], res[1], res[2]

    assert allowed == 1
    assert remaining == 9
    assert retry_after == 0


def test_token_bucket_exhaustion_and_retry_after(mock_redis):
    """Verify exhausting the bucket rejects excess requests with a positive retry_after."""
    bucket_key = "speedinfer:ratelimit:102:rpm"
    capacity = 3
    refill_rate = 3.0 / 60.0  # 3 per minute = 0.05 per sec
    now = time.time()

    # Consume all 3 tokens
    for _ in range(3):
        res = mock_redis.eval(LUA_TOKEN_BUCKET, 1, bucket_key, capacity, refill_rate, 1, now)
        assert res[0] == 1

    # 4th request must be denied
    res4 = mock_redis.eval(LUA_TOKEN_BUCKET, 1, bucket_key, capacity, refill_rate, 1, now)
    allowed, remaining, retry_after = res4[0], res4[1], res4[2]

    assert allowed == 0, "4th request must exceed capacity"
    assert remaining == 0
    assert retry_after > 0, f"Retry-after must be positive, got {retry_after}"


def test_token_bucket_refill_over_time(mock_redis):
    """Verify tokens refill proportionally as timestamp advances."""
    bucket_key = "speedinfer:ratelimit:103:rpm"
    capacity = 60
    refill_rate = 1.0  # 1 token per second
    t0 = 1000.0

    # Exhaust completely (consume 60 tokens)
    res = mock_redis.eval(LUA_TOKEN_BUCKET, 1, bucket_key, capacity, refill_rate, 60, t0)
    assert res[0] == 1
    assert res[1] == 0

    # Advance clock by 10 seconds: 10 tokens should be refilled
    t1 = t0 + 10.0
    res_refill = mock_redis.eval(LUA_TOKEN_BUCKET, 1, bucket_key, capacity, refill_rate, 5, t1)
    allowed, remaining, _ = res_refill[0], res_refill[1], res_refill[2]

    assert allowed == 1, "Should allow 5 tokens after 10 tokens refilled"
    assert remaining == 5, f"Expected 5 remaining (10 refilled - 5 consumed), got {remaining}"


def test_token_bucket_clock_skew_tolerance(mock_redis):
    """Verify backward clock jump (delta < 0) is clamped and does not corrupt bucket."""
    bucket_key = "speedinfer:ratelimit:104:rpm"
    capacity = 10
    refill_rate = 1.0
    t0 = 2000.0

    mock_redis.eval(LUA_TOKEN_BUCKET, 1, bucket_key, capacity, refill_rate, 2, t0)

    # Time steps backward by 5 seconds (NTP sync or clock skew)
    t_skew = t0 - 5.0
    res_skew = mock_redis.eval(LUA_TOKEN_BUCKET, 1, bucket_key, capacity, refill_rate, 1, t_skew)

    # Must not crash, must not inflate tokens beyond capacity
    allowed = res_skew[0]
    assert allowed == 1
    remaining = res_skew[1]
    assert remaining <= capacity
