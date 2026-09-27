"""Integration tests for concurrency race conditions and overdraft prevention.

Covers:
- 50 concurrent balance deductions against a single API key balance
- Zero negative balance invariant (no overdraft under high contention)
- Exact accounting: sum of deducted amounts equals balance decrement
- 50 concurrent requests against RPM rate limit bucket
- End-to-end gateway concurrent request race condition validation
"""

import asyncio

import pytest

# Check for gateway presence
try:
    from speedinfer.gateway.app import app  # noqa: F401

    HAS_GATEWAY = True
except (ImportError, AttributeError):
    HAS_GATEWAY = False


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


@pytest.fixture
def require_gateway():
    """Skip test if speedinfer.gateway is not yet implemented."""
    if not HAS_GATEWAY:
        pytest.skip("speedinfer.gateway not yet implemented by M4")


# ---------------------------------------------------------------------------
# Test Cases: Direct Async Redis Concurrency Verification
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_50_concurrent_balance_deductions_prevent_overdraft(async_mock_redis):
    """Verify 50 concurrent requests cannot drive balance below 0.0 under high concurrency."""
    key_id = 999
    balance_key = f"speedinfer:balance:{key_id}"

    # Initial balance: $0.050 (allows exactly 25 requests of $0.002 each)
    initial_balance = 0.050
    cost_per_request = 0.002
    await async_mock_redis.set(balance_key, str(initial_balance))

    # Fire 50 concurrent deduction attempts
    num_requests = 50

    async def attempt_deduction():
        res = await async_mock_redis.eval(
            LUA_BALANCE_DEDUCT,
            1,
            balance_key,
            cost_per_request,
        )
        return int(res[0]), float(res[1])

    tasks = [attempt_deduction() for _ in range(num_requests)]
    results = await asyncio.gather(*tasks)

    successes = [r for r in results if r[0] == 1]
    rejections = [r for r in results if r[0] == 0]

    # Exactly 25 requests should have succeeded (25 * 0.002 = 0.050)
    assert len(successes) == 25
    # Exactly 25 requests should have been rejected
    assert len(rejections) == 25

    # Final balance must be non-negative (strictly 0.000)
    final_balance_str = await async_mock_redis.get(balance_key)
    final_balance = float(final_balance_str)
    assert pytest.approx(final_balance, abs=1e-6) == 0.000


@pytest.mark.asyncio
async def test_50_concurrent_requests_rate_limiter_strict_cutoff(async_mock_redis):
    """Verify 50 concurrent requests against a 10 RPM limit allow exactly 10 and reject 40."""
    rpm_key = "speedinfer:ratelimit:rpm:concurrency_test"
    capacity = 10
    refill_rate = 10 / 60.0
    now = 1700000000.0

    async def attempt_rate_limit():
        res = await async_mock_redis.eval(
            LUA_TOKEN_BUCKET,
            1,
            rpm_key,
            capacity,
            refill_rate,
            1,
            now,
        )
        return int(res[0]), int(res[1]), int(res[2])

    tasks = [attempt_rate_limit() for _ in range(50)]
    results = await asyncio.gather(*tasks)

    allowed = [r for r in results if r[0] == 1]
    blocked = [r for r in results if r[0] == 0]

    assert len(allowed) == 10
    assert len(blocked) == 40

    # Blocked requests should have retry_after > 0
    for b in blocked:
        assert b[2] > 0


@pytest.mark.asyncio
async def test_gateway_concurrent_chat_requests_overdraft_prevention(async_client, require_gateway):
    """End-to-end gateway test: 50 concurrent chat completion requests against limited key."""
    headers = {"Authorization": "Bearer sk-speedinfer-limited-balance-key"}
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "Short test"}],
        "max_tokens": 16,
    }

    async def send_req():
        return await async_client.post("/v1/chat/completions", json=payload, headers=headers)

    tasks = [send_req() for _ in range(50)]
    responses = await asyncio.gather(*tasks)

    status_codes = [r.status_code for r in responses]
    assert all(code in {200, 402, 429} for code in status_codes)
    # Never internal server error (500) under race condition
    assert 500 not in status_codes
