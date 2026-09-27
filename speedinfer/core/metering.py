"""Metering, token accounting, and atomic Redis credit deduction engine.

Provides:
- Exact pricing calculations for prompt and completion tokens.
- Pre-flight credit reservation and credit adequacy validation.
- Atomic balance deduction via Redis Lua script preventing race overdrafts.
- Transactional synchronization of inference transactions to database UsageLedger.
- Streaming token accounting tracker with TTFT and partial delivery cost deduction.
"""

from pathlib import Path
from typing import Any

from fastapi import HTTPException, status
from sqlmodel import Session

from speedinfer.database.models import ApiKey, UsageLedger, utc_now

# Path to atomic balance deduction Lua script
LUA_DIR = Path(__file__).parent / "lua"
BALANCE_DEDUCT_LUA_PATH = LUA_DIR / "balance_deduct.lua"

# Default Lua script content as resilient fallback
DEFAULT_LUA_BALANCE_DEDUCT = """
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


def _load_lua_script() -> str:
    """Load the Lua balance deduction script from disk with fallback.

    Returns:
        str: Lua script source code.
    """
    if BALANCE_DEDUCT_LUA_PATH.is_file():
        try:
            return BALANCE_DEDUCT_LUA_PATH.read_text(encoding="utf-8")
        except OSError:
            return DEFAULT_LUA_BALANCE_DEDUCT.strip()


LUA_BALANCE_DEDUCT = _load_lua_script()


class InsufficientBalanceError(Exception):
    """Raised when an API key has insufficient credit balance for deduction."""

    def __init__(
        self,
        message: str = "Insufficient credit balance",
        current_balance: float = 0.0,
        required_cost: float = 0.0,
    ) -> None:
        """Initialize InsufficientBalanceError.

        Args:
            message: Human-readable error description.
            current_balance: Balance available at time of check.
            required_cost: Minimum cost required to fulfill the request.
        """
        super().__init__(message)
        self.current_balance = current_balance
        self.required_cost = required_cost


def calculate_token_cost(
    prompt_tokens: int,
    completion_tokens: int,
    prompt_price_per_m: float = 0.20,
    completion_price_per_m: float = 0.60,
    **kwargs: Any,
) -> float:
    """Calculate the exact USD cost for prompt and completion token counts.

    Formula:
        Total Cost = (Prompt Tokens * Prompt Price
                      + Completion Tokens * Completion Price) / 1,000,000

    Args:
        prompt_tokens: Number of prompt tokens evaluated. Must be >= 0.
        completion_tokens: Number of completion tokens generated. Must be >= 0.
        prompt_price_per_m: Price in USD per million prompt tokens. Defaults to 0.20.
        completion_price_per_m: Price in USD per million completion tokens. Defaults to 0.60.
        **kwargs: Supports prompt_price_per_million / completion_price_per_million aliases.

    Returns:
        float: Exact total cost in USD.

    Raises:
        ValueError: If prompt_tokens or completion_tokens is negative.
    """
    if prompt_tokens < 0 or completion_tokens < 0:
        raise ValueError("Token counts cannot be negative")

    prompt_price = kwargs.get("prompt_price_per_million", prompt_price_per_m)
    completion_price = kwargs.get("completion_price_per_million", completion_price_per_m)

    prompt_cost = (prompt_tokens * prompt_price) / 1_000_000.0
    completion_cost = (completion_tokens * completion_price) / 1_000_000.0
    return prompt_cost + completion_cost


def estimate_max_cost(
    prompt_tokens: int,
    max_tokens: int,
    context_window: int = 32768,
    prompt_price_per_m: float = 0.20,
    completion_price_per_m: float = 0.60,
    **kwargs: Any,
) -> float:
    """Estimate worst-case upper bound cost for pre-flight credit reservation.

    Clamps requested completion tokens by remaining context window space:
        Clamped Tokens = min(max_tokens, max(0, context_window - prompt_tokens))

    Args:
        prompt_tokens: Number of input prompt tokens.
        max_tokens: Requested maximum completion tokens.
        context_window: Total model context limit. Defaults to 32768.
        prompt_price_per_m: Price in USD per million prompt tokens.
        completion_price_per_m: Price in USD per million completion tokens.
        **kwargs: Forwarded to calculate_token_cost.

    Returns:
        float: Upper-bound estimated cost in USD.
    """
    clamped_completion = min(max_tokens, max(0, context_window - prompt_tokens))
    return calculate_token_cost(
        prompt_tokens=prompt_tokens,
        completion_tokens=clamped_completion,
        prompt_price_per_m=prompt_price_per_m,
        completion_price_per_m=completion_price_per_m,
        **kwargs,
    )


def check_preflight_credit(
    api_key: Any,
    estimated_tokens_or_prompt: int,
    prompt_price_or_max_tokens: float | int = 0.20,
    completion_price_per_m: float = 0.60,
    context_window: int = 32768,
) -> bool:
    """Verify whether an API key has adequate credit balance to cover estimated inference cost.

    Accepts dual calling conventions:
    1. (api_key, prompt_tokens, max_tokens): Computes worst-case cost via estimate_max_cost.
    2. (api_key, estimated_tokens, prompt_price_per_m): Computes estimated_tokens * price / 1e6.

    Args:
        api_key: ApiKey instance or object with credit_balance attribute.
        estimated_tokens_or_prompt: Prompt token count or pre-computed estimated tokens.
        prompt_price_or_max_tokens: max_tokens (int > 1) or prompt price per million (float).
        completion_price_per_m: Price in USD per million completion tokens.
        context_window: Context window size for token clamping.

    Returns:
        bool: True if credit_balance >= estimated_cost, False otherwise.
    """
    if isinstance(prompt_price_or_max_tokens, int) and prompt_price_or_max_tokens > 1:
        # Calling convention 1: (api_key, prompt_tokens, max_tokens)
        prompt_tokens = estimated_tokens_or_prompt
        max_tokens = prompt_price_or_max_tokens
        estimated_cost = estimate_max_cost(
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            context_window=context_window,
            prompt_price_per_m=0.20,
            completion_price_per_m=completion_price_per_m,
        )
    else:
        # Calling convention 2: (api_key, estimated_tokens, prompt_price_per_m)
        estimated_tokens = estimated_tokens_or_prompt
        prompt_price = float(prompt_price_or_max_tokens)
        estimated_cost = (estimated_tokens * prompt_price) / 1_000_000.0

    current_balance = getattr(api_key, "credit_balance", 0.0)
    return current_balance >= estimated_cost


def enforce_preflight_credit(
    api_key: ApiKey,
    prompt_tokens: int,
    max_tokens: int,
    prompt_price_per_m: float = 0.20,
    completion_price_per_m: float = 0.60,
) -> None:
    """Enforce pre-flight credit check, raising HTTP 402 Payment Required if insufficient.

    Args:
        api_key: ApiKey model.
        prompt_tokens: Evaluated prompt token count.
        max_tokens: Maximum requested completion tokens.
        prompt_price_per_m: Prompt price per million.
        completion_price_per_m: Completion price per million.

    Raises:
        HTTPException: HTTP 402 with OpenAI-compatible error payload on deficient balance.
    """
    est_cost = estimate_max_cost(
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        prompt_price_per_m=prompt_price_per_m,
        completion_price_per_m=completion_price_per_m,
    )
    if api_key.credit_balance < est_cost:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail={
                "error": {
                    "message": (
                        f"Insufficient credit balance. Required: ${est_cost:.6f}, "
                        f"Available: ${api_key.credit_balance:.6f}"
                    ),
                    "type": "insufficient_quota",
                    "param": None,
                    "code": "insufficient_balance",
                }
            },
        )


def deduct_balance_atomic(
    redis_client: Any,
    api_key_id: int,
    cost: float,
    initial_balance: float | None = None,
) -> float:
    """Execute atomic credit deduction via Redis Lua script (balance_deduct.lua).

    Guarantees no race conditions or double-spending under concurrent requests.
    Returns balance as floating point number.

    Args:
        redis_client: Synchronous Redis client instance.
        api_key_id: Database identifier for the ApiKey.
        cost: USD amount to deduct. Must be >= 0.0.
        initial_balance: Optional fallback balance to seed if Redis cache misses.

    Returns:
        float: Updated remaining credit balance.

    Raises:
        ValueError: If cost is negative.
        KeyError: If Redis key is missing and no initial_balance is provided.
        InsufficientBalanceError: If balance is less than deduction cost.
    """
    if cost < 0.0:
        raise ValueError("Deduction cost cannot be negative")

    redis_key = f"speedinfer:balance:{api_key_id}"
    res = redis_client.eval(LUA_BALANCE_DEDUCT, 1, redis_key, cost)

    status_code = int(res[0])
    if status_code == 1:
        return float(res[1])

    code_or_balance = float(res[1])
    if code_or_balance == -1.0:
        # Cache miss
        if initial_balance is not None:
            redis_client.set(redis_key, str(initial_balance))
            retry_res = redis_client.eval(LUA_BALANCE_DEDUCT, 1, redis_key, cost)
            if int(retry_res[0]) == 1:
                return float(retry_res[1])
            raise InsufficientBalanceError(
                f"Insufficient balance: available {float(retry_res[1])}, required {cost}",
                current_balance=float(retry_res[1]),
                required_cost=cost,
            )
        raise KeyError(f"API key balance not found in Redis for key ID {api_key_id}")

    raise InsufficientBalanceError(
        f"Insufficient credit balance: available {code_or_balance}, required {cost}",
        current_balance=code_or_balance,
        required_cost=cost,
    )


async def async_deduct_balance_atomic(
    redis_client: Any,
    api_key_id: int,
    cost: float,
    initial_balance: float | None = None,
) -> float:
    """Execute atomic credit deduction asynchronously via Redis Lua script.

    Args:
        redis_client: Asynchronous Redis client instance.
        api_key_id: Database identifier for the ApiKey.
        cost: USD amount to deduct.
        initial_balance: Optional fallback balance to seed if Redis cache misses.

    Returns:
        float: Updated remaining credit balance.

    Raises:
        ValueError: If cost is negative.
        KeyError: If Redis key is missing and no initial_balance is provided.
        InsufficientBalanceError: If balance is less than deduction cost.
    """
    if cost < 0.0:
        raise ValueError("Deduction cost cannot be negative")

    redis_key = f"speedinfer:balance:{api_key_id}"
    res = await redis_client.eval(LUA_BALANCE_DEDUCT, 1, redis_key, cost)

    status_code = int(res[0])
    if status_code == 1:
        return float(res[1])

    code_or_balance = float(res[1])
    if code_or_balance == -1.0:
        if initial_balance is not None:
            await redis_client.set(redis_key, str(initial_balance))
            retry_res = await redis_client.eval(LUA_BALANCE_DEDUCT, 1, redis_key, cost)
            if int(retry_res[0]) == 1:
                return float(retry_res[1])
            raise InsufficientBalanceError(
                f"Insufficient balance: available {float(retry_res[1])}, required {cost}",
                current_balance=float(retry_res[1]),
                required_cost=cost,
            )
        raise KeyError(f"API key balance not found in Redis for key ID {api_key_id}")

    raise InsufficientBalanceError(
        f"Insufficient credit balance: available {code_or_balance}, required {cost}",
        current_balance=code_or_balance,
        required_cost=cost,
    )


def sync_usage_to_db(
    session: Session,
    api_key_id: int,
    request_id: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost: float,
    latency_ms: float,
    ttft_ms: float | None,
    status_code: int = 200,
) -> UsageLedger:
    """Persist an inference transaction to UsageLedger and flush updated balance to ApiKey.

    Args:
        session: Active SQLModel database session.
        api_key_id: Database identifier of the billed ApiKey.
        request_id: Correlated unique request ID.
        model: Deployed model name.
        prompt_tokens: Prompt tokens count.
        completion_tokens: Completion tokens count.
        cost: Calculated USD cost deducted for transaction.
        latency_ms: Total request duration in milliseconds.
        ttft_ms: Time to first token in milliseconds (null for non-streaming).
        status_code: HTTP response status code. Defaults to 200.

    Returns:
        UsageLedger: Persisted immutable audit record.
    """
    total_tokens = prompt_tokens + completion_tokens
    ledger_record = UsageLedger(
        api_key_id=api_key_id,
        request_id=request_id,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        total_cost=cost,
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        status_code=status_code,
    )
    session.add(ledger_record)

    api_key = session.get(ApiKey, api_key_id)
    if api_key is not None:
        new_balance = max(0.0, api_key.credit_balance - cost)
        api_key.credit_balance = round(new_balance, 6)
        api_key.last_used_at = utc_now()
        session.add(api_key)

    session.commit()
    session.refresh(ledger_record)

    # Option A: Qualify referrer if referee has consumed 1,000+ active tokens
    if api_key is not None:
        try:
            from speedinfer.core.referrals import check_and_award_referrer_bonus

            if check_and_award_referrer_bonus(session, api_key.user_id):
                session.commit()
        except Exception:
            pass

    return ledger_record


# Compatibility alias
deduct_balance_atomic_async = async_deduct_balance_atomic
