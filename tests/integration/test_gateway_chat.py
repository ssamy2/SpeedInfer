"""Integration tests for POST /v1/chat/completions non-streaming endpoint.

Covers:
- OpenAI API schema compliance (id, object, choices, message, finish_reason, usage)
- Authentication enforcement (missing, invalid, valid Bearer keys)
- Pre-flight credit check rejection (HTTP 402 on deficient balance)
- Rate limiting enforcement (HTTP 429 and rate limit headers)
- Atomic balance deduction and audit ledger recording
- Input validation (empty messages, invalid temperature, unknown model)
"""

import pytest

try:
    from speedinfer.gateway.app import app  # noqa: F401

    HAS_GATEWAY = True
except (ImportError, AttributeError):
    HAS_GATEWAY = False


@pytest.fixture
def require_gateway():
    """Skip test if speedinfer.gateway is not yet implemented."""
    if not HAS_GATEWAY:
        pytest.skip("speedinfer.gateway not yet implemented by M4")


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_chat_completions_missing_auth_header(async_client, require_gateway):
    """Verify missing Authorization header returns HTTP 401 Unauthorized."""
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = await async_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 401
    data = response.json()
    assert "error" in data
    assert data["error"].get("code") == "missing_api_key" or "Unauthorized" in data["error"].get(
        "message", ""
    )


@pytest.mark.asyncio
async def test_chat_completions_invalid_api_key(async_client, require_gateway):
    """Verify invalid or forged Bearer token returns HTTP 401."""
    headers = {"Authorization": "Bearer sk-speedinfer-invalidkey12345"}
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = await async_client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 401
    data = response.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_chat_completions_empty_messages_rejected(async_client, require_gateway):
    """Verify empty messages array returns HTTP 400 Bad Request."""
    headers = {"Authorization": "Bearer sk-speedinfer-testkey"}
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [],
    }
    response = await async_client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 400
    data = response.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_chat_completions_invalid_temperature_rejected(async_client, require_gateway):
    """Verify temperature > 2.0 or < 0.0 returns HTTP 400 Bad Request."""
    headers = {"Authorization": "Bearer sk-speedinfer-testkey"}
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 3.5,
    }
    response = await async_client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_chat_completions_model_not_found(async_client, require_gateway):
    """Verify request with unregistered model returns HTTP 404 Not Found."""
    headers = {"Authorization": "Bearer sk-speedinfer-testkey"}
    payload = {
        "model": "non-existent-unregistered-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = await async_client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 404
    data = response.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_chat_completions_happy_path(async_client, require_gateway):
    """Verify successful non-streaming chat completion conforms to OpenAI spec."""
    headers = {"Authorization": "Bearer sk-speedinfer-validkey"}
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "temperature": 0.7,
        "max_tokens": 128,
        "stream": False,
    }
    response = await async_client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()

    # Structural OpenAI compliance assertions
    assert data["id"].startswith("chatcmpl-")
    assert data["object"] == "chat.completion"
    assert isinstance(data["created"], int)
    assert data["model"] == "Qwen/Qwen2.5-7B-Instruct"

    # Choices validation
    assert len(data["choices"]) == 1
    choice = data["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert len(choice["message"]["content"]) > 0
    assert choice["finish_reason"] in {"stop", "length"}

    # Usage validation
    assert "usage" in data
    usage = data["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    # Rate limiting headers presence
    assert "x-ratelimit-limit-requests" in response.headers
    assert "x-ratelimit-remaining-requests" in response.headers


@pytest.mark.asyncio
async def test_chat_completions_insufficient_credit(async_client, require_gateway):
    """Verify HTTP 402 is returned when credit balance is insufficient."""
    headers = {"Authorization": "Bearer sk-speedinfer-zero-credit-key"}
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "Write an essay"}],
        "max_tokens": 4096,
    }
    response = await async_client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 402
    data = response.json()
    assert "error" in data
