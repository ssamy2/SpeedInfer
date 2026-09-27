"""Integration tests for POST /v1/completions legacy text completions endpoint.

Covers:
- OpenAI legacy text completions schema (id, object, choices, text, usage)
- Authentication and permission checks
- Input validation (empty prompt, invalid parameters)
- Metering and rate limiting integration
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
async def test_legacy_completions_missing_auth(async_client, require_gateway):
    """Verify missing Authorization header returns HTTP 401."""
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "prompt": "Once upon a time",
    }
    response = await async_client.post("/v1/completions", json=payload)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_legacy_completions_empty_prompt_rejected(async_client, require_gateway):
    """Verify empty string prompt returns HTTP 400 Bad Request."""
    headers = {"Authorization": "Bearer sk-speedinfer-validkey"}
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "prompt": "",
    }
    response = await async_client.post("/v1/completions", json=payload, headers=headers)
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_legacy_completions_happy_path(async_client, require_gateway):
    """Verify successful legacy completion matches OpenAI specification."""
    headers = {"Authorization": "Bearer sk-speedinfer-validkey"}
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "prompt": "Complete this sentence: Fast inference is",
        "max_tokens": 64,
        "temperature": 0.5,
    }
    response = await async_client.post("/v1/completions", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()

    assert data["id"].startswith("cmpl-")
    assert data["object"] == "text_completion"
    assert isinstance(data["created"], int)
    assert data["model"] == "Qwen/Qwen2.5-7B-Instruct"

    assert len(data["choices"]) > 0
    choice = data["choices"][0]
    assert "text" in choice
    assert isinstance(choice["text"], str)
    assert choice["finish_reason"] in {"stop", "length"}

    assert "usage" in data
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["completion_tokens"] > 0
    assert data["usage"]["total_tokens"] == (
        data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]
    )


@pytest.mark.asyncio
async def test_legacy_completions_insufficient_credit(async_client, require_gateway):
    """Verify HTTP 402 is returned when credit is exhausted."""
    headers = {"Authorization": "Bearer sk-speedinfer-zero-credit-key"}
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "prompt": "Generate text",
        "max_tokens": 1024,
    }
    response = await async_client.post("/v1/completions", json=payload, headers=headers)
    assert response.status_code == 402


@pytest.mark.asyncio
async def test_legacy_completions_no_double_settle(async_client, require_gateway, monkeypatch):
    """Verify settle is called exactly once on successful completion."""
    import speedinfer.gateway.routes.completions as comp_module

    settle_calls = []
    original_settle = comp_module.settle

    def spy_settle(*args, **kwargs):
        settle_calls.append((args, kwargs))
        return original_settle(*args, **kwargs)

    monkeypatch.setattr(comp_module, "settle", spy_settle)

    headers = {"Authorization": "Bearer sk-speedinfer-validkey"}
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "prompt": "Hello",
        "max_tokens": 16,
    }
    response = await async_client.post("/v1/completions", json=payload, headers=headers)
    assert response.status_code == 200
    assert len(settle_calls) == 1
    assert settle_calls[0][1].get("success") is True
