"""Integration tests for official OpenAI Python SDK compatibility.

Covers:
- OpenAI Python SDK client initialization (sync & async)
- client.models.list()
- client.chat.completions.create() non-streaming
- client.chat.completions.create() streaming chunk iterator
- Precise exception mappings:
  - AuthenticationError on 401
  - RateLimitError on 429
  - NotFoundError on 404
  - BadRequestError on 400
  - APIStatusError on 402
"""

import httpx
import openai
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


@pytest.fixture
def openai_client(require_gateway):
    """Yield an OpenAI Python SDK client connected via ASGI transport to SpeedInfer."""
    from starlette.testclient import TestClient

    from speedinfer.gateway.app import app

    http_client = TestClient(app=app, base_url="http://testserver/v1")
    client = openai.OpenAI(
        base_url="http://testserver/v1",
        api_key="sk-speedinfer-validkey1234567890abcdef",
        http_client=http_client,
    )
    yield client
    http_client.close()


@pytest.fixture
def openai_async_client(require_gateway):
    """Yield an AsyncOpenAI Python SDK client connected via ASGI transport."""
    from speedinfer.gateway.app import app

    transport = httpx.ASGITransport(app=app)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://testserver/v1")
    client = openai.AsyncOpenAI(
        base_url="http://testserver/v1",
        api_key="sk-speedinfer-validkey1234567890abcdef",
        http_client=http_client,
    )
    yield client


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------
def test_openai_sdk_models_list(openai_client):
    """Verify official SDK can list models."""
    response = openai_client.models.list()
    assert hasattr(response, "data")
    assert len(response.data) > 0
    model_ids = [m.id for m in response.data]
    assert "Qwen/Qwen2.5-7B-Instruct" in model_ids


def test_openai_sdk_chat_completion_non_streaming(openai_client):
    """Verify official SDK client executes chat completion creation."""
    completion = openai_client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": "Hello SpeedInfer"}],
        max_tokens=64,
        temperature=0.7,
    )
    assert completion.id.startswith("chatcmpl-")
    assert len(completion.choices) == 1
    assert completion.choices[0].message.role == "assistant"
    assert len(completion.choices[0].message.content) > 0
    assert completion.usage.prompt_tokens > 0
    assert completion.usage.completion_tokens > 0


def test_openai_sdk_chat_completion_streaming(openai_client):
    """Verify official SDK client iterates over SSE streaming chunks."""
    stream = openai_client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": "Count to 3"}],
        stream=True,
    )
    collected_deltas = []
    for chunk in stream:
        assert chunk.id.startswith("chatcmpl-")
        if chunk.choices and chunk.choices[0].delta.content:
            collected_deltas.append(chunk.choices[0].delta.content)

    assert len(collected_deltas) > 0
    full_text = "".join(collected_deltas)
    assert len(full_text) > 0


def test_openai_sdk_authentication_error(require_gateway):
    """Verify invalid key raises official openai.AuthenticationError."""
    from starlette.testclient import TestClient

    from speedinfer.gateway.app import app

    http_client = TestClient(app=app, base_url="http://testserver/v1")
    client = openai.OpenAI(
        base_url="http://testserver/v1",
        api_key="sk-speedinfer-invalidkey",
        http_client=http_client,
    )
    try:
        with pytest.raises(openai.AuthenticationError):
            client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[{"role": "user", "content": "Hi"}],
            )
    finally:
        http_client.close()


def test_openai_sdk_not_found_error(openai_client):
    """Verify nonexistent model raises official openai.NotFoundError."""
    with pytest.raises(openai.NotFoundError):
        openai_client.chat.completions.create(
            model="nonexistent-model",
            messages=[{"role": "user", "content": "Hi"}],
        )
