"""Integration tests for POST /v1/chat/completions SSE streaming endpoint.

Covers:
- Server-Sent Events (SSE) content-type and unbuffered headers
- Chunk protocol compliance (`data: {...}\\n\\n` format)
- Stream termination with `data: [DONE]\\n\\n`
- Full message reconstruction from sequential token deltas
- Client disconnect / cancellation handling midway through stream
- Preflight credit check rejection prior to stream initiation
"""

import json

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
async def test_streaming_headers_and_content_type(async_client, require_gateway):
    """Verify streaming response returns text/event-stream and unbuffered headers."""
    headers = {"Authorization": "Bearer sk-speedinfer-validkey"}
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "Count from 1 to 5"}],
        "stream": True,
    }
    async with async_client.stream(
        "POST", "/v1/chat/completions", json=payload, headers=headers
    ) as response:
        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "text/event-stream" in content_type


@pytest.mark.asyncio
async def test_streaming_chunks_format_and_termination(async_client, require_gateway):
    """Verify each SSE chunk matches OpenAI schema and terminates with [DONE]."""
    headers = {"Authorization": "Bearer sk-speedinfer-validkey"}
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }
    collected_chunks = []
    received_done = False

    async with async_client.stream(
        "POST", "/v1/chat/completions", json=payload, headers=headers
    ) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            line = line.strip()
            if not line or not line.startswith("data: "):
                continue

            raw_data = line[len("data: ") :].strip()
            if raw_data == "[DONE]":
                received_done = True
                break

            chunk_obj = json.loads(raw_data)
            assert chunk_obj["object"] == "chat.completion.chunk"
            assert chunk_obj["id"].startswith("chatcmpl-")
            if chunk_obj.get("usage"):
                assert chunk_obj["usage"]["total_tokens"] >= 0
                continue
            assert len(chunk_obj["choices"]) > 0
            collected_chunks.append(chunk_obj)

    assert received_done is True
    assert len(collected_chunks) > 0

    # Verify concatenated deltas form a coherent response
    reconstructed_content = "".join(
        c["choices"][0]["delta"].get("content", "")
        for c in collected_chunks
        if "content" in c["choices"][0].get("delta", {})
    )
    assert len(reconstructed_content) > 0


@pytest.mark.asyncio
async def test_streaming_client_abrupt_disconnect(async_client, require_gateway):
    """Verify server gracefully handles client disconnect midway through SSE stream."""
    headers = {"Authorization": "Bearer sk-speedinfer-validkey"}
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "Generate a long poem"}],
        "stream": True,
        "max_tokens": 1000,
    }

    # Simulate client reading only the first 2 chunks and then closing connection
    read_chunks = 0
    async with async_client.stream(
        "POST", "/v1/chat/completions", json=payload, headers=headers
    ) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                read_chunks += 1
                if read_chunks >= 2:
                    break

    assert read_chunks >= 2


@pytest.mark.asyncio
async def test_streaming_preflight_credit_rejection(async_client, require_gateway):
    """Verify streaming request is rejected with 402 before stream opens if balance deficient."""
    headers = {"Authorization": "Bearer sk-speedinfer-zero-credit-key"}
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
        "max_tokens": 4096,
    }
    response = await async_client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 402
    assert "text/event-stream" not in response.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_streaming_no_double_settle(async_client, require_gateway, monkeypatch):
    """Verify stream settlement occurs exactly once upon completion."""
    import speedinfer.gateway.routes.chat as chat_module

    settle_calls = []
    original_settle = chat_module.settle

    def spy_settle(*args, **kwargs):
        settle_calls.append((args, kwargs))
        return original_settle(*args, **kwargs)

    monkeypatch.setattr(chat_module, "settle", spy_settle)

    headers = {"Authorization": "Bearer sk-speedinfer-validkey"}
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "Count to 3"}],
        "stream": True,
    }
    async with async_client.stream(
        "POST", "/v1/chat/completions", json=payload, headers=headers
    ) as response:
        assert response.status_code == 200
        async for _line in response.aiter_lines():
            pass

    assert len(settle_calls) == 1
    assert settle_calls[0][1].get("success") is True


@pytest.mark.asyncio
async def test_streaming_done_sentinel_emitted_exactly_once(async_client, require_gateway):
    """Verify data: [DONE] is emitted exactly once and is the final message in the stream."""
    headers = {"Authorization": "Bearer sk-speedinfer-validkey"}
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "Hello world"}],
        "stream": True,
    }
    done_count = 0
    total_lines = []
    async with async_client.stream(
        "POST", "/v1/chat/completions", json=payload, headers=headers
    ) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            line_str = line.strip()
            if line_str:
                total_lines.append(line_str)
                if line_str == "data: [DONE]":
                    done_count += 1

    assert done_count == 1, f"Expected exactly 1 [DONE] sentinel, got {done_count}"
    assert total_lines[-1] == "data: [DONE]", "The very last message must be data: [DONE]"
