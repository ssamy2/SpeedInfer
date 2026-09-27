"""Integration tests for scripts/test_inference.py CLI harness."""

from starlette.testclient import TestClient

import scripts.test_inference as cli
from speedinfer.gateway.app import app


def test_inference_script_end_to_end():
    """Verify test_inference.py functions against active application lifespan."""
    with TestClient(app, base_url="http://testserver") as client:
        headers = cli.get_headers("sk-speedinfer-test-key")

        # 1. Health check
        assert cli.check_health(client, "http://testserver") is True

        # 2. Balance inspection
        bal = cli.get_usage_balance(client, "http://testserver", headers)
        assert bal is not None
        assert bal > 0.0

        # 3. Non-streaming chat
        chat_res = cli.test_non_streaming_chat(
            client,
            "http://testserver",
            headers,
            "Qwen/Qwen2.5-7B-Instruct",
            "Hello",
        )
        assert "choices" in chat_res
        assert len(chat_res["choices"]) > 0

        # 4. Streaming chat
        stream_text = cli.test_streaming_chat(
            client,
            "http://testserver",
            headers,
            "Qwen/Qwen2.5-7B-Instruct",
            "Hello",
        )
        assert len(stream_text) > 0
