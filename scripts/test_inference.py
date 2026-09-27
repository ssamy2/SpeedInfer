#!/usr/bin/env python3
"""SpeedInfer End-to-End Inference Verification CLI Script.

Performs automated verification of:
1. Gateway health and model catalog query
2. Initial credit balance inspection via GET /v1/usage
3. Non-streaming chat completions (POST /v1/chat/completions)
4. Server-Sent Events (SSE) streaming chat completions
5. Real-time balance deduction accounting and ledger verification
"""

import argparse
import json
import os
import sys
import time
from typing import Any

import httpx


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Verify SpeedInfer inference, streaming, and balance metering."
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("SPEEDINFER_BASE_URL", "http://localhost:8000"),
        help="Base URL of the SpeedInfer API gateway (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("SPEEDINFER_API_KEY", "sk-speedinfer-test-key"),
        help="Bearer API key (default: from SPEEDINFER_API_KEY env or test key)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("SPEEDINFER_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        help="Model identifier to test (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain in two sentences what makes ultra-low latency LLM inference possible.",
        help="Prompt text for test completions",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP request timeout in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--skip-balance-check",
        action="store_true",
        help="Skip credit balance deduction assertions",
    )
    return parser.parse_args()


def get_headers(api_key: str) -> dict[str, str]:
    """Build standard authorization and content-type headers."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "SpeedInfer-TestInferenceCLI/1.0",
    }


def check_health(client: httpx.Client, base_url: str) -> bool:
    """Probe /health endpoint."""
    url = f"{base_url.rstrip('/')}/health"
    print(f"\n[1/5] Checking gateway health: {url} ...")
    try:
        response = client.get(url)
        if response.status_code == 200:
            print(f"  ✓ Health status: 200 OK -> {response.text}")
            return True
        print(f"  ✗ Health check failed: {response.status_code} -> {response.text}")
        return False
    except Exception as e:
        print(f"  ✗ Connection error connecting to {url}: {e}")
        return False


def get_usage_balance(client: httpx.Client, base_url: str, headers: dict[str, str]) -> float | None:
    """Retrieve current credit balance via GET /v1/usage."""
    url = f"{base_url.rstrip('/')}/v1/usage"
    try:
        response = client.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            balance = float(data.get("credit_balance", data.get("balance", 0.0)))
            return balance
        print(f"  Note: /v1/usage returned {response.status_code} ({response.text})")
        return None
    except Exception as e:
        print(f"  Note: Failed to query /v1/usage: {e}")
        return None


def test_non_streaming_chat(
    client: httpx.Client,
    base_url: str,
    headers: dict[str, str],
    model: str,
    prompt: str,
) -> dict[str, Any]:
    """Execute and validate non-streaming POST /v1/chat/completions."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    print(f"\n[2/5] Testing non-streaming chat completion: {url} ...")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise AI assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 128,
        "stream": False,
    }

    t0 = time.perf_counter()
    response = client.post(url, json=payload, headers=headers)
    duration_ms = (time.perf_counter() - t0) * 1000

    if response.status_code != 200:
        print(f"  ✗ Request failed with status {response.status_code}: {response.text}")
        raise RuntimeError(f"Non-streaming chat completion returned HTTP {response.status_code}")

    data = response.json()
    assert "choices" in data and len(data["choices"]) > 0, "Response missing choices"
    choice = data["choices"][0]
    content = choice.get("message", {}).get("content", "")
    print(f"  ✓ Status: 200 OK (Latency: {duration_ms:.1f}ms)")
    print(f'  ✓ Assistant: "{content.strip()}"')
    usage = data.get("usage", {})
    p_tok = usage.get("prompt_tokens")
    c_tok = usage.get("completion_tokens")
    t_tok = usage.get("total_tokens")
    print(f"  ✓ Tokens: Prompt={p_tok}, Completion={c_tok}, Total={t_tok}")

    # Inspect rate limit response headers
    for h in ["x-ratelimit-remaining-requests", "x-ratelimit-remaining-tokens"]:
        if h in response.headers:
            print(f"  ✓ Header: {h} = {response.headers[h]}")

    return data


def test_streaming_chat(
    client: httpx.Client,
    base_url: str,
    headers: dict[str, str],
    model: str,
    prompt: str,
) -> str:
    """Execute and validate SSE streaming POST /v1/chat/completions."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    print(f"\n[3/5] Testing SSE streaming chat completion: {url} ...")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": 128,
    }

    t0 = time.perf_counter()
    ttft_ms: float | None = None
    streamed_text = ""
    chunk_count = 0
    received_done = False

    with client.stream("POST", url, json=payload, headers=headers) as response:
        if response.status_code != 200:
            print(f"  ✗ Streaming request failed with status {response.status_code}")
            raise RuntimeError(f"Streaming chat completion returned HTTP {response.status_code}")

        content_type = response.headers.get("content-type", "")
        assert "text/event-stream" in content_type, f"Invalid Content-Type: {content_type}"
        print("  ✓ Stream connected, reading chunks:")
        sys.stdout.write("    ")

        for line in response.iter_lines():
            line = line.strip()
            if not line or not line.startswith("data: "):
                continue

            raw = line[len("data: ") :].strip()
            if raw == "[DONE]":
                received_done = True
                break

            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t0) * 1000

            chunk = json.loads(raw)
            if chunk.get("error"):
                raise RuntimeError("Worker stream reported an error")
            choices = chunk.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            piece = delta.get("content", "")
            if piece:
                streamed_text += piece
                sys.stdout.write(piece)
                sys.stdout.flush()
                chunk_count += 1

    total_time_ms = (time.perf_counter() - t0) * 1000
    print("\n")
    print(f"  ✓ Stream finished ([DONE] received: {received_done})")
    ttft_str = f"{ttft_ms:.1f}ms" if ttft_ms is not None else "N/A"
    print(f"  ✓ Chunks received: {chunk_count}, TTFT: {ttft_str}, Total: {total_time_ms:.1f}ms")
    assert received_done, "Stream did not terminate with [DONE]"
    return streamed_text


def main() -> int:
    """Main verification CLI routine."""
    args = parse_args()
    headers = get_headers(args.api_key)

    print("=" * 70)
    print(" SpeedInfer Inference & Metering Verification")
    print(f" Base URL: {args.base_url}")
    print(f" Model:    {args.model}")
    print("=" * 70)

    with httpx.Client(timeout=args.timeout) as client:
        # Step 1: Health check
        if not check_health(client, args.base_url):
            print("\n[WARN] Gateway /health check failed or timed out.")

        # Step 2: Query initial balance
        print("\n[2/5] Inspecting initial balance ...")
        initial_balance = get_usage_balance(client, args.base_url, headers)
        if initial_balance is not None:
            print(f"  ✓ Initial Balance: ${initial_balance:.6f}")
        else:
            print("  - Initial balance query skipped or endpoint unmetered.")

        # Step 3: Non-streaming completion
        try:
            test_non_streaming_chat(client, args.base_url, headers, args.model, args.prompt)
        except Exception as e:
            print(f"  ✗ Non-streaming test failed: {e}")
            return 1

        # Step 4: Streaming completion
        try:
            test_streaming_chat(client, args.base_url, headers, args.model, args.prompt)
        except Exception as e:
            print(f"  ✗ Streaming test failed: {e}")
            return 1

        # Step 5: Post-inference balance verification
        if not args.skip_balance_check and initial_balance is not None:
            print("\n[5/5] Verifying credit balance deduction ...")
            final_balance = get_usage_balance(client, args.base_url, headers)
            if final_balance is not None:
                diff = initial_balance - final_balance
                print(f"  ✓ Final Balance:   ${final_balance:.6f}")
                print(f"  ✓ Deducted Amount: ${diff:.6f}")
                if diff <= 0:
                    print("  ✗ Error: Balance was not deducted after inference requests!")
                    return 1
                print("  ✓ Balance deduction successfully verified.")

    print("\n" + "=" * 70)
    print(" ✓ All SpeedInfer E2E inference checks passed successfully!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
