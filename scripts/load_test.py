#!/usr/bin/env python3
"""SpeedInfer Locust Load Testing & Rate Limit Benchmark Script.

Validates:
1. Low-latency chat completion throughput under concurrent load
2. Dual token-bucket rate limit enforcement (RPM/TPM) and HTTP 429 response handling
3. SSE streaming chunk delivery and TTFT latency measurement
4. Credit balance exhaustion rejection (HTTP 402)
5. Standard rate limit headers presence (x-ratelimit-remaining-*, retry-after)

Usage:
  locust -f scripts/load_test.py --host http://localhost:8000
  locust -f scripts/load_test.py --headless -u 20 -r 5 --run-time 30s --host http://localhost:8000
  python3 scripts/load_test.py --headless -u 10 -r 2 --run-time 15s --host http://localhost:8000
"""

import os
import time

from locust import HttpUser, between, events, task

# Configurable test settings from environment
DEFAULT_MODEL = os.getenv("SPEEDINFER_MODEL", "Qwen/Qwen2.5-7B-Instruct")
TEST_API_KEY = os.getenv("SPEEDINFER_API_KEY", "sk-speedinfer-loadtest-key")


class BaseSpeedInferUser(HttpUser):
    """Base user with authenticated headers."""

    abstract = True

    def on_start(self):
        self.headers = {
            "Authorization": f"Bearer {TEST_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "SpeedInfer-LocustLoadTester/1.0",
        }


class StandardChatUser(BaseSpeedInferUser):
    """Simulates regular end-user sending non-streaming conversational prompts."""

    wait_time = between(1.0, 3.0)

    @task(3)
    def chat_completion(self):
        """Send standard non-streaming chat completion request."""
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [
                {"role": "system", "content": "You are a concise assistant."},
                {
                    "role": "user",
                    "content": "List 3 principles of distributed systems.",
                },
            ],
            "max_tokens": 128,
            "temperature": 0.7,
            "stream": False,
        }

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/v1/chat/completions [non-streaming]",
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        response.success()
                    else:
                        response.failure("Response missing 'choices'")
                except Exception as e:
                    response.failure(f"JSON parse error: {e}")
            elif response.status_code in {429, 402}:
                # Rate limited or out of credits - valid behavior under high load
                response.success()
            else:
                err_snippet = response.text[:100]
                response.failure(f"Unexpected status: {response.status_code} ({err_snippet})")

    @task(1)
    def check_health(self):
        """Probe /health endpoint."""
        with self.client.get("/health", catch_response=True, name="/health") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")


class StreamingChatUser(BaseSpeedInferUser):
    """Simulates interactive UI client consuming Server-Sent Events (SSE) streams."""

    wait_time = between(2.0, 5.0)

    @task(2)
    def streaming_chat(self):
        """Send streaming chat completion request and measure TTFT."""
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": "Write a short haiku about speed."}],
            "max_tokens": 64,
            "stream": True,
        }

        start_time = time.perf_counter()
        ttft_recorded = False

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            headers=self.headers,
            stream=True,
            catch_response=True,
            name="/v1/chat/completions [streaming SSE]",
        ) as response:
            if response.status_code == 200:
                received_done = False
                chunks = 0
                for line in response.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                    if line_str.startswith("data: "):
                        if not ttft_recorded:
                            ttft = (time.perf_counter() - start_time) * 1000
                            events.request.fire(
                                request_type="SSE_TTFT",
                                name="/v1/chat/completions [TTFT]",
                                response_time=ttft,
                                response_length=len(line),
                                exception=None,
                                context=None,
                            )
                            ttft_recorded = True

                        raw = line_str[6:].strip()
                        if raw == "[DONE]":
                            received_done = True
                            break
                        chunks += 1

                if received_done and chunks > 0:
                    response.success()
                else:
                    response.failure(f"Stream incomplete (chunks={chunks}, done={received_done})")
            elif response.status_code in {429, 402}:
                response.success()
            else:
                response.failure(f"Streaming failed: {response.status_code}")


class BurstRateLimitTester(BaseSpeedInferUser):
    """Fires back-to-back requests with no wait time to intentionally trigger RPM/TPM limits."""

    wait_time = between(0.01, 0.05)

    @task
    def burst_request(self):
        """Rapid request to test rate limiter cutoff."""
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 8,
            "stream": False,
        }

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/v1/chat/completions [burst rate limit]",
        ) as response:
            if response.status_code == 200:
                # Should have rate limit headers
                assert "x-ratelimit-remaining-requests" in response.headers
                response.success()
            elif response.status_code == 429:
                # Rate limit rejection must contain retry-after
                if "retry-after" in response.headers:
                    response.success()
                else:
                    response.failure("HTTP 429 missing Retry-After header")
            elif response.status_code == 402:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")


if __name__ == "__main__":
    from locust.main import main

    main()
