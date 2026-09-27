"""Production-path contracts with a controlled HTTP worker (no synthetic proxy fallback)."""

import json
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException
from sqlmodel import select

from speedinfer.config import Settings
from speedinfer.core.auth import generate_api_key
from speedinfer.core.inference_billing import InferenceReservation, reserve, settle
from speedinfer.database.models import ApiKey, UsageLedger, User
from speedinfer.database.session import get_session
from speedinfer.engine.registry import BackendWorker, ModelRegistry
from speedinfer.gateway.app import app
from speedinfer.gateway.proxy import InferenceProxy
from speedinfer.gateway.redis import get_async_redis
from speedinfer.gateway.routes import chat, completions
from tests.conftest import TEST_PEPPER


@pytest.fixture
async def live_contract(db_session, async_mock_redis, monkeypatch):
    settings = Settings(environment="production", api_key_pepper=TEST_PEPPER)
    monkeypatch.setattr("speedinfer.gateway.proxy.get_settings", lambda: settings)
    monkeypatch.setattr("speedinfer.core.auth.get_settings", lambda: settings)
    user = User(email="contract@example.test", name="Contract")
    db_session.add(user)
    db_session.commit()
    raw, prefix, hashed = generate_api_key(pepper=TEST_PEPPER)
    key = ApiKey(
        user_id=user.id,
        name="contract",
        prefix=prefix,
        key_hash=hashed,
        credit_balance=1,
        permissions="chat:completions,completions,models:read,usage:read",
        rpm_limit=1000,
        tpm_limit=1000000,
    )
    db_session.add(key)
    db_session.commit()
    registry = ModelRegistry()
    registry.register_model(
        name="test/model",
        base_model_path="test/model",
        context_length=1024,
        backends=[BackendWorker(url="http://worker")],
    )
    proxy = InferenceProxy(registry)
    calls = []
    state = {"status": 200, "usage": True, "disconnect": False}

    def worker(request):
        calls.append(json.loads(request.content))
        if state["status"] != 200:
            return httpx.Response(state["status"], json={"error": "private worker detail"})
        usage = {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
        if calls[-1].get("stream"):
            chunks = [
                {
                    "id": "worker-1",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": "Real worker response"}}],
                }
            ]
            if state["usage"]:
                chunks.append({"choices": [], "usage": usage})
            text = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
            if not state["disconnect"]:
                text += "data: [DONE]\n\n"
            return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})
        legacy = request.url.path.endswith("/completions") and "/chat/" not in request.url.path
        return httpx.Response(
            200,
            json={
                "id": "reused-worker-id",
                "model": "test/model",
                "object": "text_completion" if legacy else "chat.completion",
                "created": 1,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        **(
                            {"text": "Real worker response"}
                            if legacy
                            else {
                                "message": {"role": "assistant", "content": "Real worker response"}
                            }
                        ),
                    }
                ],
                "usage": usage,
            },
        )

    proxy.http_client = httpx.AsyncClient(transport=httpx.MockTransport(worker))
    overrides = {
        get_session: lambda: db_session,
        get_async_redis: lambda: async_mock_redis,
        chat.get_model_registry: lambda: registry,
        chat.get_inference_proxy: lambda: proxy,
        completions.get_model_registry: lambda: registry,
        completions.get_inference_proxy: lambda: proxy,
    }
    app.dependency_overrides.update(overrides)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {raw}"},
    ) as client:
        yield client, key, calls, state, proxy
    for dependency in overrides:
        app.dependency_overrides.pop(dependency, None)
    await proxy.close()


def body(**kwargs):
    return {"model": "test/model", "messages": [{"role": "user", "content": "Hello"}], **kwargs}


@pytest.mark.asyncio
async def test_real_usage_and_unique_gateway_ledger(live_contract, db_session):
    client, key, calls, _, _ = live_contract
    for _ in range(2):
        response = await client.post("/v1/chat/completions", json=body(max_tokens=None))
        assert response.status_code == 200, response.text
        assert response.json()["usage"]["total_tokens"] == 17
    db_session.refresh(key)
    assert key.credit_balance == pytest.approx(1 - 2 * 0.0000054)
    assert len(db_session.exec(select(UsageLedger)).all()) == 2
    assert calls[0]["max_tokens"] == 128


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_zero_credit_prevents_worker_call(live_contract, db_session, stream):
    client, key, calls, _, _ = live_contract
    key.credit_balance = key.paid_balance = key.trial_balance = 0
    db_session.add(key)
    db_session.commit()
    response = await client.post("/v1/chat/completions", json=body(stream=stream))
    assert response.status_code == 402
    assert response.json()["error"]["code"] == "insufficient_balance"
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_backend_failure_refunds_without_mock_success(live_contract, db_session, stream):
    client, key, _, state, _ = live_contract
    state["status"] = 500
    response = await client.post("/v1/chat/completions", json=body(stream=stream))
    assert response.status_code == 503
    assert "private worker detail" not in response.text
    db_session.refresh(key)
    assert key.credit_balance == pytest.approx(1)
    assert not db_session.exec(select(UsageLedger)).all()


@pytest.mark.asyncio
async def test_stream_uses_worker_usage_and_measured_ttft(live_contract, db_session):
    client, key, calls, _, _ = live_contract
    response = await client.post("/v1/chat/completions", json=body(stream=True))
    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    assert calls[0]["stream_options"] == {"include_usage": True}
    ledger = db_session.exec(select(UsageLedger)).one()
    assert (ledger.prompt_tokens, ledger.completion_tokens) == (12, 5)
    assert 0 <= ledger.ttft_ms <= ledger.latency_ms
    db_session.refresh(key)
    assert key.credit_balance == pytest.approx(1 - 0.0000054)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["usage", "disconnect"])
async def test_unmetered_or_truncated_stream_is_refunded(live_contract, db_session, failure):
    client, key, _, state, _ = live_contract
    state[failure] = False if failure == "usage" else True
    response = await client.post("/v1/chat/completions", json=body(stream=True))
    assert "stream_failed" in response.text
    assert "data: [DONE]" not in response.text
    db_session.refresh(key)
    assert key.credit_balance == pytest.approx(1)
    assert not db_session.exec(select(UsageLedger)).all()


@pytest.mark.asyncio
async def test_scope_validation_legacy_and_limits(live_contract, db_session):
    client, key, calls, _, _ = live_contract
    for change in [{"max_tokens": -1}, {"max_tokens": 0}, {"n": 2}, {"top_p": 2}]:
        assert (await client.post("/v1/chat/completions", json=body(**change))).status_code == 400
    assert calls == []
    response = await client.post("/v1/completions", json={"model": "test/model", "prompt": "Hi"})
    assert response.status_code == 200
    response = await client.post(
        "/v1/completions", json={"model": "test/model", "prompt": "Hi", "stream": True}
    )
    assert response.status_code == 400
    key.permissions = "models:read"
    db_session.add(key)
    db_session.commit()
    assert (await client.post("/v1/chat/completions", json=body())).status_code == 403


@pytest.mark.asyncio
async def test_durable_holds_no_double_spend_or_double_refund(live_contract, db_session):
    _, key, _, _, _ = live_contract
    hold = reserve(db_session, key.id, 0.8)
    with pytest.raises(HTTPException) as error:
        reserve(db_session, key.id, 0.8)
    assert error.value.status_code == 402
    settle(db_session, hold, cost=0.1, success=True, model="test/model")
    settle(db_session, hold)
    db_session.refresh(key)
    assert key.credit_balance == pytest.approx(0.9)
    assert db_session.exec(select(InferenceReservation)).one().state == "settled"


@pytest.mark.asyncio
async def test_redis_outage_fails_closed(monkeypatch):
    import speedinfer.gateway.redis as redis_module

    settings = Settings(environment="production", api_key_pepper=TEST_PEPPER)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setattr(redis_module, "get_settings", lambda: settings)
    monkeypatch.setattr(redis_module, "_async_redis_client", None)
    client = AsyncMock()
    client.ping.side_effect = ConnectionError("offline")
    monkeypatch.setattr("redis.asyncio.from_url", lambda *a, **k: client)
    with pytest.raises(HTTPException) as error:
        await redis_module.get_async_redis()
    assert error.value.status_code == 503


@pytest.mark.asyncio
async def test_invalid_model_revoked_key_and_rate_limit(live_contract, db_session):
    client, key, calls, _, _ = live_contract
    assert (
        await client.post("/v1/chat/completions", json=body(model="unknown"))
    ).status_code == 404
    key.tpm_limit = 1
    db_session.add(key)
    db_session.commit()
    response = await client.post("/v1/chat/completions", json=body())
    assert response.status_code == 429
    assert calls == []
    key.is_active = False
    db_session.add(key)
    db_session.commit()
    assert (await client.post("/v1/chat/completions", json=body())).status_code == 401


@pytest.mark.asyncio
async def test_missing_worker_never_returns_synthetic_text(live_contract, db_session):
    client, key, calls, _, proxy = live_contract
    proxy.registry.get_model("test/model").backends.clear()
    response = await client.post("/v1/chat/completions", json=body())
    assert response.status_code == 503
    assert not calls
    db_session.refresh(key)
    assert key.credit_balance == pytest.approx(1)


def test_reservations_across_independent_connections(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from sqlalchemy import create_engine
    from sqlmodel import Session, SQLModel

    engine = create_engine(
        f"sqlite:///{tmp_path / 'holds.db'}",
        connect_args={"check_same_thread": False, "timeout": 20},
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        user = User(email="parallel@example.test")
        session.add(user)
        session.commit()
        key = ApiKey(
            user_id=user.id,
            name="parallel",
            prefix="sk-speedinfer-parallel",
            key_hash="a" * 64,
            credit_balance=1,
        )
        session.add(key)
        session.commit()
        key_id = key.id

    def attempt(_):
        with Session(engine) as session:
            try:
                return reserve(session, key_id, 0.6)
            except HTTPException as exc:
                assert exc.status_code == 402
                return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(attempt, range(8)))
    assert sum(item is not None for item in results) == 1
    with Session(engine) as session:
        assert session.get(ApiKey, key_id).credit_balance == pytest.approx(0.4)
        hold = next(item for item in results if item)
        settle(session, hold)
        assert session.get(ApiKey, key_id).credit_balance == pytest.approx(1)
    engine.dispose()


@pytest.mark.asyncio
async def test_public_preview_metadata_and_guide(live_contract):
    client, *_ = live_contract
    response = await client.get("/")
    assert 'property="og:image"' in response.text
    assert "speedinfer-social-v1.png" in response.text
    assert "Sub-100ms" not in response.text
    assert "does not retain prompt or completion content by default" in response.text
    image = await client.get("/static/assets/speedinfer-social-v1.png")
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"
    guide = await client.get("/guide")
    assert guide.status_code == 200
    assert "https://speedinfer.com/v1" in guide.text
    assert "402" in guide.text

    # Test /company/team page
    team_resp = await client.get("/company/team")
    assert team_resp.status_code == 200
    assert "Our Team & Company" in team_resp.text
    assert "Sami Mahmoud" in team_resp.text
    assert "Hamza Ibrahim Khalil El-Geziry" in team_resp.text
    assert "Megsy for Digital Platforms Development and E-Commerce L.L.C" in team_resp.text

    # Test redirects
    redir_team = await client.get("/team", follow_redirects=False)
    assert redir_team.status_code == 301
    assert redir_team.headers["location"] == "/company/team"

    redir_company = await client.get("/company", follow_redirects=False)
    assert redir_company.status_code == 301
    assert redir_company.headers["location"] == "/company/team"

    # Test /product and /platform pages
    product_resp = await client.get("/product")
    assert product_resp.status_code == 200
    assert "SpeedInfer Platform" in product_resp.text
    assert "Inference Gateway" in product_resp.text
    assert "Model Registry" in product_resp.text
    assert "Continuous iteration-level batching" in product_resp.text
    assert "PagedAttention" in product_resp.text
    assert "deployment-aware data handling" in product_resp.text

    platform_resp = await client.get("/platform")
    assert platform_resp.status_code == 200
    assert "SpeedInfer Platform" in platform_resp.text




@pytest.mark.asyncio
@pytest.mark.parametrize("available", [False, True])
async def test_readiness_checks_actual_catalog(
    live_contract, monkeypatch, async_mock_redis, available
):
    import speedinfer.gateway.routes.health as health

    client, *_ = live_contract
    upstream = AsyncMock()
    upstream.__aenter__.return_value = upstream
    upstream.get.return_value = httpx.Response(
        200,
        json={"data": [{"id": "test/model"}] if available else []},
        request=httpx.Request("GET", "http://worker/v1/models"),
    )
    settings = Settings(
        environment="production",
        api_key_pepper=TEST_PEPPER,
        default_model="test/model",
        vllm_api_key="worker-only-secret",
    )
    monkeypatch.setattr(health, "get_settings", lambda: settings)
    monkeypatch.setattr(health, "get_async_redis", AsyncMock(return_value=async_mock_redis))
    constructor = __import__("unittest.mock", fromlist=["Mock"]).Mock(return_value=upstream)
    monkeypatch.setattr(health.httpx, "AsyncClient", constructor)
    response = await client.get("/ready")
    assert response.status_code == (200 if available else 503)
    assert response.json()["checks"]["inference"] == available
    assert "worker-only-secret" not in response.text
    assert constructor.call_args.kwargs["headers"]["Authorization"] == "Bearer worker-only-secret"


@pytest.mark.asyncio
async def test_timeout_and_invalid_usage_refund(live_contract, db_session):
    client, key, _, _, proxy = live_contract
    await proxy.http_client.aclose()

    def timeout(request):
        raise httpx.ReadTimeout("test timeout", request=request)

    proxy.http_client = httpx.AsyncClient(transport=httpx.MockTransport(timeout))
    response = await client.post("/v1/chat/completions", json=body())
    assert response.status_code == 503
    db_session.refresh(key)
    assert key.credit_balance == pytest.approx(1)


@pytest.mark.asyncio
async def test_unknown_parameters_are_not_silently_ignored(live_contract):
    client, _, calls, _, _ = live_contract
    response = await client.post("/v1/chat/completions", json=body(tensorrt_llm_runtime=True))
    assert response.status_code == 400
    assert calls == []
