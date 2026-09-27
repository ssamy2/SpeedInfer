"""Integration tests for SpeedInfer frontend static serving and dual-auth models."""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from speedinfer.database.session import get_session
from speedinfer.engine.registry import BackendWorker
from speedinfer.gateway.app import app, get_registry


@pytest.fixture(autouse=True)
def isolated_database(db_session):
    app.dependency_overrides[get_session] = lambda: db_session
    registry = get_registry()
    if not registry.list_models():
        registry.register_model(
            name="test/model",
            base_model_path="test/model",
            backends=[BackendWorker(url="http://localhost:9999", worker_id="test")],
        )
    yield
    app.dependency_overrides.pop(get_session, None)


@pytest.mark.asyncio
async def test_frontend_spa_serving():
    """Verify that root / and /app endpoints serve the SPA index.html with 200 OK."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Root route
        res = await client.get("/")
        assert res.status_code == 200
        assert "<!DOCTYPE html>" in res.text
        assert "SpeedInfer" in res.text

        # 2. SPA /app route
        res_app = await client.get("/app")
        assert res_app.status_code == 200
        routes = (
            "/app/buckets",
            "/app/training/new",
            "/legal",
            "/legal/terms",
            "/legal/privacy",
        )
        for route in routes:
            assert (await client.get(route)).status_code == 200

        homepage = (await client.get("/")).text
        assert 'base_url="https://speedinfer.com/v1"' in homepage
        assert "localhost:8000" not in homepage
        assert 'class="hero-technical-links"' in homepage
        for linked_route in (
            "/architecture",
            "/nvidia",
            "/docs/inference-workers",
            "/docs/nvidia",
        ):
            assert f'href="{linked_route}"' in homepage

        docs_page = (await client.get("/docs")).text
        assert "https://speedinfer.com/v1" in docs_page
        assert "localhost:8000" not in docs_page

        for route, marker in (
            ("/architecture", "Control Plane"),
            ("/docs/inference-workers", "Connected inference workers"),
            ("/docs/inference-workers/nvidia", "NVIDIA GPU workers"),
            ("/docs/inference-workers/triton", "Triton HTTP v2"),
            ("/docs/inference-workers/tensorrt-llm", "TensorRT-LLM workers"),
            ("/nvidia", "Built for NVIDIA-accelerated inference"),
            ("/docs/nvidia", "Connect NVIDIA-backed workers"),
            ("/docs/nvidia/cuda", "Prepare an NVIDIA worker host"),
            ("/docs/nvidia/triton", "Connect Triton Inference Server"),
            ("/docs/nvidia/tensorrt-llm", "Serve TensorRT-LLM models"),
        ):
            response = await client.get(route)
            assert response.status_code == 200
            assert marker in response.text

        # Verify HEAD requests work for all public marketing and doc pages
        for head_route in (
            "/",
            "/product",
            "/platform",
            "/company/team",
            "/architecture",
            "/nvidia",
            "/docs",
            "/guide",
            "/robots.txt",
            "/sitemap.xml",
        ):
            head_res = await client.head(head_route)
            assert (
                head_res.status_code == 200
            ), f"HEAD {head_route} failed with {head_res.status_code}"

        # Verify robots.txt allows crawling
        res_robots = await client.get("/robots.txt")
        assert res_robots.status_code == 200
        assert "Allow: /" in res_robots.text
        assert "sitemap.xml" in res_robots.text

        # Verify sitemap.xml contains key public URLs
        res_sitemap = await client.get("/sitemap.xml")
        assert res_sitemap.status_code == 200
        assert "https://speedinfer.com/" in res_sitemap.text
        assert "https://speedinfer.com/product" in res_sitemap.text
        assert "https://speedinfer.com/company/team" in res_sitemap.text
        assert "https://speedinfer.com/nvidia" in res_sitemap.text

        # 3. Static CSS assets
        res_css = await client.get("/static/css/theme.css")
        assert res_css.status_code == 200
        assert "--bg-base" in res_css.text

        # 4. Static JS assets
        res_js = await client.get("/static/js/app.js")
        assert res_js.status_code == 200
        assert "SpeedInfer" in res_js.text
        assert "modalReturnFocus" in res_js.text

        # Contact UI module is served independently and loaded by app.js.
        res_contact_js = await client.get("/static/js/contact.js")
        assert res_contact_js.status_code == 200
        assert "initContact" in res_contact_js.text


@pytest.mark.asyncio
async def test_models_endpoint_accepts_jwt_and_api_key():
    """Verify that /v1/models accepts both user JWT bearer token and API key."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Register a test user with unique email
        unique_email = f"frontend-test-{uuid.uuid4().hex[:8]}@speedinfer.local"
        reg_payload = {
            "email": unique_email,
            "password": "Password123!",
            "name": "Frontend Tester",
            "initial_balance": 10.0,
            "create_api_key": True,
        }
        reg_res = await client.post("/v1/auth/register", json=reg_payload)
        assert reg_res.status_code == 201
        reg_data = reg_res.json()
        jwt_token = reg_data["access_token"]
        raw_key = reg_data["api_key"]["key"]

        # 1. Call /v1/models using JWT access token
        jwt_headers = {"Authorization": f"Bearer {jwt_token}"}
        res_jwt = await client.get("/v1/models", headers=jwt_headers)
        assert res_jwt.status_code == 200
        models_data = res_jwt.json()
        assert models_data["object"] == "list"
        assert len(models_data["data"]) >= 1

        # 2. Call /v1/models using API key
        key_headers = {"Authorization": f"Bearer {raw_key}"}
        res_key = await client.get("/v1/models", headers=key_headers)
        assert res_key.status_code == 200
        models_key_data = res_key.json()
        assert models_key_data["object"] == "list"
        assert len(models_key_data["data"]) >= 1
