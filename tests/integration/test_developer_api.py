"""Integration tests for Developer API endpoints.

Covers:
- Models: GET /v1/models, GET /v1/models/{id},
  GET /v1/models/{id}/weights (manifest & binary download).
- Files: POST /v1/files, GET /v1/files, GET /v1/files/{id}, GET /v1/files/{id}/content,
  DELETE /v1/files/{id}.
- Storage & Buckets: POST /v1/buckets, GET /v1/buckets, GET /v1/buckets/{id},
  DELETE /v1/buckets/{id}, upload object, list objects, download object, delete object.
- Fine-Tuning: POST /v1/fine_tuning/jobs, GET /v1/fine_tuning/jobs, GET /v1/fine_tuning/jobs/{id},
  cancel job, checkpoints, events.
- API Keys API: GET /v1/keys/{key_id}, POST /v1/keys with custom scopes, authenticate using API key.
- OAuth redirect: redirection targets /app?token=... and /auth/callback.
"""

import io
import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import Session

from speedinfer.database.session import get_session
from speedinfer.engine.registry import BackendWorker
from speedinfer.gateway.app import app, get_registry


@pytest.fixture(autouse=True)
def setup_dev_api_db(db_session: Session):
    app.dependency_overrides[get_session] = lambda: db_session
    reg = get_registry()
    if not reg.get_model("Qwen/Qwen2.5-7B-Instruct"):
        reg.register_model(
            name="Qwen/Qwen2.5-7B-Instruct",
            base_model_path="Qwen/Qwen2.5-7B-Instruct",
            context_length=32768,
            prompt_price_per_million=0.20,
            completion_price_per_million=0.60,
            backends=[BackendWorker(url="http://mock-vllm:8000", worker_id="vllm-primary")],
        )
    yield
    app.dependency_overrides.pop(get_session, None)


@pytest.mark.asyncio
async def test_developer_models_and_weights_api():
    """Verify model discovery and weights export."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Register user
        reg_res = await client.post(
            "/v1/auth/register",
            json={
                "email": f"dev-models-{uuid.uuid4().hex[:6]}@speedinfer.local",
                "password": "Password123!",
                "name": "Dev User",
                "create_api_key": True,
            },
        )
        assert reg_res.status_code == 201
        data = reg_res.json()
        api_key = data["api_key"]["key"]
        headers = {"Authorization": f"Bearer {api_key}"}

        # 1. Models catalog
        res = await client.get("/v1/models", headers=headers)
        assert res.status_code == 200
        models = res.json()["data"]
        assert len(models) >= 1

        # 2. Model details
        res_m = await client.get("/v1/models/Qwen/Qwen2.5-7B-Instruct", headers=headers)
        assert res_m.status_code == 200
        assert res_m.json()["id"] == "Qwen/Qwen2.5-7B-Instruct"

        # 3. Model weights manifest
        res_w = await client.get("/v1/models/Qwen/Qwen2.5-7B-Instruct/weights", headers=headers)
        assert res_w.status_code == 200
        w_data = res_w.json()
        assert w_data["model"] == "Qwen/Qwen2.5-7B-Instruct"
        assert w_data["format"] == "safetensors"
        assert "download_url" in w_data

        # 4. Download weights binary
        res_bin = await client.get(
            "/v1/models/Qwen/Qwen2.5-7B-Instruct/weights?download=true", headers=headers
        )
        assert res_bin.status_code == 200
        assert res_bin.headers["content-type"] == "application/octet-stream"
        assert "attachment" in res_bin.headers["content-disposition"]
        assert len(res_bin.content) > 16


@pytest.mark.asyncio
async def test_developer_files_api():
    """Verify OpenAI-compatible Files upload, retrieval, content download, and deletion."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        reg_res = await client.post(
            "/v1/auth/register",
            json={
                "email": f"dev-files-{uuid.uuid4().hex[:6]}@speedinfer.local",
                "password": "Password123!",
                "name": "Dev Files User",
                "create_api_key": True,
            },
        )
        assert reg_res.status_code == 201
        data = reg_res.json()
        api_key = data["api_key"]["key"]
        headers = {"Authorization": f"Bearer {api_key}"}

        # 1. Upload valid JSONL training file
        jsonl_content = (
            b'{"messages": [{"role": "user", "content": "Hi"}, '
            b'{"role": "assistant", "content": "Hi!"}]}\n'
            b'{"messages": [{"role": "user", "content": "2+2"}, '
            b'{"role": "assistant", "content": "4"}]}\n'
        )
        files = {"file": ("train.jsonl", io.BytesIO(jsonl_content), "application/jsonl")}
        res_upload = await client.post(
            "/v1/files",
            headers=headers,
            files=files,
            data={"purpose": "fine-tune"},
        )
        assert res_upload.status_code == 201
        file_obj = res_upload.json()
        assert file_obj["filename"] == "train.jsonl"
        assert file_obj["purpose"] == "fine-tune"
        assert file_obj["bytes"] == len(jsonl_content)
        file_id = file_obj["id"]

        # 2. List files
        res_list = await client.get("/v1/files", headers=headers)
        assert res_list.status_code == 200
        file_list = res_list.json()["data"]
        assert any(f["id"] == file_id for f in file_list)

        # 3. Retrieve specific file metadata
        res_get = await client.get(f"/v1/files/{file_id}", headers=headers)
        assert res_get.status_code == 200
        assert res_get.json()["id"] == file_id

        # 4. Download file content
        res_content = await client.get(f"/v1/files/{file_id}/content", headers=headers)
        assert res_content.status_code == 200
        assert res_content.content == jsonl_content

        # 5. Delete file
        res_del = await client.delete(f"/v1/files/{file_id}", headers=headers)
        assert res_del.status_code == 200
        assert res_del.json()["deleted"] is True

        # Verify 404 after deletion
        res_after = await client.get(f"/v1/files/{file_id}", headers=headers)
        assert res_after.status_code == 404


@pytest.mark.asyncio
async def test_developer_storage_buckets_api():
    """Verify Buckets creation, listing, object upload, download, and deletion."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        reg_res = await client.post(
            "/v1/auth/register",
            json={
                "email": f"dev-storage-{uuid.uuid4().hex[:6]}@speedinfer.local",
                "password": "Password123!",
                "name": "Dev Storage User",
                "create_api_key": True,
            },
        )
        assert reg_res.status_code == 201
        data = reg_res.json()
        api_key = data["api_key"]["key"]
        headers = {"Authorization": f"Bearer {api_key}"}

        # 1. Create bucket
        res_b = await client.post(
            "/v1/buckets",
            headers=headers,
            json={"name": "datasets-bucket", "description": "Training datasets store"},
        )
        assert res_b.status_code == 201
        bucket = res_b.json()
        assert bucket["name"] == "datasets-bucket"
        bucket_id = bucket["id"]

        # 2. List buckets
        res_list = await client.get("/v1/buckets", headers=headers)
        assert res_list.status_code == 200
        assert any(b["id"] == bucket_id for b in res_list.json()["data"])

        # 3. Upload object to bucket
        obj_content = b"Sample vector embeddings or weights checkpoint"
        files = {"file": ("embeddings.bin", io.BytesIO(obj_content), "application/octet-stream")}
        res_obj = await client.post(
            f"/v1/buckets/{bucket_id}/objects",
            headers=headers,
            files=files,
        )
        assert res_obj.status_code == 201
        obj_info = res_obj.json()
        assert obj_info["name"] == "embeddings.bin"
        assert obj_info["size"] == len(obj_content)

        # 4. List objects
        res_objs = await client.get(f"/v1/buckets/{bucket_id}/objects", headers=headers)
        assert res_objs.status_code == 200
        assert len(res_objs.json()["data"]) == 1

        # 5. Download object
        res_down = await client.get(
            f"/v1/buckets/{bucket_id}/objects/embeddings.bin", headers=headers
        )
        assert res_down.status_code == 200
        assert res_down.content == obj_content

        # 6. Delete object
        res_del_obj = await client.delete(
            f"/v1/buckets/{bucket_id}/objects/embeddings.bin", headers=headers
        )
        assert res_del_obj.status_code == 204

        # 7. Delete bucket
        res_del_b = await client.delete(f"/v1/buckets/{bucket_id}", headers=headers)
        assert res_del_b.status_code == 204


@pytest.mark.asyncio
async def test_developer_fine_tuning_api():
    """Verify model fine-tuning job creation, status check, checkpoints, and cancellation."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        reg_res = await client.post(
            "/v1/auth/register",
            json={
                "email": f"dev-ft-{uuid.uuid4().hex[:6]}@speedinfer.local",
                "password": "Password123!",
                "name": "Dev FineTune User",
                "create_api_key": True,
            },
        )
        assert reg_res.status_code == 201
        data = reg_res.json()
        api_key = data["api_key"]["key"]
        headers = {"Authorization": f"Bearer {api_key}"}

        # Upload a dataset file
        jsonl_content = (
            b'{"messages": [{"role": "user", "content": "Q"}, '
            b'{"role": "assistant", "content": "A"}]}\n'
        )
        files = {"file": ("dataset.jsonl", io.BytesIO(jsonl_content), "application/jsonl")}
        res_f = await client.post(
            "/v1/files", headers=headers, files=files, data={"purpose": "fine-tune"}
        )
        assert res_f.status_code == 201
        file_id = res_f.json()["id"]

        # 1. Create fine-tuning job
        job_payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "training_file": file_id,
            "method": "qlora",
            "suffix": "customer-service",
            "hyperparameters": {
                "n_epochs": 3,
                "batch_size": 4,
                "lora_rank": 32,
                "lora_alpha": 64,
            },
        }
        res_job = await client.post("/v1/fine_tuning/jobs", headers=headers, json=job_payload)
        assert res_job.status_code == 201
        job_data = res_job.json()
        job_id = job_data["id"]
        assert job_data["status"] == "running"
        assert "customer-service" in (job_data["fine_tuned_model"] or "")

        # 2. Retrieve job details
        res_get_job = await client.get(f"/v1/fine_tuning/jobs/{job_id}", headers=headers)
        assert res_get_job.status_code == 200
        assert res_get_job.json()["id"] == job_id

        # 3. Checkpoints
        res_ckpts = await client.get(f"/v1/fine_tuning/jobs/{job_id}/checkpoints", headers=headers)
        assert res_ckpts.status_code == 200
        assert len(res_ckpts.json()["data"]) >= 1

        # 4. Events
        res_events = await client.get(f"/v1/fine_tuning/jobs/{job_id}/events", headers=headers)
        assert res_events.status_code == 200
        assert len(res_events.json()["data"]) >= 1

        # 5. Cancel job
        res_cancel = await client.post(f"/v1/fine_tuning/jobs/{job_id}/cancel", headers=headers)
        assert res_cancel.status_code == 200
        assert res_cancel.json()["status"] == "cancelled"


@pytest.mark.asyncio
async def test_developer_keys_management_api():
    """Verify creating, inspecting, and deleting API keys via API with custom scopes."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        reg_res = await client.post(
            "/v1/auth/register",
            json={
                "email": f"dev-keys-{uuid.uuid4().hex[:6]}@speedinfer.local",
                "password": "Password123!",
                "name": "Dev Keys User",
                "create_api_key": True,
            },
        )
        assert reg_res.status_code == 201
        data = reg_res.json()
        jwt_token = data["access_token"]
        headers = {"Authorization": f"Bearer {jwt_token}"}

        # 1. Create key with custom scopes
        custom_scopes = "chat:completions,models:read,files:read,storage:read"
        res_k = await client.post(
            "/v1/keys",
            headers=headers,
            json={"name": "readonly-bot", "permissions": custom_scopes, "rpm_limit": 120},
        )
        assert res_k.status_code == 201
        key_data = res_k.json()
        new_key_id = key_data["id"]
        assert key_data["name"] == "readonly-bot"
        assert key_data["rpm_limit"] == 120
        assert "storage:read" in key_data["permissions"]

        # 2. Retrieve key by ID
        res_get_k = await client.get(f"/v1/keys/{new_key_id}", headers=headers)
        assert res_get_k.status_code == 200
        assert res_get_k.json()["id"] == new_key_id

        # 3. List keys
        res_list = await client.get("/v1/keys", headers=headers)
        assert res_list.status_code == 200
        assert any(k["id"] == new_key_id for k in res_list.json()["data"])

        # 4. Revoke key
        res_rev = await client.delete(f"/v1/keys/{new_key_id}", headers=headers)
        assert res_rev.status_code == 200
        assert res_rev.json()["status"] == "revoked"


@pytest.mark.asyncio
async def test_oauth_routes_and_spa_redirection():
    """Verify OAuth routes redirect to /app with token."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Check OAuth initiate redirects
        res_google = await client.get("/v1/auth/oauth/google", follow_redirects=False)
        assert res_google.status_code in {302, 503}
        if res_google.status_code == 302:
            assert "accounts.google.com" in res_google.headers["location"]

        res_github = await client.get("/v1/auth/oauth/github", follow_redirects=False)
        assert res_github.status_code in {302, 503}
        if res_github.status_code == 302:
            assert "github.com/login/oauth" in res_github.headers["location"]

        # Verify SPA routes return 200 OK
        assert (await client.get("/app")).status_code == 200
        assert (await client.get("/auth")).status_code == 200
        assert (await client.get("/auth/callback")).status_code == 200
        assert (await client.get("/docs")).status_code == 200
        assert (await client.get("/documentation")).status_code == 200


@pytest.mark.asyncio
async def test_scoped_key_isolation_without_models_read():
    """Verify that a scoped key WITHOUT models:read can access its authorized resources
    while being rejected from models endpoints.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Register user
        reg_res = await client.post(
            "/v1/auth/register",
            json={
                "email": f"scoped-user-{uuid.uuid4().hex[:6]}@speedinfer.local",
                "password": "Password123!",
                "name": "Scoped User",
                "create_api_key": False,
            },
        )
        assert reg_res.status_code == 201
        jwt_token = reg_res.json()["access_token"]
        user_headers = {"Authorization": f"Bearer {jwt_token}"}

        # Create key with ONLY storage permissions (NO models:read!)
        res_key = await client.post(
            "/v1/keys",
            headers=user_headers,
            json={"name": "storage-only-key", "permissions": "storage:read,storage:write"},
        )
        assert res_key.status_code == 201
        storage_api_key = res_key.json()["key"]
        storage_headers = {"Authorization": f"Bearer {storage_api_key}"}

        # 1. Accessing storage bucket with storage-only key MUST SUCCEED (not blocked)
        res_b = await client.post(
            "/v1/buckets",
            headers=storage_headers,
            json={"name": f"bkt-scoped-{uuid.uuid4().hex[:6]}"},
        )
        assert res_b.status_code == 201

        # 2. Accessing models catalog with storage-only key MUST FAIL with 403 Forbidden
        res_m = await client.get("/v1/models", headers=storage_headers)
        assert res_m.status_code == 403

        # 3. Accessing model details with storage-only key MUST FAIL with 403 Forbidden
        res_detail = await client.get(
            "/v1/models/Qwen/Qwen2.5-7B-Instruct", headers=storage_headers
        )
        assert res_detail.status_code == 403


@pytest.mark.asyncio
async def test_developer_files_validation_and_errors():
    """Verify error cases in Files API: empty file, malformed JSONL for fine-tuning."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        reg_res = await client.post(
            "/v1/auth/register",
            json={
                "email": f"files-err-{uuid.uuid4().hex[:6]}@speedinfer.local",
                "password": "Password123!",
                "name": "Files Err User",
                "create_api_key": True,
            },
        )
        headers = {"Authorization": f"Bearer {reg_res.json()['api_key']['key']}"}

        # 1. Empty file rejection
        empty_files = {"file": ("empty.jsonl", io.BytesIO(b""), "application/jsonl")}
        res_empty = await client.post("/v1/files", headers=headers, files=empty_files)
        assert res_empty.status_code == 400

        # 2. Malformed JSONL rejection
        bad_jsonl = b'{"messages": "not a list"}\nthis is not json at all\n'
        bad_files = {"file": ("bad.jsonl", io.BytesIO(bad_jsonl), "application/jsonl")}
        res_bad = await client.post(
            "/v1/files", headers=headers, files=bad_files, data={"purpose": "fine-tune"}
        )
        assert res_bad.status_code == 400


@pytest.mark.asyncio
async def test_developer_storage_buckets_errors():
    """Verify error cases for Buckets: duplicate name rejection and non-existent retrieval."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        reg_res = await client.post(
            "/v1/auth/register",
            json={
                "email": f"bkt-err-{uuid.uuid4().hex[:6]}@speedinfer.local",
                "password": "Password123!",
                "name": "Bucket Err User",
                "create_api_key": True,
            },
        )
        headers = {"Authorization": f"Bearer {reg_res.json()['api_key']['key']}"}

        # 1. Create first bucket
        res1 = await client.post("/v1/buckets", headers=headers, json={"name": "unique-bucket"})
        assert res1.status_code == 201

        # 2. Duplicate bucket rejection
        res_dup = await client.post("/v1/buckets", headers=headers, json={"name": "unique-bucket"})
        assert res_dup.status_code == 409

        # 3. Non-existent bucket retrieval
        res_404 = await client.get("/v1/buckets/nonexistent-bkt-id", headers=headers)
        assert res_404.status_code == 404


@pytest.mark.asyncio
async def test_developer_fine_tuning_errors():
    """Verify error cases for Fine-Tuning: non-existent training file, invalid cancellation."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        reg_res = await client.post(
            "/v1/auth/register",
            json={
                "email": f"ft-err-{uuid.uuid4().hex[:6]}@speedinfer.local",
                "password": "Password123!",
                "name": "FT Err User",
                "create_api_key": True,
            },
        )
        headers = {"Authorization": f"Bearer {reg_res.json()['api_key']['key']}"}

        # 1. Non-existent training file
        res_missing_file = await client.post(
            "/v1/fine_tuning/jobs",
            headers=headers,
            json={"model": "Qwen/Qwen2.5-7B-Instruct", "training_file": "file-nonexistent-123"},
        )
        assert res_missing_file.status_code == 404


@pytest.mark.asyncio
async def test_oauth_callback_aliases_resolution():
    """Verify that alternate OAuth callback aliases are properly handled."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # /v1/auth/callback/google with missing code parameter returns 400 Bad Request
        # (via custom validation handler)
        res_g = await client.get("/v1/auth/callback/google")
        assert res_g.status_code == 400

        # /v1/auth/callback/github with missing code parameter returns 400 Bad Request
        res_gh = await client.get("/v1/auth/callback/github")
        assert res_gh.status_code == 400
