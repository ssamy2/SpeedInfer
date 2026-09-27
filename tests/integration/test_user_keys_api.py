"""Integration tests for user authentication, API key lifecycle, and inference gateway.

Verifies:
- POST /v1/auth/register: User account registration, default trial balance, and default API key.
- POST /v1/auth/login: Credential authentication, wrong password rejection, and inactive accounts.
- GET /v1/auth/me: JWT-authenticated user inspection and balance calculation.
- POST /v1/keys: Dynamic API key provisioning with plaintext secret returned once.
- GET /v1/keys: User API key listing with masked prefixes and multi-tenant isolation.
- DELETE /v1/keys/{key_id}: API key revocation and protection against unauthorized cross-user ops.
- POST /v1/chat/completions: Execution of inference with a newly minted API key, and rejection
  after the key is revoked.
"""

from collections.abc import Generator

import httpx
import pytest
from sqlmodel import Session, select

from speedinfer.config import get_settings
from speedinfer.database.models import ApiKey, ModelVersion
from speedinfer.database.session import get_session
from speedinfer.engine.registry import BackendWorker
from speedinfer.gateway.app import app, get_registry


@pytest.fixture(autouse=True)
def setup_db(db_session: Session) -> Generator[None, None, None]:
    """Configure in-memory database and default model for integration tests."""
    settings = get_settings()
    reg = get_registry()
    model_name = settings.default_model
    if reg.get_model(model_name) is None:
        reg.register_model(
            name=model_name,
            base_model_path=model_name,
            context_length=settings.max_request_tokens,
            prompt_price_per_million=settings.prompt_price_per_million,
            completion_price_per_million=settings.completion_price_per_million,
            backends=[BackendWorker(url=settings.vllm_base_url, worker_id="vllm-primary")],
        )

    existing = db_session.exec(select(ModelVersion).where(ModelVersion.name == model_name)).first()
    if existing is None:
        db_m = ModelVersion(
            name=model_name,
            base_model_path=model_name,
            lifecycle_status="active",
            context_length=settings.max_request_tokens,
            prompt_price_per_million=settings.prompt_price_per_million,
            completion_price_per_million=settings.completion_price_per_million,
        )
        db_session.add(db_m)
        db_session.commit()

    app.dependency_overrides[get_session] = lambda: db_session
    yield
    app.dependency_overrides.pop(get_session, None)


@pytest.mark.asyncio
async def test_user_registration_happy_path(async_client: httpx.AsyncClient) -> None:
    """Verify new user registration yields JWT access token and funded default API key."""
    payload = {
        "email": "new_dev@speedinfer.local",
        "password": "SecurePassword2026!",
        "name": "New Developer",
        "initial_balance": 15.0,
        "create_api_key": True,
    }
    response = await async_client.post("/v1/auth/register", json=payload)
    assert response.status_code == 201
    data = response.json()

    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == "new_dev@speedinfer.local"
    assert data["user"]["name"] == "New Developer"
    assert data["user"]["balance"] == get_settings().trial_credit_balance
    assert data["user"]["is_active"] is True

    # Default API key validation
    assert data["api_key"] is not None
    api_key_info = data["api_key"]
    assert api_key_info["key"].startswith("sk-speedinfer-")
    assert api_key_info["name"] == "default"
    assert api_key_info["credit_balance"] == get_settings().trial_credit_balance
    assert api_key_info["is_active"] is True


@pytest.mark.asyncio
async def test_user_registration_duplicate_email(async_client: httpx.AsyncClient) -> None:
    """Verify duplicate email registration returns HTTP 400 with email_already_exists."""
    payload = {
        "email": "duplicate@speedinfer.local",
        "password": "Password123!",
        "name": "First User",
    }
    res1 = await async_client.post("/v1/auth/register", json=payload)
    assert res1.status_code == 201

    res2 = await async_client.post("/v1/auth/register", json=payload)
    assert res2.status_code == 400
    err = res2.json()
    assert err["error"]["code"] == "email_already_exists"


@pytest.mark.asyncio
async def test_user_registration_short_password(async_client: httpx.AsyncClient) -> None:
    """Verify passwords shorter than 6 characters are rejected with validation error."""
    payload = {
        "email": "short_pwd@speedinfer.local",
        "password": "123",
    }
    response = await async_client.post("/v1/auth/register", json=payload)
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_user_registration_without_api_key(async_client: httpx.AsyncClient) -> None:
    """Verify registration with create_api_key=False creates user with zero balance and no key."""
    payload = {
        "email": "no_key@speedinfer.local",
        "password": "Password12345!",
        "create_api_key": False,
    }
    response = await async_client.post("/v1/auth/register", json=payload)
    assert response.status_code == 201
    data = response.json()

    assert data["api_key"] is None
    assert data["user"]["balance"] == 0.0


@pytest.mark.asyncio
async def test_user_login_success_and_failure(async_client: httpx.AsyncClient) -> None:
    """Verify login authentication with valid and invalid credentials."""
    reg_payload = {
        "email": "login_test@speedinfer.local",
        "password": "MySecretPassword123!",
        "name": "Login Tester",
    }
    reg_res = await async_client.post("/v1/auth/register", json=reg_payload)
    assert reg_res.status_code == 201

    # 1. Login with correct password
    login_res = await async_client.post(
        "/v1/auth/login",
        json={"email": "login_test@speedinfer.local", "password": "MySecretPassword123!"},
    )
    assert login_res.status_code == 200
    login_data = login_res.json()
    assert "access_token" in login_data
    assert login_data["user"]["email"] == "login_test@speedinfer.local"

    # 2. Login with incorrect password
    wrong_pwd_res = await async_client.post(
        "/v1/auth/login",
        json={"email": "login_test@speedinfer.local", "password": "WrongPassword999!"},
    )
    assert wrong_pwd_res.status_code == 401
    assert wrong_pwd_res.json()["error"]["code"] == "invalid_credentials"

    # 3. Login with nonexistent email
    nonexistent_res = await async_client.post(
        "/v1/auth/login",
        json={"email": "ghost@speedinfer.local", "password": "SomePassword123!"},
    )
    assert nonexistent_res.status_code == 401
    assert nonexistent_res.json()["error"]["code"] == "invalid_credentials"


@pytest.mark.asyncio
async def test_auth_me_endpoint(async_client: httpx.AsyncClient) -> None:
    """Verify GET /v1/auth/me returns authenticated user details and active balance."""
    reg_payload = {
        "email": "me_test@speedinfer.local",
        "password": "Password123!",
        "name": "Me Tester",
        "initial_balance": 25.0,
    }
    reg_res = await async_client.post("/v1/auth/register", json=reg_payload)
    assert reg_res.status_code == 201
    token = reg_res.json()["access_token"]

    # Authenticated call
    headers = {"Authorization": f"Bearer {token}"}
    me_res = await async_client.get("/v1/auth/me", headers=headers)
    assert me_res.status_code == 200
    me_data = me_res.json()
    assert me_data["email"] == "me_test@speedinfer.local"
    assert me_data["balance"] == 0.0  # No key is created implicitly.

    # Unauthenticated call
    unauth_res = await async_client.get("/v1/auth/me")
    assert unauth_res.status_code == 401


@pytest.mark.asyncio
async def test_keys_lifecycle_and_multi_tenant_isolation(
    async_client: httpx.AsyncClient,
) -> None:
    """Verify API key creation, listing, revocation, and multi-tenant isolation."""
    # 1. Register User A
    user_a_res = await async_client.post(
        "/v1/auth/register",
        json={"email": "user_a@speedinfer.local", "password": "PasswordUserA1!", "name": "User A"},
    )
    token_a = user_a_res.json()["access_token"]
    headers_a = {"Authorization": f"Bearer {token_a}"}

    # 2. Register User B
    user_b_res = await async_client.post(
        "/v1/auth/register",
        json={"email": "user_b@speedinfer.local", "password": "PasswordUserB1!", "name": "User B"},
    )
    token_b = user_b_res.json()["access_token"]
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # 3. User A explicitly creates their first API key.
    create_key_payload = {
        "name": "production-key",
        "rpm_limit": 120,
        "tpm_limit": 120_000,
        "credit_balance": 50.0,
    }
    key_res = await async_client.post("/v1/keys", json=create_key_payload, headers=headers_a)
    assert key_res.status_code == 201
    new_key_data = key_res.json()
    assert new_key_data["name"] == "production-key"
    assert new_key_data["rpm_limit"] == 120
    assert new_key_data["tpm_limit"] == 120_000
    assert new_key_data["credit_balance"] == get_settings().trial_credit_balance
    key_a_id = new_key_data["id"]

    # 4. User A sees only the explicitly created key.
    list_a_res = await async_client.get("/v1/keys", headers=headers_a)
    assert list_a_res.status_code == 200
    list_a = list_a_res.json()
    assert list_a["total"] == 1
    assert any(k["id"] == key_a_id for k in list_a["data"])

    # 5. Multi-tenant isolation: User B lists keys (must NOT see User A's keys)
    list_b_res = await async_client.get("/v1/keys", headers=headers_b)
    assert list_b_res.status_code == 200
    list_b = list_b_res.json()
    assert list_b["total"] == 0
    assert not any(k["id"] == key_a_id for k in list_b["data"])

    # 6. User B cannot delete User A's key (HTTP 404)
    cross_delete_res = await async_client.delete(f"/v1/keys/{key_a_id}", headers=headers_b)
    assert cross_delete_res.status_code == 404

    # 7. User A revokes their key
    del_res = await async_client.delete(f"/v1/keys/{key_a_id}", headers=headers_a)
    assert del_res.status_code == 200
    del_data = del_res.json()
    assert del_data["deleted"] is True
    assert del_data["status"] == "revoked"

    # 8. User A lists keys again - revoked key is listed with status 'revoked'
    list_a_after = await async_client.get("/v1/keys", headers=headers_a)
    revoked_item = next(k for k in list_a_after.json()["data"] if k["id"] == key_a_id)
    assert revoked_item["status"] == "revoked"
    assert revoked_item["is_active"] is False


@pytest.mark.asyncio
async def test_created_api_key_works_against_chat_completions(
    async_client: httpx.AsyncClient,
    db_session: Session,
) -> None:
    """Verify newly generated API key executes chat completions and fails once revoked."""
    # 1. Register new user
    reg_res = await async_client.post(
        "/v1/auth/register",
        json={
            "email": "chat_tester@speedinfer.local",
            "password": "PasswordChat123!",
            "name": "Chat Tester",
            "create_api_key": False,
        },
    )
    assert reg_res.status_code == 201
    jwt_token = reg_res.json()["access_token"]
    jwt_headers = {"Authorization": f"Bearer {jwt_token}"}

    # 2. Generate a dedicated API key for inference
    key_payload = {
        "name": "llm-client-key",
        "credit_balance": 20.0,
        "rpm_limit": 60,
        "tpm_limit": 60_000,
    }
    key_resp = await async_client.post("/v1/keys", json=key_payload, headers=jwt_headers)
    assert key_resp.status_code == 201
    key_data = key_resp.json()
    raw_api_key = key_data["key"]
    key_id = key_data["id"]

    # Fund the key directly; users cannot self-mint credit via API
    target_key = db_session.get(ApiKey, key_id)
    assert target_key is not None
    target_key.paid_balance = 20.0
    target_key.credit_balance = 20.0
    db_session.add(target_key)
    db_session.commit()

    # 3. Call POST /v1/chat/completions using the newly generated API key
    inference_headers = {"Authorization": f"Bearer {raw_api_key}"}
    chat_payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "temperature": 0.7,
        "max_tokens": 64,
        "stream": False,
    }
    chat_res = await async_client.post(
        "/v1/chat/completions", json=chat_payload, headers=inference_headers
    )
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert chat_data["id"].startswith("chatcmpl-")
    assert len(chat_data["choices"]) == 1
    assert "4" in chat_data["choices"][0]["message"]["content"]
    assert chat_data["usage"]["total_tokens"] > 0

    # 4. Revoke the API key
    del_res = await async_client.delete(f"/v1/keys/{key_id}", headers=jwt_headers)
    assert del_res.status_code == 200

    # 5. Subsequent inference calls with the revoked key must be rejected with HTTP 401
    rejected_chat_res = await async_client.post(
        "/v1/chat/completions", json=chat_payload, headers=inference_headers
    )
    assert rejected_chat_res.status_code == 401
    err = rejected_chat_res.json()
    assert err["error"]["code"] in {"api_key_inactive", "invalid_api_key"}


@pytest.mark.asyncio
async def test_non_admin_cannot_escalate_to_admin_scope(
    async_client: httpx.AsyncClient,
) -> None:
    """Verify regular non-admin users cannot grant admin scope to API keys (HTTP 403)."""
    reg_res = await async_client.post(
        "/v1/auth/register",
        json={"email": "regular_user@speedinfer.local", "password": "Password123!"},
    )
    assert reg_res.status_code == 201
    token = reg_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Attempt to request 'admin' scope
    payload = {"name": "hacked-key", "permissions": "chat:completions,admin"}
    res = await async_client.post("/v1/keys", json=payload, headers=headers)
    assert res.status_code == 403
    err = res.json()
    assert err["error"]["code"] == "insufficient_scope"


@pytest.mark.asyncio
async def test_admin_can_create_admin_key(
    async_client: httpx.AsyncClient,
    db_session: Session,
) -> None:
    """Verify administrative users can grant admin scopes to API keys."""
    from speedinfer.core.security import hash_password
    from speedinfer.database.models import User

    admin_user = User(
        email="root_admin@speedinfer.local",
        name="Root Admin",
        password_hash=hash_password("RootAdminPassword123!"),
        is_active=True,
        is_admin=True,
    )
    db_session.add(admin_user)
    db_session.commit()
    db_session.refresh(admin_user)

    login_res = await async_client.post(
        "/v1/auth/login",
        json={"email": "root_admin@speedinfer.local", "password": "RootAdminPassword123!"},
    )
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    payload = {"name": "admin-key", "permissions": "admin,chat:completions"}
    key_res = await async_client.post("/v1/keys", json=payload, headers=headers)
    assert key_res.status_code == 201
    key_data = key_res.json()
    assert "admin" in key_data["permissions"]


@pytest.mark.asyncio
async def test_keys_pagination(async_client: httpx.AsyncClient) -> None:
    """Verify limit and offset query parameters in GET /v1/keys."""
    reg_res = await async_client.post(
        "/v1/auth/register",
        json={
            "email": "pager@speedinfer.local",
            "password": "Password123!",
            "create_api_key": False,
        },
    )
    token = reg_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Create 3 keys
    for i in range(3):
        await async_client.post("/v1/keys", json={"name": f"key-{i}"}, headers=headers)

    # Page size 2, offset 0
    page1 = await async_client.get("/v1/keys?limit=2&offset=0", headers=headers)
    assert page1.status_code == 200
    data1 = page1.json()
    assert data1["total"] == 3
    assert len(data1["data"]) == 2

    # Page size 2, offset 2
    page2 = await async_client.get("/v1/keys?limit=2&offset=2", headers=headers)
    assert page2.status_code == 200
    data2 = page2.json()
    assert data2["total"] == 3
    assert len(data2["data"]) == 1


@pytest.mark.asyncio
async def test_trailing_slash_routes(async_client: httpx.AsyncClient) -> None:
    """Verify auth and keys endpoints work with trailing slashes."""
    reg_res = await async_client.post(
        "/v1/auth/register/",
        json={"email": "slashes@speedinfer.local", "password": "Password123!"},
    )
    assert reg_res.status_code == 201
    token = reg_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    login_res = await async_client.post(
        "/v1/auth/login/",
        json={"email": "slashes@speedinfer.local", "password": "Password123!"},
    )
    assert login_res.status_code == 200

    me_res = await async_client.get("/v1/auth/me/", headers=headers)
    assert me_res.status_code == 200

    keys_res = await async_client.get("/v1/keys/", headers=headers)
    assert keys_res.status_code == 200


@pytest.mark.asyncio
async def test_create_key_whitespace_name_defaults(async_client: httpx.AsyncClient) -> None:
    """Verify API keys created with empty or whitespace name default to 'default'."""
    reg_res = await async_client.post(
        "/v1/auth/register",
        json={"email": "whitespace_test@speedinfer.local", "password": "Password123!"},
    )
    token = reg_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    res = await async_client.post("/v1/keys", json={"name": "   "}, headers=headers)
    assert res.status_code == 201
    assert res.json()["name"] == "default"


@pytest.mark.asyncio
async def test_balance_aggregation_excludes_expired_and_revoked(
    async_client: httpx.AsyncClient,
    db_session: Session,
) -> None:
    """Verify user balance only sums active and unexpired API keys."""
    reg_res = await async_client.post(
        "/v1/auth/register",
        json={
            "email": "ledger_sum@speedinfer.local",
            "password": "Password123!",
            "initial_balance": 10.0,
            "create_api_key": True,
        },
    )
    token = reg_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    key1_id = reg_res.json()["api_key"]["id"]

    # Create a second key with 25.0
    key2_res = await async_client.post(
        "/v1/keys", json={"name": "key-2", "credit_balance": 25.0}, headers=headers
    )
    assert key2_res.status_code == 201
    # Seed an operator-funded balance directly; users cannot mint credit via API.
    key1 = db_session.get(ApiKey, key1_id)
    assert key1 is not None
    key1.paid_balance = 10.0
    key1.credit_balance = 10.0
    db_session.add(key1)

    key2 = db_session.get(ApiKey, key2_res.json()["id"])
    assert key2 is not None
    key2.paid_balance = 25.0
    key2.credit_balance = 25.0
    db_session.add(key2)
    db_session.commit()

    # Total balance should be 35.0
    me_res = await async_client.get("/v1/auth/me", headers=headers)
    assert me_res.json()["balance"] == 35.0

    # Revoke key 1
    await async_client.delete(f"/v1/keys/{key1_id}", headers=headers)

    # Balance should drop to 25.0
    me_after = await async_client.get("/v1/auth/me", headers=headers)
    assert me_after.json()["balance"] == 25.0
