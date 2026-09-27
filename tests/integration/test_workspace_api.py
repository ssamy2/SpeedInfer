"""Verify ownership, real storage and zero-charge simulated model lifecycles."""

import uuid

import httpx
import pytest
from sqlmodel import Session

from speedinfer.database.session import get_session
from speedinfer.database.workspace import PolicyAcceptance
from speedinfer.gateway.app import app


@pytest.fixture
async def client(db_session: Session):
    app.dependency_overrides[get_session] = lambda: db_session
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    app.dependency_overrides.pop(get_session, None)


async def register(client, email):
    response = await client.post(
        "/v1/auth/register",
        json={"email": email, "password": "StrongTestPassword1!", "name": "Builder"},
    )
    assert response.status_code == 201, response.text
    assert response.json()["api_key"] is None
    return {"Authorization": "Bearer " + response.json()["access_token"]}


async def resource(client, headers, kind, **kwargs):
    response = await client.post(
        "/v1/workspace/resources",
        headers=headers,
        json={"kind": kind, "name": kind + " test", "request_id": uuid.uuid4().hex, **kwargs},
    )
    assert response.status_code == 201, response.text
    return response.json()


async def test_private_bucket_training_deployment_lifecycle(client):
    owner = await register(client, "owner@test.local")
    other = await register(client, "other@test.local")
    project = await resource(client, owner, "project")
    bucket = await resource(client, owner, "bucket")
    content = b'{"text":"training example"}\n'
    upload = await client.post(
        f"/v1/workspace/buckets/{bucket['id']}/objects",
        params={"name": "training.jsonl", "purpose": "dataset"},
        headers=owner,
        content=content,
    )
    assert upload.status_code == 201, upload.text
    object_id = upload.json()["id"]
    listing = await client.get("/v1/workspace/resources", headers=owner)
    assert listing.json()["objects"][0]["size"] == len(content)
    assert "content" not in listing.json()["objects"][0]
    assert (
        await client.get(f"/v1/workspace/objects/{object_id}", headers=owner)
    ).content == content
    assert (
        await client.get(f"/v1/workspace/objects/{object_id}", headers=other)
    ).status_code == 404
    assert (await client.get("/v1/workspace/resources", headers=other)).json()["objects"] == []
    job = await resource(client, owner, "training", project_id=project["id"], dataset_id=object_id)
    action_url = f"/v1/workspace/resources/{job['id']}/actions"
    assert (
        await client.post(action_url, headers=other, json={"action": "start"})
    ).status_code == 404
    assert (
        await client.post(action_url, headers=owner, json={"action": "complete"})
    ).status_code == 409
    assert (await client.post(action_url, headers=owner, json={"action": "start"})).json()[
        "status"
    ] == "running"
    result = (await client.post(action_url, headers=owner, json={"action": "complete"})).json()
    assert result["status"] == "succeeded"
    assert result["data"]["is_simulated"] is True
    assert result["data"]["estimate"]["charge_usd"] == 0
    assert "No trained weights" in result["data"]["result"]
    deployment = await resource(
        client, owner, "deployment", project_id=project["id"], artifact_id=job["id"]
    )
    assert deployment["data"]["is_simulated"] is True
    assert (
        await client.delete(f"/v1/workspace/objects/{object_id}", headers=owner)
    ).status_code == 409
    assert (await client.get("/v1/auth/me", headers=owner)).json()["balance"] == 0


async def test_validation_idempotency_and_authentication(client):
    owner = await register(client, "validation@test.local")
    other = await register(client, "isolated@test.local")
    payload = {"kind": "bucket", "name": "Private bucket", "request_id": "request-12345678"}
    first = await client.post("/v1/workspace/resources", headers=owner, json=payload)
    second = await client.post("/v1/workspace/resources", headers=owner, json=payload)
    assert first.json()["id"] == second.json()["id"]
    bucket_id = first.json()["id"]
    assert (await client.get("/v1/workspace/resources")).status_code == 401
    assert (
        await client.post(
            f"/v1/workspace/buckets/{bucket_id}/objects",
            headers=other,
            params={"name": "a.txt"},
            content=b"private",
        )
    ).status_code == 404
    for name, content in [
        ("../file.jsonl", b"{}"),
        ("data.jsonl", b"not json"),
        ("data.jsonl", b"{}"),
    ]:
        response = await client.post(
            f"/v1/workspace/buckets/{bucket_id}/objects",
            headers=owner,
            params={"name": name, "purpose": "dataset"},
            content=content,
        )
        assert response.status_code == 422
    estimate = await client.post(
        "/v1/workspace/estimate",
        headers=owner,
        json={"sku": "demo-large", "hours": 8, "replicas": 2},
    )
    assert estimate.json()["total_usd"] == 32
    assert estimate.json()["charge_usd"] == 0
    assert (
        await client.post(
            "/v1/workspace/estimate", headers=owner, json={"sku": "demo-large", "hours": -1}
        )
    ).status_code == 400
    upload = await client.post(
        f"/v1/workspace/buckets/{bucket_id}/objects",
        headers=owner,
        params={"name": "readme.txt"},
        content=b"test",
    )
    object_id = upload.json()["id"]
    assert (
        await client.delete(f"/v1/workspace/objects/{object_id}", headers=other)
    ).status_code == 404
    assert (
        await client.delete(f"/v1/workspace/objects/{object_id}", headers=owner)
    ).status_code == 204
    assert (
        await client.get(f"/v1/workspace/objects/{object_id}", headers=owner)
    ).status_code == 404


async def test_explicit_key_creation_gets_trial_once_and_cannot_mint_credit(client):
    owner = await register(client, "keys@test.local")
    assert (await client.get("/v1/keys", headers=owner)).json()["data"] == []
    first = await client.post(
        "/v1/keys", headers=owner, json={"name": "first", "credit_balance": 99999}
    )
    assert first.status_code == 201
    assert first.json()["credit_balance"] == 0
    second = await client.post(
        "/v1/keys", headers=owner, json={"name": "second", "credit_balance": 99999}
    )
    assert second.json()["credit_balance"] == 0
    assert (await client.delete(f"/v1/keys/{first.json()['id']}", headers=owner)).status_code == 200
    third = await client.post("/v1/keys", headers=owner, json={"name": "third"})
    assert third.json()["credit_balance"] == 0
    login = await client.post(
        "/v1/auth/login", json={"email": "keys@test.local", "password": "StrongTestPassword1!"}
    )
    assert login.json()["api_key"] is None


async def test_policy_acknowledgement_and_file_limit(client, db_session, monkeypatch):
    response = await client.post(
        "/v1/auth/register",
        json={
            "email": "policy@test.local",
            "password": "StrongPassword123!",
            "accepted_policy_version": "2026-09-27",
        },
    )
    assert response.status_code == 201
    user_id = response.json()["user"]["id"]
    assert db_session.get(PolicyAcceptance, (user_id, "2026-09-27")) is not None
    headers = {"Authorization": "Bearer " + response.json()["access_token"]}
    bucket = await resource(client, headers, "bucket")
    monkeypatch.setattr("speedinfer.gateway.routes.workspace.MAX_OBJECT_BYTES", 4)
    result = await client.post(
        f"/v1/workspace/buckets/{bucket['id']}/objects",
        headers=headers,
        params={"name": "too-large.txt"},
        content=b"12345",
    )
    assert result.status_code == 413
    assert (await client.get("/v1/workspace/resources", headers=headers)).json()["objects"] == []
