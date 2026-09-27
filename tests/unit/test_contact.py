"""Contact intake: persistence, privacy, retries and bounded email delivery."""

from unittest.mock import MagicMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr
from sqlmodel import select

from speedinfer.config import Settings, get_settings
from speedinfer.core.security import get_current_user
from speedinfer.database.models import ContactRequest, User
from speedinfer.database.session import get_session
from speedinfer.gateway.routes import contact


@pytest.fixture
async def contact_client(db_session):
    app = FastAPI()
    app.include_router(contact.router)
    settings = Settings(
        _env_file=None, api_key_pepper=SecretStr("isolated-test-pepper-long-enough")
    )
    app.dependency_overrides[get_session] = lambda: db_session
    app.dependency_overrides[get_settings] = lambda: settings
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client, app, settings


def payload(**changes):
    return (
        dict(
            request_id=str(uuid4()),
            category="sales",
            name="Test Visitor",
            email="visitor@example.test",
            company="Example",
            subject="Private inference",
            message="We would like to discuss an inference deployment.",
            consent=True,
        )
        | changes
    )


@pytest.mark.asyncio
async def test_persistence_and_safe_retry(contact_client, db_session):
    client, _, _ = contact_client
    data = payload()
    first = await client.post("/v1/contact/requests", json=data)
    assert first.status_code == 201
    assert first.json() == {"reference": data["request_id"], "status": "received"}
    assert (await client.post("/v1/contact/requests", json=data)).status_code == 201
    rows = db_session.exec(select(ContactRequest)).all()
    assert len(rows) == 1
    assert rows[0].delivery_status == "pending"
    assert rows[0].message == data["message"]
    assert rows[0].client_fingerprint != "127.0.0.1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"email": "victim@example.test\r\nBcc:someone@example.test"},
        {"name": "A\r\nB"},
        {"message": "too short"},
        {"consent": False},
        {"website": "https://spam.example"},
        {"message": "x" * 3001},
        {"category": "other"},
    ],
)
async def test_reject_invalid_requests(contact_client, db_session, changes):
    client, _, _ = contact_client
    assert (await client.post("/v1/contact/requests", json=payload(**changes))).status_code == 422
    assert not db_session.exec(select(ContactRequest)).all()


@pytest.mark.asyncio
async def test_rate_limit_and_idempotency(contact_client):
    client, _, _ = contact_client
    first = payload()
    for data in [first] + [payload(email=f"person{i}@example.test") for i in range(4)]:
        assert (await client.post("/v1/contact/requests", json=data)).status_code == 201
    assert (await client.post("/v1/contact/requests", json=payload())).status_code == 429
    assert (await client.post("/v1/contact/requests", json=first)).status_code == 201


@pytest.mark.asyncio
async def test_inbox_protected_and_reply_data_available(contact_client):
    client, app, _ = contact_client
    data = payload(category="support")
    await client.post("/v1/contact/requests", json=data)
    assert (await client.get("/v1/contact/requests")).status_code == 401
    app.dependency_overrides[get_current_user] = lambda: User(
        email="user@example.test", is_admin=False
    )
    assert (await client.get("/v1/contact/requests")).status_code == 403
    assert (
        await client.post(f"/v1/contact/requests/{data['request_id']}/notify")
    ).status_code == 403
    app.dependency_overrides[get_current_user] = lambda: User(
        email="admin@example.test", is_admin=True
    )
    response = await client.get("/v1/contact/requests")
    assert response.status_code == 200
    item = response.json()["data"][0]
    assert item["email"] == data["email"] and item["category"] == "support"
    assert "client_fingerprint" not in item


@pytest.mark.asyncio
async def test_failed_notification_can_be_retried(contact_client, db_session, monkeypatch):
    client, app, _ = contact_client
    data = payload()
    notify = MagicMock(return_value=False)
    monkeypatch.setattr(contact, "notify_team", notify)
    await client.post("/v1/contact/requests", json=data)
    app.dependency_overrides[get_current_user] = lambda: User(
        email="admin@example.test", is_admin=True
    )
    notify.return_value = True
    response = await client.post(f"/v1/contact/requests/{data['request_id']}/notify")
    assert response.json()["delivery_status"] == "sent"
    assert db_session.get(ContactRequest, data["request_id"]).delivery_status == "sent"
    await client.post(f"/v1/contact/requests/{data['request_id']}/notify")
    assert notify.call_count == 2


@pytest.mark.parametrize(
    "category,recipient", [("sales", "Sales@speedinfer.com"), ("support", "Support@speedinfer.com")]
)
def test_email_is_to_team_with_visitor_reply_to(monkeypatch, category, recipient):
    settings = Settings(
        _env_file=None,
        api_key_pepper=SecretStr("test-pepper-long-enough"),
        smtp_host="smtp.example.test",
        smtp_from_email="website@example.test",
    )
    smtp = MagicMock()
    monkeypatch.setattr(contact.smtplib, "SMTP", MagicMock(return_value=smtp))
    item = ContactRequest(
        id=str(uuid4()),
        category=category,
        name="Visitor",
        email="visitor@example.test",
        subject="A request",
        message="Please contact us about inference.",
        client_fingerprint="x",
    )
    assert contact.notify_team(item, settings)
    connection = smtp.__enter__.return_value
    connection.starttls.assert_called_once()
    email = connection.send_message.call_args.args[0]
    assert email["To"] == recipient
    assert email["Reply-To"] == "visitor@example.test"
    assert email["From"] == "website@example.test"
