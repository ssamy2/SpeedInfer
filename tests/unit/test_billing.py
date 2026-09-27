"""Tests for prepaid-credit fulfillment from signed Whop events."""

import base64
import hashlib
import hmac
import json
import time

from sqlmodel import Session, select

from speedinfer.config import Settings
from speedinfer.database.models import ApiKey, PaymentTransaction, User
from speedinfer.gateway.routes.billing import _credit_payment, _verify_webhook


def _settings() -> Settings:
    return Settings(
        api_key_pepper="p" * 32,
        whop_account_id="biz_speedinfer",
        whop_credit_packages="10,25,50",
    )


def test_successful_payment_is_credited_once(db_session: Session) -> None:
    user = User(email="billing@speedinfer.local", password_hash="hash")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    key = ApiKey(user_id=user.id, key_hash="a" * 64, prefix="sk-speedinfer-billing")
    db_session.add(key)
    db_session.commit()
    db_session.refresh(key)
    event = {
        "type": "payment.succeeded",
        "account_id": "biz_speedinfer",
        "data": {
            "id": "pay_credit_once",
            "currency": "usd",
            "usd_total": 25,
            "metadata": {
                "speedinfer_user_id": str(user.id),
                "speedinfer_api_key_id": str(key.id),
                "speedinfer_credits": "25.00",
            },
        },
    }

    _credit_payment(db_session, event, "msg_credit_once", _settings())
    _credit_payment(db_session, event, "msg_credit_once", _settings())

    refreshed_key = db_session.get(ApiKey, key.id)
    records = db_session.exec(select(PaymentTransaction)).all()
    assert refreshed_key is not None
    assert refreshed_key.credit_balance == 25.0
    assert len(records) == 1


def test_webhook_signature_requires_raw_body_and_recent_timestamp() -> None:
    class RequestStub:
        def __init__(self, headers: dict[str, str]) -> None:
            self.headers = headers

    secret = "ws_test_secret"
    body = json.dumps({"type": "payment.succeeded", "data": {}}).encode()
    webhook_id = "msg_signed"
    timestamp = str(int(time.time()))
    signed = f"{webhook_id}.{timestamp}.".encode() + body
    digest = hmac.new(secret.encode(), signed, hashlib.sha256).digest()
    signature = base64.b64encode(digest).decode()
    request = RequestStub(
        {
            "webhook-id": webhook_id,
            "webhook-timestamp": timestamp,
            "webhook-signature": f"v1,{signature}",
        }
    )

    event, returned_id = _verify_webhook(body, request, secret)
    assert event["type"] == "payment.succeeded"
    assert returned_id == webhook_id
