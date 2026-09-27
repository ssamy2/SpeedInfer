"""Tests for prepaid-credit fulfillment from signed Whop events."""

import base64
import hashlib
import hmac
import json
import time
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from sqlmodel import Session, select

from speedinfer.config import Settings
from speedinfer.core.inference_billing import InferenceReservation, reserve, settle
from speedinfer.database.models import ApiKey, PaymentTransaction, User
from speedinfer.gateway.routes.billing import (
    CheckoutRequest,
    _credit_payment,
    _verify_webhook,
    create_checkout,
)
from speedinfer.gateway.routes.keys import revoke_key


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


def test_webhook_signature_whsec_secret() -> None:
    class RequestStub:
        def __init__(self, headers: dict[str, str]) -> None:
            self.headers = headers

    raw_key = b"super_secret_raw_key_bytes_1234"
    secret = "whsec_" + base64.b64encode(raw_key).decode("utf-8")
    body = json.dumps({"type": "payment.succeeded", "data": {}}).encode()
    webhook_id = "msg_whsec"
    timestamp = str(int(time.time()))
    signed = f"{webhook_id}.{timestamp}.".encode() + body
    digest = hmac.new(raw_key, signed, hashlib.sha256).digest()
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


def test_webhook_signature_multi_sig_rotation() -> None:
    class RequestStub:
        def __init__(self, headers: dict[str, str]) -> None:
            self.headers = headers

    secret = "ws_test_secret"
    body = json.dumps({"type": "payment.succeeded", "data": {}}).encode()
    webhook_id = "msg_multi_sig"
    timestamp = str(int(time.time()))
    signed = f"{webhook_id}.{timestamp}.".encode() + body
    digest = hmac.new(secret.encode(), signed, hashlib.sha256).digest()
    valid_sig = base64.b64encode(digest).decode()

    # Valid signature accompanied by an old/rotated signature
    request = RequestStub(
        {
            "webhook-id": webhook_id,
            "webhook-timestamp": timestamp,
            "webhook-signature": f"v1,old_invalid_sig v1,{valid_sig}",
        }
    )
    event, returned_id = _verify_webhook(body, request, secret)
    assert returned_id == webhook_id

    # All signatures invalid
    bad_request = RequestStub(
        {
            "webhook-id": webhook_id,
            "webhook-timestamp": timestamp,
            "webhook-signature": "v1,bad1 v1,bad2",
        }
    )

    with pytest.raises(HTTPException) as exc_info:
        _verify_webhook(body, bad_request, secret)
    assert exc_info.value.status_code == 401

    # No v1 signature scheme
    no_v1_request = RequestStub(
        {
            "webhook-id": webhook_id,
            "webhook-timestamp": timestamp,
            "webhook-signature": "v2,some_future_scheme",
        }
    )
    with pytest.raises(HTTPException) as exc_info:
        _verify_webhook(body, no_v1_request, secret)
    assert exc_info.value.status_code == 400


def test_webhook_subtotal_tax_validation(db_session: Session) -> None:
    user = User(email="tax_user@speedinfer.local", password_hash="hash")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    key = ApiKey(user_id=user.id, key_hash="b" * 64, prefix="sk-speedinfer-tax")
    db_session.add(key)
    db_session.commit()
    db_session.refresh(key)

    # Subtotal matches $25 package, but total is $28.75 due to VAT/tax
    event = {
        "type": "payment.succeeded",
        "account_id": "biz_speedinfer",
        "data": {
            "id": "pay_tax_vat_123",
            "currency": "usd",
            "subtotal": 25.0,
            "usd_total": 28.75,
            "metadata": {
                "speedinfer_user_id": str(user.id),
                "speedinfer_api_key_id": str(key.id),
                "speedinfer_credits": "25.00",
            },
        },
    }

    _credit_payment(db_session, event, "msg_tax_123", _settings())

    refreshed_key = db_session.get(ApiKey, key.id)
    assert refreshed_key is not None
    assert refreshed_key.credit_balance == 25.0
    tx = db_session.exec(
        select(PaymentTransaction).where(
            PaymentTransaction.provider_payment_id == "pay_tax_vat_123"
        )
    ).first()
    assert tx is not None
    assert tx.amount_usd == 28.75
    assert tx.credits_added == 25.0


def test_webhook_credits_to_primary_key_when_target_inactive(db_session: Session) -> None:
    user = User(email="dead_key_user@speedinfer.local", password_hash="hash")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    # Primary active key
    key1 = ApiKey(
        user_id=user.id, key_hash="c" * 64, prefix="sk-speedinfer-active", is_active=True
    )
    db_session.add(key1)
    db_session.commit()
    db_session.refresh(key1)

    # Revoked secondary key
    key2 = ApiKey(
        user_id=user.id, key_hash="d" * 64, prefix="sk-speedinfer-revoked", is_active=False
    )
    db_session.add(key2)
    db_session.commit()
    db_session.refresh(key2)

    event = {
        "type": "payment.succeeded",
        "account_id": "biz_speedinfer",
        "data": {
            "id": "pay_inactive_target",
            "currency": "usd",
            "usd_total": 10.0,
            "metadata": {
                "speedinfer_user_id": str(user.id),
                "speedinfer_api_key_id": str(key2.id),
                "speedinfer_credits": "10.00",
            },
        },
    }

    _credit_payment(db_session, event, "msg_inactive_target", _settings())

    refreshed_key1 = db_session.get(ApiKey, key1.id)
    refreshed_key2 = db_session.get(ApiKey, key2.id)
    assert refreshed_key1.credit_balance == 10.0
    assert refreshed_key2.credit_balance == 0.0

    tx = db_session.exec(
        select(PaymentTransaction).where(
            PaymentTransaction.provider_payment_id == "pay_inactive_target"
        )
    ).first()
    assert tx is not None
    assert tx.api_key_id == key1.id


@pytest.mark.asyncio
async def test_checkout_payload_with_and_without_product_id(db_session: Session) -> None:
    user = User(id=101, email="checkout_user@speedinfer.local", password_hash="hash")
    db_session.add(user)
    db_session.commit()
    key = ApiKey(
        id=101, user_id=user.id, key_hash="e" * 64, prefix="sk-speedinfer-chk", is_active=True
    )
    db_session.add(key)
    db_session.commit()

    captured_requests = []

    async def mock_post(self, url, json=None, headers=None):
        captured_requests.append(json)

        class MockResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"purchase_url": "https://whop.com/checkout/test123"}

        return MockResp()

    settings_with_prod = Settings(
        api_key_pepper="p" * 32,
        whop_api_key="key_123",
        whop_account_id="biz_speedinfer",
        whop_product_id="prod_abc123",
        whop_credit_packages="10,25,50",
    )

    with patch("httpx.AsyncClient.post", new=mock_post):
        await create_checkout(
            payload=CheckoutRequest(amount_usd=10.0, api_key_id=key.id),
            current_user=user,
            session=db_session,
            settings=settings_with_prod,
        )

    assert len(captured_requests) == 1
    assert captured_requests[0]["plan"].get("product_id") == "prod_abc123"

    settings_without_prod = Settings(
        api_key_pepper="p" * 32,
        whop_api_key="key_123",
        whop_account_id="biz_speedinfer",
        whop_product_id="",
        whop_credit_packages="10,25,50",
    )

    with patch("httpx.AsyncClient.post", new=mock_post):
        await create_checkout(
            payload=CheckoutRequest(amount_usd=10.0, api_key_id=key.id),
            current_user=user,
            session=db_session,
            settings=settings_without_prod,
        )

    assert len(captured_requests) == 2
    assert "product_id" not in captured_requests[1]["plan"]


@pytest.mark.asyncio
async def test_permanent_key_deletion_safeguard_with_payment_records(db_session: Session) -> None:
    user = User(id=99, email="audit_trail@speedinfer.local", password_hash="hash")
    db_session.add(user)
    db_session.commit()

    key = ApiKey(
        id=99, user_id=user.id, key_hash="f" * 64, prefix="sk-audit-key", is_active=True
    )
    db_session.add(key)
    db_session.commit()

    tx = PaymentTransaction(
        provider_payment_id="pay_audit_test",
        webhook_id="wh_audit_test",
        user_id=user.id,
        api_key_id=key.id,
        amount_usd=10.0,
        credits_added=10.0,
        currency="usd",
    )
    db_session.add(tx)
    db_session.commit()

    # Attempt permanent deletion
    with pytest.raises(HTTPException) as exc_info:
        await revoke_key(key_id=key.id, current_user=user, session=db_session, permanent=True)

    assert exc_info.value.status_code == 409
    assert "This key has financial payment records. Revoke it instead." in exc_info.value.detail

    # Verify key and payment records are NOT deleted
    assert db_session.get(ApiKey, key.id) is not None
    assert db_session.get(PaymentTransaction, tx.id) is not None


def test_floating_point_rounding_in_reserve_and_settle(db_session: Session) -> None:

    user = User(email="rounding@speedinfer.local", password_hash="hash")
    db_session.add(user)
    db_session.commit()

    key = ApiKey(
        user_id=user.id,
        key_hash="0" * 64,
        prefix="sk-rounding",
        trial_balance=1.000000,
        paid_balance=1.000000,
        credit_balance=2.000000,
        is_active=True,
    )
    db_session.add(key)
    db_session.commit()

    # Reserve with float value that has excessive precision
    hold_id = reserve(db_session, key.id, 0.000010000000000012)
    hold = db_session.get(InferenceReservation, hold_id)
    assert hold is not None
    assert hold.amount == 0.000010
    assert hold.trial == 0.000010

    # Settle with fractional cost
    settle(db_session, hold_id, cost=0.000003000000000004, success=True)
    db_session.refresh(key)
    assert key.trial_balance == 0.999997
    assert key.credit_balance == 1.999997


@pytest.mark.asyncio
async def test_webhook_route_alias(async_client) -> None:
    from speedinfer.config import get_settings
    from speedinfer.gateway.app import app

    settings = Settings(
        api_key_pepper="p" * 32,
        whop_account_id="biz_speedinfer",
        whop_webhook_secret="whsec_dGVzdHNlY3JldA==",
        whop_credit_packages="10,25,50",
    )
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        # Calling /v1/billing/webhook without headers returns 400 (not 404 or 405)
        res_alias = await async_client.post("/v1/billing/webhook", content=b"{}")
        assert res_alias.status_code == 400
        assert "Missing webhook signature headers" in res_alias.text

        # Calling /v1/billing/webhooks/whop without headers also returns 400
        res_orig = await async_client.post("/v1/billing/webhooks/whop", content=b"{}")
        assert res_orig.status_code == 400
        assert "Missing webhook signature headers" in res_orig.text
    finally:
        app.dependency_overrides.pop(get_settings, None)


def test_webhook_signature_comma_separated_and_mixed_formats() -> None:
    class RequestStub:
        def __init__(self, headers: dict[str, str]) -> None:
            self.headers = headers

    secret = "ws_test_secret"
    body = json.dumps({"type": "payment.succeeded", "data": {}}).encode("utf-8")
    webhook_id = "msg_comma_test"
    timestamp = str(int(time.time()))
    signed = f"{webhook_id}.{timestamp}.".encode() + body
    digest = hmac.new(secret.encode("utf-8"), signed, hashlib.sha256).digest()
    valid_sig = base64.b64encode(digest).decode("utf-8")

    # Comma-separated with space
    req_comma_space = RequestStub(
        {
            "webhook-id": webhook_id,
            "webhook-timestamp": timestamp,
            "webhook-signature": f"v1,old_sig, v1,{valid_sig}",
        }
    )
    event, _ = _verify_webhook(body, req_comma_space, secret)
    assert event["type"] == "payment.succeeded"

    # Comma-separated without space
    req_comma_nospace = RequestStub(
        {
            "webhook-id": webhook_id,
            "webhook-timestamp": timestamp,
            "webhook-signature": f"v1,old_sig,v1,{valid_sig}",
        }
    )
    event, _ = _verify_webhook(body, req_comma_nospace, secret)
    assert event["type"] == "payment.succeeded"


def test_webhook_payload_with_utf8_non_ascii_metadata(db_session: Session) -> None:
    class RequestStub:
        def __init__(self, headers: dict[str, str]) -> None:
            self.headers = headers

    user = User(email="arabic_user@speedinfer.local", password_hash="hash")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    key = ApiKey(
        user_id=user.id,
        key_hash="1" * 64,
        prefix="sk-speedinfer-arabic",
        is_active=True,
    )
    db_session.add(key)
    db_session.commit()
    db_session.refresh(key)

    secret = "whsec_" + base64.b64encode(b"arabic_test_secret_key_123456").decode("utf-8")
    payload_dict = {
        "type": "payment.succeeded",
        "account_id": "biz_speedinfer",
        "data": {
            "id": "pay_utf8_arabic_123",
            "currency": "usd",
            "subtotal": 10.0,
            "usd_total": 10.0,
            "metadata": {
                "speedinfer_user_id": str(user.id),
                "speedinfer_api_key_id": str(key.id),
                "speedinfer_credits": "10.00",
                "customer_name": "سامي محمود 🚀",
                "note": "شحن رصيد تجريبي",
            },
        },
    }
    body = json.dumps(payload_dict, ensure_ascii=False).encode("utf-8")
    webhook_id = "msg_arabic_utf8"
    timestamp = str(int(time.time()))

    key_bytes = b"arabic_test_secret_key_123456"
    signed = f"{webhook_id}.{timestamp}.".encode() + body
    sig = base64.b64encode(hmac.new(key_bytes, signed, hashlib.sha256).digest()).decode("utf-8")

    request = RequestStub(
        {
            "webhook-id": webhook_id,
            "webhook-timestamp": timestamp,
            "webhook-signature": f"v1,{sig}",
        }
    )

    event, returned_id = _verify_webhook(body, request, secret)
    assert returned_id == webhook_id
    assert event["data"]["metadata"]["customer_name"] == "سامي محمود 🚀"

    _credit_payment(db_session, event, returned_id, _settings())
    db_session.refresh(key)
    assert key.credit_balance == 10.0


def test_webhook_svix_header_fallbacks() -> None:
    class RequestStub:
        def __init__(self, headers: dict[str, str]) -> None:
            self.headers = headers

    secret = "whsec_" + base64.b64encode(b"fallback_secret_bytes_12345").decode("utf-8")
    body = b'{"type": "payment.succeeded", "data": {}}'
    webhook_id = "msg_svix_hdr"
    timestamp = f"{int(time.time())}.5"  # Decimal timestamp
    key_bytes = b"fallback_secret_bytes_12345"
    signed = f"{webhook_id}.{timestamp}.".encode() + body
    sig = base64.b64encode(hmac.new(key_bytes, signed, hashlib.sha256).digest()).decode("utf-8")

    request = RequestStub(
        {
            "svix-id": webhook_id,
            "svix-timestamp": timestamp,
            "svix-signature": f"v1,{sig}",
        }
    )

    event, returned_id = _verify_webhook(body, request, secret)
    assert returned_id == webhook_id
    assert event["type"] == "payment.succeeded"


def test_webhook_credits_resurrects_revoked_key_when_no_active_keys(db_session: Session) -> None:
    user = User(email="all_revoked@speedinfer.local", password_hash="hash")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    # Only one key, and it is inactive
    revoked_key = ApiKey(
        user_id=user.id,
        key_hash="2" * 64,
        prefix="sk-speedinfer-resurrect",
        is_active=False,
    )
    db_session.add(revoked_key)
    db_session.commit()
    db_session.refresh(revoked_key)

    event = {
        "type": "payment.succeeded",
        "account_id": "biz_speedinfer",
        "data": {
            "id": "pay_resurrect_key",
            "currency": "usd",
            "usd_total": 25.0,
            "metadata": {
                "speedinfer_user_id": str(user.id),
                "speedinfer_api_key_id": str(revoked_key.id),
                "speedinfer_credits": "25.00",
            },
        },
    }

    _credit_payment(db_session, event, "msg_resurrect", _settings())

    db_session.refresh(revoked_key)
    assert revoked_key.is_active is True
    assert revoked_key.credit_balance == 25.0


def test_webhook_credits_auto_provisions_key_when_zero_keys(db_session: Session) -> None:
    user = User(email="zero_keys@speedinfer.local", password_hash="hash")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    event = {
        "type": "payment.succeeded",
        "account_id": "biz_speedinfer",
        "data": {
            "id": "pay_auto_provision",
            "currency": "usd",
            "usd_total": 50.0,
            "metadata": {
                "speedinfer_user_id": str(user.id),
                "speedinfer_api_key_id": "99999",  # Nonexistent
                "speedinfer_credits": "50.00",
            },
        },
    }

    _credit_payment(db_session, event, "msg_auto_provision", _settings())

    provisioned_key = db_session.exec(
        select(ApiKey).where(ApiKey.user_id == user.id, ApiKey.is_active == True)  # noqa: E712
    ).first()
    assert provisioned_key is not None
    assert provisioned_key.credit_balance == 50.0
    assert provisioned_key.is_active is True


def test_settle_failed_request_zeros_out_cost(db_session: Session) -> None:
    user = User(email="failed_settle@speedinfer.local", password_hash="hash")
    db_session.add(user)
    db_session.commit()

    key = ApiKey(
        user_id=user.id,
        key_hash="3" * 64,
        prefix="sk-speedinfer-settle-fail",
        trial_balance=1.0,
        paid_balance=1.0,
        credit_balance=2.0,
        is_active=True,
    )
    db_session.add(key)
    db_session.commit()

    hold_id = reserve(db_session, key.id, 0.5)
    db_session.refresh(key)
    assert key.credit_balance == 1.5

    # Pass cost > 0 but success=False (failed/aborted request)
    settle(db_session, hold_id, cost=0.25, success=False)
    db_session.refresh(key)
    # The hold must be 100% refunded to the customer
    assert key.credit_balance == 2.0
    assert key.trial_balance == 1.0
    assert key.paid_balance == 1.0


@pytest.mark.asyncio
async def test_permanent_key_deletion_safeguard_with_usage_ledger(db_session: Session) -> None:
    from speedinfer.database.models import UsageLedger

    user = User(email="ledger_safeguard@speedinfer.local", password_hash="hash")
    db_session.add(user)
    db_session.commit()

    key = ApiKey(
        user_id=user.id,
        key_hash="4" * 64,
        prefix="sk-speedinfer-ledger",
        is_active=True,
    )
    db_session.add(key)
    db_session.commit()

    ledger = UsageLedger(
        api_key_id=key.id,
        request_id="req_test_ledger",
        model="test-model",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        total_cost=0.001,
        latency_ms=50.0,
        status_code=200,
    )
    db_session.add(ledger)
    db_session.commit()

    with pytest.raises(HTTPException) as exc_info:
        await revoke_key(key_id=key.id, current_user=user, session=db_session, permanent=True)

    assert exc_info.value.status_code == 409
    assert "This key has billing records. Revoke it instead." in exc_info.value.detail



