"""Whop-hosted checkout and signed webhook fulfillment for prepaid API credit."""

import base64
import hashlib
import hmac
import json
import time
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from speedinfer.config import Settings, get_settings
from speedinfer.core.email import send_payment_invoice_email
from speedinfer.core.referrals import check_and_award_referrer_bonus
from speedinfer.core.security import get_current_user
from speedinfer.database.models import ApiKey, PaymentTransaction, User
from speedinfer.database.session import get_session
from speedinfer.gateway.redis import get_sync_redis

router = APIRouter(prefix="/v1/billing", tags=["Billing"])
WHOP_BASE_URL = "https://api.whop.com/api/v1"


class CheckoutRequest(BaseModel):
    """A fixed credit package and the receiving key selected by its owner."""

    amount_usd: float = Field(gt=0, le=10_000)
    api_key_id: int | None = Field(default=None, gt=0)


def _secret(value: Any) -> str:
    return value.get_secret_value() if hasattr(value, "get_secret_value") else str(value)


def _payment_error(message: str, code: str = "billing_unavailable") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"error": {"message": message, "type": "api_error", "param": None, "code": code}},
    )


def _select_key(session: Session, user_id: int, requested_key_id: int | None) -> ApiKey:
    statement = select(ApiKey).where(ApiKey.user_id == user_id, ApiKey.is_active == True)  # noqa: E712
    if requested_key_id is not None:
        statement = statement.where(ApiKey.id == requested_key_id)
    api_key = session.exec(statement.order_by(ApiKey.created_at.asc())).first()
    if api_key is None:
        raise HTTPException(
            status_code=400, detail="Create an active API key before adding credit."
        )
    return api_key


@router.get("/packages")
async def list_packages(
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, list[float]]:
    """Return only server-approved package amounts for the UI."""
    return {"packages": list(settings.credit_packages())}


@router.post("/checkout")
async def create_checkout(
    payload: CheckoutRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, str | float | int]:
    """Create a Whop-hosted, one-time checkout without exposing credentials."""
    if settings.whop_api_key is None or not settings.whop_account_id:
        raise _payment_error("Billing is not configured yet. Please contact Sales@speedinfer.com.")
    amount = round(payload.amount_usd, 2)
    if amount not in settings.credit_packages():
        raise HTTPException(status_code=400, detail="Select one of the published credit packages.")
    api_key = _select_key(session, int(current_user.id), payload.api_key_id)
    body = {
        "account_id": settings.whop_account_id,
        "mode": "payment",
        "metadata": {
            "speedinfer_user_id": str(current_user.id),
            "speedinfer_api_key_id": str(api_key.id),
            "speedinfer_credits": f"{amount:.2f}",
        },
        "redirect_url": f"{settings.public_base_url.rstrip('/')}/app?payment=processing#usage",
        "plan": {
            "account_id": settings.whop_account_id,
            "title": f"SpeedInfer ${amount:.0f} Credits",
            "description": f"${amount:.2f} prepaid inference credit for SpeedInfer.",
            "currency": "usd",
            "plan_type": "one_time",
            "release_method": "buy_now",
            "initial_price": amount,
            "unlimited_stock": True,
            "visibility": "hidden",
        },
    }
    headers = {
        "Authorization": f"Bearer {_secret(settings.whop_api_key)}",
        "Api-Version-Date": settings.whop_api_version_date,
        "Idempotency-Key": str(uuid.uuid4()),
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                f"{WHOP_BASE_URL}/checkout_configurations", json=body, headers=headers
            )
        response.raise_for_status()
        checkout = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise _payment_error("Unable to start checkout. Please try again shortly.") from exc
    purchase_url = checkout.get("purchase_url")
    if not isinstance(purchase_url, str) or not purchase_url.startswith("https://"):
        raise _payment_error("Payment provider did not return a valid checkout URL.")
    return {"checkout_url": purchase_url, "amount_usd": amount, "api_key_id": int(api_key.id)}


def _verify_webhook(body: bytes, request: Request, secret: str) -> tuple[dict[str, Any], str]:
    """Verify Standard Webhooks HMAC against exactly the raw received bytes."""
    webhook_id = request.headers.get("webhook-id", "")
    timestamp = request.headers.get("webhook-timestamp", "")
    signature = request.headers.get("webhook-signature", "")
    if not webhook_id or not timestamp or not signature:
        raise HTTPException(status_code=400, detail="Missing webhook signature headers.")
    try:
        if abs(time.time() - int(timestamp)) > 300:
            raise ValueError
        supplied = signature.split(",", 1)[1]
        signed = f"{webhook_id}.{timestamp}.".encode() + body
        digest = hmac.new(secret.encode(), signed, hashlib.sha256).digest()
        expected = base64.b64encode(digest).decode()
    except (IndexError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid webhook signature headers.") from None
    if not hmac.compare_digest(expected, supplied):
        raise HTTPException(status_code=401, detail="Invalid webhook signature.")
    try:
        event = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid webhook JSON.") from None
    if not isinstance(event, dict):
        raise HTTPException(status_code=400, detail="Invalid webhook payload.")
    return event, webhook_id


def _credit_payment(
    session: Session, event: dict[str, Any], webhook_id: str, settings: Settings
) -> None:
    payment = event.get("data")
    if not isinstance(payment, dict) or event.get("type") != "payment.succeeded":
        return
    if event.get("account_id", event.get("company_id")) != settings.whop_account_id:
        return
    metadata = payment.get("metadata")
    if not isinstance(metadata, dict):
        return
    try:
        user_id = int(metadata["speedinfer_user_id"])
        api_key_id = int(metadata["speedinfer_api_key_id"])
        credits = round(float(metadata["speedinfer_credits"]), 2)
        paid = round(float(payment.get("usd_total", payment.get("total"))), 2)
        payment_id = str(payment["id"])
    except (KeyError, TypeError, ValueError):
        # The account can use Whop for other products; only fulfill our own checkouts.
        return
    if (
        credits not in settings.credit_packages()
        or paid != credits
        or payment.get("currency") != "usd"
    ):
        raise HTTPException(
            status_code=400, detail="Payment amount does not match a SpeedInfer package."
        )
    existing = session.exec(
        select(PaymentTransaction).where(PaymentTransaction.provider_payment_id == payment_id)
    ).first()
    if existing:
        return
    api_key = session.get(ApiKey, api_key_id)
    if api_key is None or api_key.user_id != user_id:
        raise HTTPException(status_code=400, detail="Payment target key is invalid.")
    api_key.credit_balance = round(float(api_key.credit_balance) + credits, 6)
    session.add(api_key)
    session.add(
        PaymentTransaction(
            provider_payment_id=payment_id,
            webhook_id=webhook_id,
            user_id=user_id,
            api_key_id=api_key_id,
            amount_usd=paid,
            credits_added=credits,
            currency="usd",
        )
    )
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        return

    # Deliver payment receipt / invoice confirmation email
    user = session.get(User, user_id)
    if user is not None and user.email:
        send_payment_invoice_email(
            to_email=user.email,
            transaction_id=payment_id,
            amount_usd=paid,
            credits_added=credits,
            date_str=str(datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")),
            settings=settings,
        )

    # Option A: Qualify referrer if this is the referee's first payment
    if check_and_award_referrer_bonus(session, user_id, settings=settings):
        session.commit()

    try:
        redis_client = get_sync_redis()
        redis_key = f"speedinfer:balance:{api_key_id}"
        if redis_client.get(redis_key) is not None:
            redis_client.incrbyfloat(redis_key, credits)
    except Exception:
        pass


@router.post("/webhooks/whop", include_in_schema=False)
async def whop_webhook(
    request: Request,
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> Response:
    """Receive signed provider events and apply each successful payment once."""
    if settings.whop_webhook_secret is None or not settings.whop_account_id:
        raise HTTPException(status_code=503, detail="Billing webhook is not configured.")
    event, webhook_id = _verify_webhook(
        await request.body(), request, _secret(settings.whop_webhook_secret)
    )
    _credit_payment(session, event, webhook_id, settings)
    return Response(status_code=200)
