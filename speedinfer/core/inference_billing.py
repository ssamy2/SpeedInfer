"""Durable pre-inference holds and atomic settlement in the source-of-truth database."""

import math
import uuid

from fastapi import HTTPException
from sqlalchemy import update
from sqlmodel import Field, Session, SQLModel

from speedinfer.database.models import ApiKey, UsageLedger, utc_now


class InferenceReservation(SQLModel, table=True):
    __tablename__ = "inference_reservation"
    id: str = Field(primary_key=True)
    api_key_id: int = Field(foreign_key="apikey.id", index=True)
    amount: float
    trial: float
    paid: float
    state: str = Field(default="reserved", index=True)
    created_at: str = Field(default_factory=lambda: utc_now().isoformat())


def reserve(session: Session, key_id: int, amount: float) -> str:
    """Serialize against other holds; persist before sending anything to a worker."""
    if not math.isfinite(amount) or amount < 0:
        raise ValueError("Invalid reservation amount")
    amount = round(amount, 6)
    result = session.execute(
        update(ApiKey)
        .where(
            ApiKey.id == key_id,
            ApiKey.is_active == True,  # noqa: E712
            ApiKey.credit_balance >= amount,
        )
        .values(credit_balance=ApiKey.credit_balance)
    )
    if result.rowcount != 1:
        session.rollback()
        raise HTTPException(
            402,
            detail={
                "error": {
                    "message": "Insufficient available credit for the request reservation.",
                    "type": "insufficient_quota",
                    "code": "insufficient_balance",
                }
            },
        )
    session.expire_all()
    key = session.get(ApiKey, key_id)
    trial = round(min(key.trial_balance, amount), 6)
    paid = round(amount - trial, 6)
    key.trial_balance = round(max(0.0, key.trial_balance - trial), 6)
    key.paid_balance = round(max(0.0, key.paid_balance - paid), 6)
    key.credit_balance = round(key.trial_balance + key.paid_balance, 6)
    hold = InferenceReservation(
        id=uuid.uuid4().hex, api_key_id=key_id, amount=amount, trial=trial, paid=paid
    )
    hold_id = hold.id
    session.add(key)
    session.add(hold)
    session.commit()
    return hold_id


def settle(
    session: Session,
    hold_id: str,
    *,
    cost: float = 0,
    model: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    latency_ms: float = 0,
    ttft_ms: float | None = None,
    success: bool = False,
) -> None:
    """Charge measured usage and release the remainder once, in one transaction.

    Interrupted/unmetered responses are refunded. Process-crash holds remain durable
    for operator reconciliation rather than being silently charged or spent twice.
    """
    result = session.execute(
        update(InferenceReservation)
        .where(InferenceReservation.id == hold_id, InferenceReservation.state == "reserved")
        .values(state="settling")
    )
    if result.rowcount != 1:
        session.rollback()
        return
    session.expire_all()
    hold = session.get(InferenceReservation, hold_id)
    cost = round(cost, 6)
    hold_amount = round(hold.amount, 6)
    max_allowed = max(hold_amount * 1.5 + 0.0001, hold_amount + 1e-6)
    if not math.isfinite(cost) or cost < 0 or cost > max_allowed:
        session.rollback()
        raise HTTPException(502, detail="Backend usage exceeds the authorized reservation.")
    if not success:
        cost = 0.0
    cost = round(min(cost, hold_amount), 6)
    hold_trial = round(hold.trial, 6)
    hold_paid = round(hold.paid, 6)
    used_trial = round(min(hold_trial, cost), 6)
    refund_trial = round(hold_trial - used_trial, 6)
    refund_paid = round(hold_paid - (cost - used_trial), 6)
    refund_total = round(refund_trial + refund_paid, 6)
    session.execute(
        update(ApiKey)
        .where(ApiKey.id == hold.api_key_id)
        .values(
            trial_balance=ApiKey.trial_balance + refund_trial,
            paid_balance=ApiKey.paid_balance + refund_paid,
            credit_balance=ApiKey.credit_balance + refund_total,
        )
    )
    hold.state = "settled" if success else "released"
    session.add(hold)
    if success:
        session.add(
            UsageLedger(
                api_key_id=hold.api_key_id,
                request_id=hold.id,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                total_cost=cost,
                latency_ms=latency_ms,
                ttft_ms=ttft_ms,
                status_code=200,
            )
        )
    session.commit()
