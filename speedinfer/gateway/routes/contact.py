"""Persistent contact intake and administrator-only email notification workflow."""

import hashlib
import hmac
import re
import smtplib
import ssl
from datetime import UTC, datetime, timedelta
from email.message import EmailMessage
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import func, or_
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from speedinfer.config import Settings, get_settings
from speedinfer.core.security import get_current_user
from speedinfer.database.models import ContactRequest, User
from speedinfer.database.session import get_session

router = APIRouter(prefix="/v1/contact", tags=["Contact"])


class ContactPayload(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    request_id: UUID
    category: Literal["sales", "support"]
    name: str = Field(min_length=2, max_length=100)
    email: str = Field(min_length=5, max_length=254)
    company: str = Field(default="", max_length=120)
    subject: str = Field(min_length=3, max_length=160)
    message: str = Field(min_length=20, max_length=3000)
    consent: Literal[True]
    website: str = Field(default="", max_length=200)  # Honeypot, not a real field.

    @field_validator("email")
    @classmethod
    def valid_email(cls, value: str) -> str:
        if not re.fullmatch(
            r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,63}", value
        ):
            raise ValueError("Enter a valid email address.")
        return value.lower()

    @field_validator("name", "subject", "company")
    @classmethod
    def single_line(cls, value: str) -> str:
        if any(ord(char) < 32 for char in value):
            raise ValueError("Use a single line without control characters.")
        return value


def notify_team(item: ContactRequest, settings: Settings) -> bool:
    """Send only to configured team inboxes; visitor address is Reply-To, never To."""
    if not settings.smtp_host or not settings.smtp_from_email:
        return False
    recipient = settings.sales_email if item.category == "sales" else settings.support_email
    email = EmailMessage()
    email["From"] = settings.smtp_from_email
    email["To"] = recipient
    email["Reply-To"] = item.email
    email["Subject"] = f"[SpeedInfer {item.category}] {item.subject}"
    email["Message-ID"] = f"<{item.id}@speedinfer.com>"
    email.set_content(
        f"Reference: {item.id}\nName: {item.name}\nEmail: {item.email}\n"
        f"Company: {item.company or 'Not provided'}\n\n{item.message}\n\n"
        "Reply to this email to respond directly to the requester."
    )
    try:
        context = ssl.create_default_context()
        if settings.smtp_implicit_tls:
            connection = smtplib.SMTP_SSL(
                settings.smtp_host,
                settings.smtp_port,
                timeout=settings.smtp_timeout_seconds,
                context=context,
            )
        else:
            connection = smtplib.SMTP(
                settings.smtp_host,
                settings.smtp_port,
                timeout=settings.smtp_timeout_seconds,
            )
        with connection as smtp:
            if not settings.smtp_implicit_tls:
                smtp.starttls(context=context)
            if settings.smtp_username:
                smtp.login(
                    settings.smtp_username,
                    settings.smtp_password.get_secret_value() if settings.smtp_password else "",
                )
            smtp.send_message(email)
        return True
    except (OSError, smtplib.SMTPException, ValueError):
        # Do not log SMTP credentials or the visitor's message.
        return False


def deliver(item: ContactRequest, session: Session, settings: Settings) -> None:
    if item.delivery_status == "sent":
        return
    item.delivery_status = "sent" if notify_team(item, settings) else "pending"
    session.add(item)
    session.commit()


@router.get("/options")
def contact_options(settings: Annotated[Settings, Depends(get_settings)]) -> dict:
    return {"sales_email": settings.sales_email, "support_email": settings.support_email}


@router.post("/requests", status_code=201)
def submit_request(
    payload: ContactPayload,
    request: Request,
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict:
    if payload.website:
        raise HTTPException(422, "Unable to accept this request. Please email our team.")
    # A UUID retained by the form makes network retries safe, without leaking contents.
    reference = str(payload.request_id)
    if session.get(ContactRequest, reference):
        return {"reference": reference, "status": "received"}
    client = request.client.host if request.client else "unknown"
    fingerprint = hmac.new(
        settings.api_key_pepper.get_secret_value().encode(),
        client.encode(),
        hashlib.sha256,
    ).hexdigest()
    since = datetime.now(UTC) - timedelta(hours=1)
    recent = session.exec(
        select(func.count())
        .select_from(ContactRequest)
        .where(
            ContactRequest.created_at >= since,
            or_(
                ContactRequest.client_fingerprint == fingerprint,
                ContactRequest.email == payload.email,
            ),
        )
    ).one()
    if recent >= 5:
        raise HTTPException(429, "Too many requests. Please wait an hour or email our team.")
    item = ContactRequest(
        id=reference,
        category=payload.category,
        name=payload.name,
        email=payload.email,
        company=payload.company,
        subject=payload.subject,
        message=payload.message,
        client_fingerprint=fingerprint,
    )
    session.add(item)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        if session.get(ContactRequest, reference):
            return {"reference": reference, "status": "received"}
        raise
    deliver(item, session, settings)
    return {"reference": reference, "status": "received"}


def require_admin(user: Annotated[User, Depends(get_current_user)]) -> User:
    if not user.is_admin:
        raise HTTPException(403, "Administrator access required.")
    return user


@router.get("/requests")
def list_requests(
    admin: Annotated[User, Depends(require_admin)],
    session: Annotated[Session, Depends(get_session)],
    offset: int = 0,
) -> dict:
    if offset < 0:
        raise HTTPException(422, "Offset must not be negative.")
    items = session.exec(
        select(ContactRequest).order_by(ContactRequest.created_at.desc()).offset(offset).limit(50)
    ).all()
    return {"data": [item.model_dump(exclude={"client_fingerprint"}) for item in items]}


@router.post("/requests/{reference}/notify")
def retry_notification(
    reference: UUID,
    admin: Annotated[User, Depends(require_admin)],
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict:
    item = session.get(ContactRequest, str(reference))
    if not item:
        raise HTTPException(404, "Request not found.")
    deliver(item, session, settings)
    return {"delivery_status": item.delivery_status}
