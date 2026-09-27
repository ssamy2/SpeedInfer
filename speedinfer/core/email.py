"""Email delivery service supporting verification codes, password resets, and invoices.

Includes production SMTP dispatch with TLS/STARTTLS support and graceful in-memory
and logger fallbacks for development and automated testing environments.
"""

import smtplib
import ssl
from email.message import EmailMessage
from typing import Any

from speedinfer.config import Settings, get_settings
from speedinfer.logging import get_logger

logger = get_logger("speedinfer.email")

# In-memory mailbox registry for testing verification flows without real SMTP servers
MOCK_OUTBOX: list[dict[str, Any]] = []


def clear_mock_outbox() -> None:
    """Clear in-memory sent messages registry (useful for unit tests)."""
    MOCK_OUTBOX.clear()


def get_latest_mock_email(email: str | None = None) -> dict[str, Any] | None:
    """Retrieve the most recent mock email sent, optionally filtered by recipient."""
    for message in reversed(MOCK_OUTBOX):
        if email is None or message["to"] == email:
            return message
    return None


def send_email(
    to_email: str,
    subject: str,
    body: str,
    settings: Settings | None = None,
    html_body: str | None = None,
) -> bool:
    """Send an email using configured SMTP provider or capture in outbox if unconfigured.

    Args:
        to_email: Recipient mailbox address.
        subject: Email subject line.
        body: Plaintext message body.
        settings: Application runtime settings.
        html_body: Optional HTML formatted body alternative.

    Returns:
        bool: True if dispatched via SMTP or recorded to dev/test outbox.
    """
    cfg = settings or get_settings()

    record: dict[str, Any] = {
        "to": to_email,
        "subject": subject,
        "body": body,
        "html_body": html_body,
    }
    MOCK_OUTBOX.append(record)

    # In dev or test environments, or when SMTP is unconfigured, log and return True
    if cfg.environment in ("test", "development") or not cfg.smtp_host or not cfg.smtp_from_email:
        logger.info(
            "Dev/Test email dispatched to %s with subject '%s'",
            to_email,
            subject,
        )
        return True

    email_msg = EmailMessage()
    email_msg["From"] = cfg.smtp_from_email
    email_msg["To"] = to_email
    email_msg["Subject"] = subject
    email_msg.set_content(body)

    if html_body:
        email_msg.add_alternative(html_body, subtype="html")

    try:
        context = ssl.create_default_context()
        if cfg.smtp_implicit_tls:
            connection = smtplib.SMTP_SSL(
                cfg.smtp_host,
                cfg.smtp_port,
                timeout=cfg.smtp_timeout_seconds,
                context=context,
            )
        else:
            connection = smtplib.SMTP(
                cfg.smtp_host,
                cfg.smtp_port,
                timeout=cfg.smtp_timeout_seconds,
            )

        with connection as smtp:
            if not cfg.smtp_implicit_tls:
                smtp.starttls(context=context)
            if cfg.smtp_username:
                pwd = (
                    cfg.smtp_password.get_secret_value()
                    if hasattr(cfg.smtp_password, "get_secret_value") and cfg.smtp_password
                    else str(cfg.smtp_password or "")
                )
                smtp.login(cfg.smtp_username, pwd)
            smtp.send_message(email_msg)
        return True
    except (OSError, smtplib.SMTPException, ValueError) as exc:
        logger.error("Failed to deliver email via SMTP to %s: %s", to_email, str(exc))
        return False


def send_verification_email(to_email: str, code: str, settings: Settings | None = None) -> bool:
    """Send a 6-digit email address verification code."""
    subject = "[SpeedInfer] Verify your email address"
    body = (
        f"Welcome to SpeedInfer!\n\n"
        f"Your 6-digit verification code is: {code}\n\n"
        f"This code will expire in 15 minutes. Enter this code on the verification screen "
        f"to activate your account and claim your complimentary inference trial credits.\n\n"
        f"If you did not request this account, please disregard this email."
    )
    return send_email(to_email, subject, body, settings=settings)


def send_password_reset_email(to_email: str, code: str, settings: Settings | None = None) -> bool:
    """Send a 6-digit password reset security code."""
    subject = "[SpeedInfer] Password Reset Security Code"
    body = (
        f"We received a request to reset your SpeedInfer account password.\n\n"
        f"Your password reset verification code is: {code}\n\n"
        f"This code is valid for 15 minutes. If you did not initiate this request, "
        f"you can safely ignore this email; your existing password will remain secure."
    )
    return send_email(to_email, subject, body, settings=settings)


def send_payment_invoice_email(
    to_email: str,
    transaction_id: str,
    amount_usd: float,
    credits_added: float,
    date_str: str,
    settings: Settings | None = None,
) -> bool:
    """Send receipt and invoice confirmation upon successful payment credit fulfillment."""
    subject = f"[SpeedInfer] Payment Receipt - ${amount_usd:.2f} Credits Added"
    body = (
        f"Thank you for your payment to SpeedInfer.\n\n"
        f"Transaction Details:\n"
        f"- Reference ID: {transaction_id}\n"
        f"- Amount Paid: ${amount_usd:.2f} USD\n"
        f"- Inference Credits Added: ${credits_added:.2f}\n"
        f"- Date: {date_str}\n\n"
        f"Your credits have been added to your primary API key and are immediately "
        f"available for low-latency inference.\n\n"
        f"SpeedInfer AI Technologies\n"
        f"https://speedinfer.com"
    )
    return send_email(to_email, subject, body, settings=settings)
