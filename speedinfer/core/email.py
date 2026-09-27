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


def _render_email_html(heading: str, body_html: str) -> str:
    """Wrap content in a dark-themed, responsive HTML template."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {{
      margin: 0; padding: 0; background-color: #0b0f19;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      color: #e2e8f0;
    }}
    .wrapper {{
      max-width: 560px; margin: 32px auto; background-color: #111827;
      border: 1px solid #1f2937; border-radius: 12px; overflow: hidden;
    }}
    .header {{
      padding: 28px 32px 20px; border-bottom: 1px solid #1f2937; text-align: center;
      background: linear-gradient(180deg, #161f30 0%, #111827 100%);
    }}
    .logo {{ font-size: 24px; font-weight: 800; color: #ffffff; letter-spacing: -0.5px; }}
    .logo span {{ color: #06b6d4; }}
    .content {{ padding: 32px; font-size: 15px; line-height: 1.6; color: #cbd5e1; }}
    .heading {{
      font-size: 18px; font-weight: 700; color: #ffffff; margin-top: 0; margin-bottom: 16px;
    }}
    .code-box {{
      margin: 24px 0; padding: 18px; background-color: #1e293b;
      border: 1px solid #334155; border-radius: 8px; text-align: center;
    }}
    .code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 32px; font-weight: 700; color: #38bdf8; letter-spacing: 8px; margin: 0;
    }}
    .meta-table {{ width: 100%; margin: 20px 0; border-collapse: collapse; }}
    .meta-table td {{ padding: 10px 12px; border-bottom: 1px solid #1f2937; font-size: 14px; }}
    .meta-table td:first-child {{ color: #94a3b8; font-weight: 500; width: 45%; }}
    .meta-table td:last-child {{
      color: #f8fafc; font-weight: 600; font-family: ui-monospace, monospace;
    }}
    .footer {{
      padding: 20px 32px; background-color: #0d131f; border-top: 1px solid #1f2937;
      text-align: center; font-size: 12px; color: #64748b;
    }}
    .footer a {{ color: #06b6d4; text-decoration: none; }}
  </style>
</head>
<body>
  <div class="wrapper">
    <div class="header">
      <div class="logo">Speed<span>Infer</span></div>
    </div>
    <div class="content">
      <h2 class="heading">{heading}</h2>
      {body_html}
    </div>
    <div class="footer">
      &copy; 2026 SpeedInfer AI Technologies &bull; Ultra-low latency GPU inference<br>
      <a href="https://speedinfer.com">https://speedinfer.com</a>
    </div>
  </div>
</body>
</html>"""


def send_verification_email(to_email: str, code: str, settings: Settings | None = None) -> bool:
    """Send a 6-digit email address verification code."""
    subject = "[SpeedInfer] Verify your email address"
    body = (
        f"Welcome to SpeedInfer!\n\n"
        f"Your 6-digit verification code is: {code}\n\n"
        f"This code will expire in 15 minutes. Enter this code on the verification screen "
        f"to activate your account.\n\n"
        f"If you did not request this account, please disregard this email."
    )
    html_content = (
        f"<p>Welcome to <strong>SpeedInfer</strong>! Please verify your email address to complete "
        f"your account activation.</p>"
        f'<div class="code-box"><div class="code">{code}</div></div>'
        f'<p style="color:#94a3b8; font-size:13px;">This security code will expire in '
        f"<strong>15 minutes</strong>. If you did not create a SpeedInfer account, you can "
        f"safely ignore this email.</p>"
    )
    html_body = _render_email_html("Verify Your Email Address", html_content)
    return send_email(to_email, subject, body, settings=settings, html_body=html_body)


def send_password_reset_email(to_email: str, code: str, settings: Settings | None = None) -> bool:
    """Send a 6-digit password reset security code."""
    subject = "[SpeedInfer] Password Reset Security Code"
    body = (
        f"We received a request to reset your SpeedInfer account password.\n\n"
        f"Your password reset verification code is: {code}\n\n"
        f"This code is valid for 15 minutes. If you did not initiate this request, "
        f"you can safely ignore this email; your existing password will remain secure."
    )
    html_content = (
        f"<p>We received a request to reset your SpeedInfer account password.</p>"
        f'<div class="code-box"><div class="code">{code}</div></div>'
        f'<p style="color:#94a3b8; font-size:13px;">This code is valid for '
        f"<strong>15 minutes</strong>. If you did not initiate this request, you can safely "
        f"ignore this email; your existing password will remain secure.</p>"
    )
    html_body = _render_email_html("Password Reset Request", html_content)
    return send_email(to_email, subject, body, settings=settings, html_body=html_body)


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
    html_content = (
        f"<p>Thank you for your payment. Your inference credits have been credited to your "
        f"account and are available immediately.</p>"
        f'<table class="meta-table">'
        f"<tr><td>Transaction Reference</td><td>{transaction_id}</td></tr>"
        f"<tr><td>Amount Paid</td><td>${amount_usd:.2f} USD</td></tr>"
        f"<tr><td>Credits Added</td><td>${credits_added:.2f}</td></tr>"
        f"<tr><td>Date (UTC)</td><td>{date_str}</td></tr>"
        f"</table>"
        f'<p style="color:#94a3b8; font-size:13px;">You can view your real-time usage and API '
        f'keys anytime in the <a href="https://speedinfer.com" style="color:#06b6d4;">'
        f"SpeedInfer Dashboard</a>.</p>"
    )
    html_body = _render_email_html("Payment Confirmation & Receipt", html_content)
    return send_email(to_email, subject, body, settings=settings, html_body=html_body)
