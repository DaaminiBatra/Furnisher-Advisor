import logging
import smtplib
from email.message import EmailMessage

from .config import get_settings

log = logging.getLogger(__name__)


def send_reset_email(to: str, name: str, link: str) -> None:
    """Send the password-reset email. With no SMTP configured (local dev), the link
    is logged instead. Failures never propagate: the endpoint must respond the same
    way whether or not the address exists."""
    s = get_settings()
    if not s.smtp_host:
        log.info("SMTP not configured; reset link for %s: %s", to, link)
        return

    msg = EmailMessage()
    msg["Subject"] = "Reset your Furnisher Advisor password"
    msg["From"] = s.mail_from or s.smtp_user
    msg["To"] = to
    msg.set_content(
        f"Hi {name},\n\nYou asked to reset your password. This link works for "
        f"{s.reset_token_minutes} minutes:\n\n{link}\n\n"
        "If you didn't request this, you can ignore this email."
    )
    try:
        with smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=15) as smtp:
            smtp.starttls()
            if s.smtp_user:
                smtp.login(s.smtp_user, s.smtp_password)
            smtp.send_message(msg)
    except Exception:  # noqa: BLE001
        log.exception("Failed to send reset email to %s", to)
