"""
Email utility module.
Sends transactional emails (password reset, username reminder) via SMTP.

Required environment variables:
    SMTP_HOST     - SMTP server hostname (default: localhost)
    SMTP_PORT     - SMTP server port (default: 587)
    SMTP_USER     - SMTP login username
    SMTP_PASSWORD - SMTP login password
    SMTP_FROM     - Sender address (default: noreply@kona-ml.app)
    SMTP_USE_TLS  - Use STARTTLS when "true" (default: true)
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def _get_smtp_config() -> dict:
    return {
        "host": os.environ.get("SMTP_HOST", "localhost"),
        "port": int(os.environ.get("SMTP_PORT", "587")),
        "user": os.environ.get("SMTP_USER", ""),
        "password": os.environ.get("SMTP_PASSWORD", ""),
        "from_addr": os.environ.get("SMTP_FROM", "noreply@kona-ml.app"),
        "use_tls": os.environ.get("SMTP_USE_TLS", "true").lower() == "true",
    }


def _send_email(to_addr: str, subject: str, body_html: str, body_text: str) -> None:
    """
    Send an email. Raises an exception on failure so the caller can handle it.
    """
    cfg = _get_smtp_config()

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = cfg["from_addr"]
    msg["To"] = to_addr

    msg.attach(MIMEText(body_text, "plain"))
    msg.attach(MIMEText(body_html, "html"))

    with smtplib.SMTP(cfg["host"], cfg["port"], timeout=10) as server:
        if cfg["use_tls"]:
            server.starttls()
        if cfg["user"] and cfg["password"]:
            server.login(cfg["user"], cfg["password"])
        server.sendmail(cfg["from_addr"], [to_addr], msg.as_string())


def send_password_reset_email(to_email: str, franchise_id: str, reset_url: str) -> None:
    """
    Send a password-reset link to the given email address.

    Args:
        to_email:    Recipient email address.
        franchise_id: The franchise's login ID (shown for reference).
        reset_url:   Full URL of the reset page including the token.

    Raises:
        Exception on SMTP failure.
    """
    subject = "Kona ML – Password Reset Request"

    body_text = f"""Hi {franchise_id},

We received a request to reset the password for your Kona ML account.

Click the link below to choose a new password (expires in 1 hour):

  {reset_url}

If you did not request a password reset you can safely ignore this email.

— The Kona ML Team
"""

    body_html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"></head>
<body style="font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;background:#f4f4f4;margin:0;padding:20px;">
  <div style="max-width:520px;margin:0 auto;background:#fff;border-radius:8px;
              box-shadow:0 4px 12px rgba(0,0,0,.1);overflow:hidden;">
    <div style="background:linear-gradient(135deg,#667eea,#764ba2);padding:28px 30px;text-align:center;">
      <h1 style="margin:0;color:#fff;font-size:22px;">🐧 Kona ML</h1>
    </div>
    <div style="padding:30px;">
      <p style="color:#333;font-size:15px;">Hi <strong>{franchise_id}</strong>,</p>
      <p style="color:#555;font-size:14px;line-height:1.6;">
        We received a request to reset your Kona ML password.
        Click the button below to choose a new password. This link expires in&nbsp;<strong>1&nbsp;hour</strong>.
      </p>
      <div style="text-align:center;margin:28px 0;">
        <a href="{reset_url}"
           style="background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;
                  padding:12px 28px;border-radius:6px;text-decoration:none;
                  font-weight:600;font-size:15px;display:inline-block;">
          Reset Password
        </a>
      </div>
      <p style="color:#888;font-size:12px;line-height:1.5;">
        If you did not request a password reset you can safely ignore this email.
        Your password will remain unchanged.
      </p>
    </div>
  </div>
</body>
</html>"""

    _send_email(to_email, subject, body_html, body_text)


def send_username_reminder_email(to_email: str, franchise_id: str, franchise_name: str) -> None:
    """
    Send a username/franchise-ID reminder email.

    Args:
        to_email:      Recipient email address.
        franchise_id:  The franchise's login ID.
        franchise_name: The franchise's display name.

    Raises:
        Exception on SMTP failure.
    """
    subject = "Kona ML – Your Login ID"

    body_text = f"""Hi {franchise_name},

You requested a reminder of your Kona ML login ID.

Your Franchise ID is: {franchise_id}

You can use this ID (or the email address on your account) to log in at any time.

If you did not make this request you can safely ignore this email.

— The Kona ML Team
"""

    body_html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"></head>
<body style="font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;background:#f4f4f4;margin:0;padding:20px;">
  <div style="max-width:520px;margin:0 auto;background:#fff;border-radius:8px;
              box-shadow:0 4px 12px rgba(0,0,0,.1);overflow:hidden;">
    <div style="background:linear-gradient(135deg,#667eea,#764ba2);padding:28px 30px;text-align:center;">
      <h1 style="margin:0;color:#fff;font-size:22px;">🐧 Kona ML</h1>
    </div>
    <div style="padding:30px;">
      <p style="color:#333;font-size:15px;">Hi <strong>{franchise_name}</strong>,</p>
      <p style="color:#555;font-size:14px;line-height:1.6;">
        You requested a reminder of your login details.
      </p>
      <div style="background:#f8f8ff;border:1px solid #dde;border-radius:6px;
                  padding:16px 20px;margin:20px 0;text-align:center;">
        <p style="margin:0;color:#555;font-size:13px;">Your Franchise ID</p>
        <p style="margin:6px 0 0;color:#333;font-size:22px;font-weight:700;
                  letter-spacing:.5px;">{franchise_id}</p>
      </div>
      <p style="color:#555;font-size:14px;line-height:1.6;">
        You can log in using your Franchise ID <em>or</em> the email address on your account.
      </p>
      <p style="color:#888;font-size:12px;line-height:1.5;">
        If you did not make this request you can safely ignore this email.
      </p>
    </div>
  </div>
</body>
</html>"""

    _send_email(to_email, subject, body_html, body_text)
