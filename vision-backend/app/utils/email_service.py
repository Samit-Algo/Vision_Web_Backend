"""
Email notification service (async).
====================================

Sends event notifications by email with event details and evidence frame.
Independent of Kafka; triggered at the same time as Kafka from the event notifier.
Fully configurable via settings (can be disabled or adjusted without touching Kafka).
"""
import base64
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from typing import Any, Dict, List, Optional

from ..core.config import get_settings

logger = logging.getLogger(__name__)

try:
    import aiosmtplib
    AIOSMTP_AVAILABLE = True
except ImportError:
    AIOSMTP_AVAILABLE = False
    aiosmtplib = None  # type: ignore


def _build_html_body(payload: Dict[str, Any]) -> str:
    """Build a professional HTML email body from the event payload."""
    event_info = payload.get("event") or {}
    agent_info = payload.get("agent") or {}
    camera_info = payload.get("camera") or {}
    metadata_info = payload.get("metadata") or {}

    label = event_info.get("label", "Unknown event")
    event_type = event_info.get("event_type") or "—"
    timestamp = event_info.get("timestamp", "—")
    rule_index = event_info.get("rule_index", "—")
    agent_name = agent_info.get("agent_name", "—")
    agent_id = agent_info.get("agent_id", "—")
    camera_id = agent_info.get("camera_id") or "—"
    device_id = camera_info.get("device_id") or "—"
    session_id = metadata_info.get("session_id") or "—"
    event_id = metadata_info.get("event_id") or "—"
    video_timestamp = metadata_info.get("video_timestamp") or "—"

    # Optional report summary (e.g. from class_count or VLM)
    report = None
    detections = metadata_info.get("detections")
    if isinstance(detections, dict) and detections.get("rule_report"):
        report = detections.get("rule_report")

    report_html = ""
    if report:
        if isinstance(report, dict):
            report_items = "".join(
                f"<li><strong>{k}:</strong> {v}</li>"
                for k, v in report.items()
            )
        else:
            report_items = f"<li>{report}</li>"
        report_html = f"""
        <h3 style="color:#555;font-size:14px;margin-top:16px;">Report / Details</h3>
        <ul style="margin:4px 0;padding-left:20px;color:#444;">{report_items}</ul>
        """

    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Vision Event Alert</title>
</head>
<body style="font-family:Segoe UI,Helvetica,Arial,sans-serif;background:#f5f5f5;margin:0;padding:24px;">
  <div style="max-width:600px;margin:0 auto;background:#fff;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.08);overflow:hidden;">
    <div style="background:linear-gradient(135deg,#1a237e 0%,#283593 100%);color:#fff;padding:20px 24px;">
      <h1 style="margin:0;font-size:20px;font-weight:600;">Vision Event Alert</h1>
      <p style="margin:8px 0 0;opacity:0.9;font-size:14px;">{label}</p>
    </div>
    <div style="padding:24px;">
      <table style="width:100%;border-collapse:collapse;font-size:14px;color:#333;">
        <tr><td style="padding:8px 0;border-bottom:1px solid #eee;"><strong>Event</strong></td><td style="padding:8px 0;border-bottom:1px solid #eee;">{label}</td></tr>
        <tr><td style="padding:8px 0;border-bottom:1px solid #eee;"><strong>Type</strong></td><td style="padding:8px 0;border-bottom:1px solid #eee;">{event_type}</td></tr>
        <tr><td style="padding:8px 0;border-bottom:1px solid #eee;"><strong>Time (UTC)</strong></td><td style="padding:8px 0;border-bottom:1px solid #eee;">{timestamp}</td></tr>
        <tr><td style="padding:8px 0;border-bottom:1px solid #eee;"><strong>Video time</strong></td><td style="padding:8px 0;border-bottom:1px solid #eee;">{video_timestamp}</td></tr>
        <tr><td style="padding:8px 0;border-bottom:1px solid #eee;"><strong>Agent</strong></td><td style="padding:8px 0;border-bottom:1px solid #eee;">{agent_name}</td></tr>
        <tr><td style="padding:8px 0;border-bottom:1px solid #eee;"><strong>Agent ID</strong></td><td style="padding:8px 0;border-bottom:1px solid #eee;">{agent_id}</td></tr>
        <tr><td style="padding:8px 0;border-bottom:1px solid #eee;"><strong>Camera ID</strong></td><td style="padding:8px 0;border-bottom:1px solid #eee;">{camera_id}</td></tr>
        <tr><td style="padding:8px 0;border-bottom:1px solid #eee;"><strong>Device ID</strong></td><td style="padding:8px 0;border-bottom:1px solid #eee;">{device_id}</td></tr>
        <tr><td style="padding:8px 0;border-bottom:1px solid #eee;"><strong>Session ID</strong></td><td style="padding:8px 0;border-bottom:1px solid #eee;">{session_id}</td></tr>
        <tr><td style="padding:8px 0;border-bottom:1px solid #eee;"><strong>Event ID</strong></td><td style="padding:8px 0;border-bottom:1px solid #eee;">{event_id}</td></tr>
        <tr><td style="padding:8px 0;border-bottom:1px solid #eee;"><strong>Rule index</strong></td><td style="padding:8px 0;border-bottom:1px solid #eee;">{rule_index}</td></tr>
      </table>
      {report_html}
      <p style="margin-top:20px;font-size:13px;color:#666;">Evidence frame is attached to this email.</p>
    </div>
    <div style="background:#fafafa;padding:12px 24px;font-size:12px;color:#888;">
      This is an automated message from Vision Backend. Do not reply to this email.
    </div>
  </div>
</body>
</html>
"""


def _build_subject(payload: Dict[str, Any]) -> str:
    """Build a clear, professional subject line."""
    event_info = payload.get("event") or {}
    agent_info = payload.get("agent") or {}
    label = event_info.get("label", "Event")
    event_type = event_info.get("event_type") or ""
    agent_name = agent_info.get("agent_name", "Agent")
    timestamp = event_info.get("timestamp", "")[:19] if event_info.get("timestamp") else ""
    severity = "Alert"
    if event_type == "fall_detected":
        severity = "Critical"
    elif any(k in (label or "").lower() for k in ["weapon", "fire", "intrusion"]):
        severity = "Critical"
    elif any(k in (label or "").lower() for k in ["violation", "restricted", "alert"]):
        severity = "Warning"
    return f"[Vision {severity}] {label} — {agent_name} {timestamp}".strip()


async def send_event_notification_email_async(
    to_emails: List[str],
    event_payload: Dict[str, Any],
    frame_base64: Optional[str] = None,
) -> bool:
    """
    Send an event notification email asynchronously (HTML body + evidence attachment).

    This is the only public entry point for email notifications. It is independent
    of Kafka and is triggered separately from the event notifier when an event occurs.

    Args:
        to_emails: List of recipient email addresses.
        event_payload: Same structure as Kafka payload (event, agent, camera, metadata).
        frame_base64: Base64-encoded JPEG of the processed/annotated frame (evidence).

    Returns:
        True if send succeeded, False otherwise. Logs errors; does not raise.
    """
    if not AIOSMTP_AVAILABLE:
        print("[email_service] ❌ aiosmtplib not available. Install with: pip install aiosmtplib")
        logger.warning("[email_service] aiosmtplib not available")
        return False

    if not to_emails:
        print("[email_service] ⚠️ No recipients, skip send")
        return False

    settings = get_settings()
    if not settings.smtp_host or not settings.smtp_user or not settings.smtp_password:
        print(
            "[email_service] ❌ SMTP not configured. Set SMTP_HOST, SMTP_USER, SMTP_PASSWORD in .env"
        )
        logger.warning("[email_service] SMTP not configured")
        return False

    event_id = (event_payload.get("metadata") or {}).get("event_id") or "event"
    print(
        f"[email_service] 📤 Attempting to send event email | event_id={event_id} | "
        f"recipients={to_emails} | smtp={settings.smtp_host}:{settings.smtp_port}"
    )

    try:
        msg = MIMEMultipart("related")
        msg["Subject"] = _build_subject(event_payload)
        msg["From"] = f"{settings.email_from_name} <{settings.email_from}>"
        msg["To"] = ", ".join(to_emails)

        html = _build_html_body(event_payload)
        msg.attach(MIMEText(html, "html", "utf-8"))

        if frame_base64:
            try:
                image_bytes = base64.b64decode(frame_base64)
                att = MIMEImage(image_bytes, _subtype="jpeg")
                att.add_header(
                    "Content-Disposition",
                    "attachment",
                    filename=f"evidence_{event_id}.jpg",
                )
                msg.attach(att)
            except Exception as e:
                logger.warning("[email_service] Could not attach evidence image: %s", e)

        await aiosmtplib.send(
            msg,
            sender=settings.email_from,
            recipients=to_emails,
            hostname=settings.smtp_host,
            port=settings.smtp_port,
            username=settings.smtp_user,
            password=settings.smtp_password,
            use_tls=settings.smtp_use_tls,
        )
        print(
            f"[email_service] ✅ Email sent successfully | event_id={event_id} | to={to_emails}"
        )
        logger.info("[email_service] Event notification email sent to %s | event_id=%s", to_emails, event_id)
        return True
    except Exception as e:
        print(f"[email_service] ❌ Email send failed | event_id={event_id} | error={e}")
        logger.exception("[email_service] Failed to send event email: %s", e)
        return False
