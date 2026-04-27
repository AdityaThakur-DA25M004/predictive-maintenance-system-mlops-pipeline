"""
Alert Notifier for the Predictive Maintenance System.

Sends email/webhook notifications for:
  - Data drift detected
  - Model retraining triggered
  - Feedback accuracy degraded
  - High error rate

Configure via environment variables:
  ALERT_EMAIL_ENABLED=true
  ALERT_SMTP_HOST=smtp.gmail.com
  ALERT_SMTP_PORT=587
  ALERT_SMTP_USER=your_email@gmail.com
  ALERT_SMTP_PASSWORD=your_app_password
  ALERT_RECIPIENT=ops-team@yourcompany.com
"""

import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Configuration (from environment)
ALERT_EMAIL_ENABLED = os.environ.get("ALERT_EMAIL_ENABLED", "false").lower() == "true"
SMTP_HOST = os.environ.get("ALERT_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("ALERT_SMTP_PORT", "587"))
SMTP_USER = os.environ.get("ALERT_SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("ALERT_SMTP_PASSWORD", "")
ALERT_SENDER = os.environ.get("ALERT_SENDER", SMTP_USER)
ALERT_RECIPIENT = os.environ.get("ALERT_RECIPIENT", SMTP_USER)

# Feedback accuracy threshold — below this value the accuracy alert fires
ACCURACY_ALERT_THRESHOLD = float(os.environ.get("ACCURACY_ALERT_THRESHOLD", "0.7"))
FEEDBACK_WINDOW = int(os.environ.get("FEEDBACK_WINDOW", "100"))


def _record_alert(alert_type: str, sent: bool) -> None:
    """Increment the Prometheus alert counter (non-fatal if metrics not available)."""
    try:
        from api.metrics import ALERT_NOTIFICATIONS_TOTAL
        channel = "email" if sent else "log"
        ALERT_NOTIFICATIONS_TOTAL.labels(alert_type=alert_type, channel=channel).inc()
    except Exception:
        pass  # metrics not available in Airflow workers — safe to ignore


# Core email sender
def _send_email(subject: str, body_html: str, alert_type: str = "generic") -> bool:
    """
    Send an email alert. Returns True on success, False on failure.
    Silently logs errors — alerts should never crash the main pipeline.
    Records outcome in Prometheus via _record_alert.
    """
    if not ALERT_EMAIL_ENABLED:
        logger.info(f"[ALERT-LOG] {subject}")
        logger.info("[ALERT-LOG] Email disabled. Set ALERT_EMAIL_ENABLED=true to send.")
        _record_alert(alert_type, sent=False)
        return False

    if not SMTP_USER or not SMTP_PASSWORD:
        logger.warning("ALERT_SMTP_USER or ALERT_SMTP_PASSWORD not set. Skipping email.")
        _record_alert(alert_type, sent=False)
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[PM-ALERT] {subject}"
        msg["From"] = ALERT_SENDER
        msg["To"] = ALERT_RECIPIENT
        msg.attach(MIMEText(body_html, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(ALERT_SENDER, [ALERT_RECIPIENT], msg.as_string())

        logger.info(f"Alert email sent: {subject}")
        _record_alert(alert_type, sent=True)
        return True

    except Exception as e:
        logger.error(f"Failed to send alert email: {e}")
        _record_alert(alert_type, sent=False)
        return False


# Alert: Data Drift Detected
def send_drift_alert(drifted_features: list[str], n_drifted: int) -> bool:
    """Send an alert when data drift is detected."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    features_list = "".join(f"<li>{f}</li>" for f in drifted_features)

    subject = f"Data Drift Detected — {n_drifted} features affected"
    body = f"""
    <html><body>
    <h2 style="color: #E74C3C;">⚠️ Data Drift Alert</h2>
    <p><strong>Time:</strong> {timestamp}</p>
    <p><strong>Drifted features ({n_drifted}):</strong></p>
    <ul>{features_list}</ul>
    <p><strong>Recommended action:</strong> Review the drifted features and consider
    triggering model retraining via the <code>/retrain</code> endpoint or Airflow DAG.</p>
    <hr>
    <p style="color: #888;">Predictive Maintenance System — Automated Alert</p>
    </body></html>
    """
    return _send_email(subject, body, alert_type="drift")


# Alert: Retraining Triggered  (called at trigger time from API + Airflow)
def send_retrain_alert(reason: str, model_version: str = "unknown",
                       new_f1: Optional[float] = None,
                       triggered_by: str = "api",
                       data_source: str = "unknown") -> bool:
    """
    Send an alert when model retraining is triggered.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f1_line = f"<p><strong>New F1 score:</strong> {new_f1:.4f}</p>" if new_f1 else ""
    source_color = "#22c55e" if data_source == "uploaded" else "#f59e0b"
    source_label = "Uploaded CSV ✅" if data_source == "uploaded" else "Default dataset ⚠️"

    subject = f"Retraining Triggered — Reason: {reason} | Source: {data_source}"
    body = f"""
    <html><body>
    <h2 style="color: #F59E0B;">🔁 Retraining Alert</h2>
    <p><strong>Time:</strong> {timestamp}</p>
    <p><strong>Triggered by:</strong> {triggered_by}</p>
    <p><strong>Reason:</strong> {reason}</p>
    <p><strong>Current model version:</strong> {model_version}</p>
    <p><strong>Training data source:</strong>
       <span style="color:{source_color}; font-weight:bold;">{source_label}</span></p>
    {f1_line}
    <p><strong>Status:</strong> Retraining pipeline has been triggered.
    Check Airflow UI for progress.</p>
    <hr>
    <p style="color: #888;">Predictive Maintenance System — Automated Alert</p>
    </body></html>
    """
    return _send_email(subject, body, alert_type="retrain")


# ---------------------------------------------------------------------------
# Alert: Training Complete  (called from Airflow after run_training() succeeds)
# ---------------------------------------------------------------------------
def send_training_complete_alert(new_f1: float, model_version: str,
                                  run_id: str, data_source: str = "unknown",
                                  duration_seconds: Optional[float] = None) -> bool:
    """
    Send an alert when a full training run completes successfully via Airflow DAG.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dur_line = (f"<p><strong>Training duration:</strong> {duration_seconds:.0f}s "
                f"({duration_seconds/60:.1f} min)</p>") if duration_seconds else ""
    source_color = "#22c55e" if data_source == "uploaded" else "#f59e0b"
    source_label = "Uploaded CSV ✅" if data_source == "uploaded" else "Default dataset"
    f1_color = "#22c55e" if new_f1 >= 0.7 else "#ef4444"

    subject = f"✅ Training Complete — F1={new_f1:.4f} | version={model_version}"
    body = f"""
    <html><body>
    <h2 style="color: #22c55e;">✅ Training Complete</h2>
    <p><strong>Time:</strong> {timestamp}</p>
    <p><strong>New F1 score:</strong>
       <span style="color:{f1_color}; font-weight:bold; font-size:1.2em;">{new_f1:.4f}</span></p>
    <p><strong>Model version:</strong> {model_version}</p>
    <p><strong>MLflow run ID:</strong> <code>{run_id}</code></p>
    <p><strong>Training data source:</strong>
       <span style="color:{source_color}; font-weight:bold;">{source_label}</span></p>
    {dur_line}
    <p>Restart the API container to load the new model:
       <code>docker compose restart api</code></p>
    <hr>
    <p style="color: #888;">Predictive Maintenance System — Automated Alert</p>
    </body></html>
    """
    return _send_email(subject, body, alert_type="training_complete")


# ---------------------------------------------------------------------------
# Alert: Feedback Accuracy Degraded
# ---------------------------------------------------------------------------
def send_accuracy_alert(rolling_accuracy: float,
                        threshold: float = ACCURACY_ALERT_THRESHOLD,
                        window: int = FEEDBACK_WINDOW) -> bool:
    """Send an alert when live feedback accuracy drops below threshold."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    subject = f"Feedback Accuracy Degraded — {rolling_accuracy:.1%} (threshold: {threshold:.0%})"
    body = f"""
    <html><body>
    <h2 style="color: #E74C3C;">📉 Accuracy Degradation Alert</h2>
    <p><strong>Time:</strong> {timestamp}</p>
    <p><strong>Rolling accuracy (last {window}):</strong>
       <strong style="color:#ef4444;">{rolling_accuracy:.1%}</strong></p>
    <p><strong>Threshold:</strong> {threshold:.0%}</p>
    <p><strong>Recommended action:</strong> The model's real-world performance has
    degraded. Upload a fresh dataset and trigger retraining.</p>
    <hr>
    <p style="color: #888;">Predictive Maintenance System — Automated Alert</p>
    </body></html>
    """
    return _send_email(subject, body, alert_type="accuracy")


# Alert: High Error Rate
def send_error_rate_alert(error_rate: float, threshold: float = 0.05) -> bool:
    """Send an alert when the API error rate exceeds the threshold."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    subject = f"High Error Rate — {error_rate:.1%} (threshold: {threshold:.0%})"
    body = f"""
    <html><body>
    <h2 style="color: #E74C3C;">🚨 Error Rate Alert</h2>
    <p><strong>Time:</strong> {timestamp}</p>
    <p><strong>Current error rate:</strong>
       <strong style="color:#ef4444;">{error_rate:.1%}</strong></p>
    <p><strong>SLO threshold:</strong> {threshold:.0%}</p>
    <p><strong>Recommended action:</strong> Check API logs for errors.
    Run <code>docker compose logs api --tail=100</code></p>
    <hr>
    <p style="color: #888;">Predictive Maintenance System — Automated Alert</p>
    </body></html>
    """
    return _send_email(subject, body, alert_type="error_rate")


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing alert notifier...")
    print(f"Email enabled: {ALERT_EMAIL_ENABLED}")
    print(f"SMTP host: {SMTP_HOST}:{SMTP_PORT}")
    print(f"Sender: {ALERT_SENDER}")
    print(f"Recipient: {ALERT_RECIPIENT}")
    print()

    # Test each alert type (will log to console if email disabled)
    send_drift_alert(["Air temperature [K]", "Torque [Nm]"], n_drifted=2)
    send_retrain_alert("drift_detected", model_version="1.0", new_f1=0.84)
    send_accuracy_alert(rolling_accuracy=0.65, threshold=0.7)
    send_error_rate_alert(error_rate=0.08, threshold=0.05)
    print("\nDone. Check logs above for [ALERT-LOG] messages.")