import time
import os
import requests
from dotenv import load_dotenv

load_dotenv()

RECIPIENT           = "ulrich.kamsu@camtrack.net"
EMAIL_USER          = os.getenv("EMAIL_USER")
GRAPH_CLIENT_ID     = os.getenv("GRAPH_CLIENT_ID")
GRAPH_TENANT_ID     = os.getenv("GRAPH_TENANT_ID")
GRAPH_CLIENT_SECRET = os.getenv("GRAPH_CLIENT_SECRET")

def _get_access_token():
    url = f"https://login.microsoftonline.com/{GRAPH_TENANT_ID}/oauth2/v2.0/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": GRAPH_CLIENT_ID,
        "client_secret": GRAPH_CLIENT_SECRET,
        "scope": "https://graph.microsoft.com/.default",
    }
    resp = requests.post(url, data=data)
    resp.raise_for_status()
    return resp.json()["access_token"]

def send_email(subject, body):
    try:
        token = _get_access_token()
        url = f"https://graph.microsoft.com/v1.0/users/{EMAIL_USER}/sendMail"
        payload = {
            "message": {
                "subject": subject,
                "body": {"contentType": "Text", "content": body},
                "toRecipients": [{"emailAddress": {"address": RECIPIENT}}],
            }
        }
        resp = requests.post(url, json=payload,
                             headers={"Authorization": f"Bearer {token}"})
        resp.raise_for_status()
        print(f"[Email sent] {subject}")
    except Exception as e:
        print(f"[Email failed] {e}")


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


class TrainingNotifier:
    def __init__(self, model_name, total_epochs):
        self.model_name = model_name
        self.total_epochs = total_epochs
        self.start_time = None
        self.half_sent = False

    def on_train_start(self):
        self.start_time = time.time()
        send_email(
            subject=f"🚀 Training Started: {self.model_name}",
            body=(
                f"Training has started for model: {self.model_name}\n"
                f"Total epochs: {self.total_epochs}\n\n"
                f"You will receive updates at 50% and when training completes."
            )
        )

    def on_epoch_end(self, current_epoch):
        if self.start_time is None:
            return

        elapsed = time.time() - self.start_time
        progress = current_epoch / self.total_epochs
        remaining = (elapsed / progress) - elapsed if progress > 0 else 0

        # Send at 50%
        if not self.half_sent and current_epoch >= self.total_epochs // 2:
            self.half_sent = True
            send_email(
                subject=f"⏳ Training 50% Done: {self.model_name}",
                body=(
                    f"Model: {self.model_name}\n"
                    f"Progress: {current_epoch}/{self.total_epochs} epochs (50%)\n"
                    f"Elapsed: {format_time(elapsed)}\n"
                    f"Estimated remaining: {format_time(remaining)}"
                )
            )

    def on_train_end(self, best_model_path=""):
        elapsed = time.time() - self.start_time if self.start_time else 0
        send_email(
            subject=f"✅ Training Complete: {self.model_name}",
            body=(
                f"Training finished for model: {self.model_name}\n"
                f"Total time: {format_time(elapsed)}\n"
                f"Best model saved at: {best_model_path}"
            )
        )
