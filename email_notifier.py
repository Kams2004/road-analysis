import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

RECIPIENT    = "ulrich.kamsu@camtrack.net"
SENDER_EMAIL = "simokylian1@gmail.com"
SENDER_PASSWORD = "ipzqubfqydprothg"  # app password (no spaces)
SMTP_HOST    = "smtp.gmail.com"
SMTP_PORT    = 587


def send_email(subject, body):
    try:
        msg = MIMEMultipart()
        msg["From"]    = SENDER_EMAIL
        msg["To"]      = RECIPIENT
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECIPIENT, msg.as_string())
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
        self.model_name   = model_name
        self.total_epochs = total_epochs
        self.start_time   = None
        self.half_sent    = False

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
        elapsed   = time.time() - self.start_time
        progress  = current_epoch / self.total_epochs
        remaining = (elapsed / progress) - elapsed if progress > 0 else 0

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
