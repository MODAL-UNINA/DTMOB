# type: ignore
from pathlib import Path
import os

HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "8080"))

bind = f"{HOST}:{PORT}"
module = "DTMOB_webapp.wsgi:application"

workers = int(os.getenv("GUNICORN_WORKERS", "8"))
threads = 1

timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))

# Ensure log directory exists
LOG_DIR = Path(os.getenv("LOGS_DIR", "/logs"))

print(f"Log directory: {LOG_DIR}")

# Log file locations
accesslog = str(LOG_DIR / "gunicorn_access.log")
errorlog = str(LOG_DIR / "gunicorn_error.log")

capture_output = True

# Log level
loglevel = "debug"
