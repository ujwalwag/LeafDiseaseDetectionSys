"""Gunicorn settings for production (e.g. Render). Override with env vars if needed."""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
worker_class = "sync"
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "600"))
graceful_timeout = int(os.environ.get("GUNICORN_GRACEFUL_TIMEOUT", "120"))
keepalive = 5
