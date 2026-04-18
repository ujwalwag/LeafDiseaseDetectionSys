"""WSGI entry for hosts that expect `gunicorn app:app` (e.g. Render Flask quickstart)."""
from backend_app import app

__all__ = ["app"]
