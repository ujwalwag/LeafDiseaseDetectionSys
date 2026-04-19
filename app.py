"""WSGI entry for hosts that expect `gunicorn app:app` (e.g. Render Flask quickstart)."""
import os

from backend_app import app

__all__ = ["app"]


if __name__ == "__main__":
    if not os.path.exists("templates"):
        os.makedirs("templates")
    app.run(debug=True, host="0.0.0.0", port=5000)
