"""
Vercel Default Python Entrypoint (/api/index.py)
Exports 'app', 'application', and 'handler' to satisfy all Vercel serverless specifications.
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from api.predict import handler, app, application, wsgi_app, predict_profile, LocalDevHandler

__all__ = ["handler", "app", "application", "wsgi_app", "predict_profile", "LocalDevHandler"]

if __name__ == "__main__":
    from http.server import HTTPServer
    import os
    port = int(os.environ.get("PORT", 3000))
    server = HTTPServer(("0.0.0.0", port), LocalDevHandler)
    print(f"Server started at http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        server.server_close()
