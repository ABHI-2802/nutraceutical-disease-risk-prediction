"""
Nutraceutical Disease Risk Prediction - Web Application Entrypoint
Exports top-level 'app', 'application', and 'handler' variables required by Vercel.
"""
import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from api.predict import handler, LocalDevHandler, predict_profile

# Top-level exports for Vercel Python Runtime ([PYTHON_ENTRYPOINT_NOT_FOUND] resolution)
handler = LocalDevHandler
app = LocalDevHandler
application = LocalDevHandler

__all__ = ["app", "application", "handler", "predict_profile"]

if __name__ == "__main__":
    from http.server import HTTPServer
    port = int(os.environ.get("PORT", 3000))
    server = HTTPServer(("0.0.0.0", port), LocalDevHandler)
    print(f"🚀 Nutraceutical Health Intelligence Dashboard running at http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        server.server_close()
