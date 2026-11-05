"""
Nutraceutical Disease Risk Prediction - Vercel Serverless Application Entrypoint
Defines explicit top-level 'app' (WSGI), 'application' (WSGI), and 'handler' (HTTP).
"""
import json
import math
import mimetypes
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Load model weights for zero-dependency high-speed predictions
LINEAR_WEIGHTS = {}
for p in [BASE_DIR / "linear_weights.json", BASE_DIR / "models/linear_weights.json"]:
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                LINEAR_WEIGHTS = json.load(f)
            break
        except Exception:
            pass

TARGETS = ["Cardio", "Diabetes", "Osteoporosis", "Anemia"]

def disease_label(target: str) -> str:
    labels = {
        "Anemia": "Anemia",
        "Osteoporosis": "Osteoporosis",
        "Diabetes": "Type 2 Diabetes",
        "Cardio": "Cardiovascular Disease",
    }
    return labels.get(target, target)

def score_state(score: float) -> str:
    if score < 0.33:
        return "Lower model score"
    if score < 0.67:
        return "Intermediate model score"
    return "Higher model score"

def predict_profile(data: dict) -> dict:
    height = float(data.get("Height", 170.0))
    weight = float(data.get("Weight", 70.0))
    height_m = height / 100.0 if height > 0 else 1.7
    bmi = weight / (height_m * height_m) if height_m > 0 else 24.2

    bmi_cat = "Normal"
    if bmi < 18.5:
        bmi_cat = "Underweight"
    elif bmi >= 30.0:
        bmi_cat = "Obesity"
    elif bmi >= 25.0:
        bmi_cat = "Overweight"

    waist = float(data.get("Waist", 84.0))
    sys_bp = float(data.get("Systolic_BP", 122.0))
    dia_bp = float(data.get("Diastolic_BP", 78.0))
    pulse_pressure = max(0.0, sys_bp - dia_bp)

    row = {
        "Age": float(data.get("Age", 40)),
        "Height": height,
        "Weight": weight,
        "BMI_calc": bmi,
        "Waist": waist,
        "Systolic_BP": sys_bp,
        "Diastolic_BP": dia_bp,
        "Vitamin_D": float(data.get("Vitamin_D", 25.0)),
        "Iron": float(data.get("Iron", 15.0)),
        "Calcium": float(data.get("Calcium", 900.0)),
        "Vitamin_B12": float(data.get("Vitamin_B12", 2.5)),
        "Omega_3": float(data.get("Omega_3", 1.5)),
        "Zinc": float(data.get("Zinc", 12.0)),
        "Magnesium": float(data.get("Magnesium", 300.0)),
        "Protein": float(data.get("Protein", 65.0)),
        "Calories": None, "Sugars": None, "Fiber": None, "Total_Fat": None, "Sat_Fat": None, "Sodium": None, "Potassium": None,
        "Waist_to_Height_Ratio": waist / (height + 1e-5),
        "Sodium_to_Potassium_Ratio": None,
        "Calcium_to_Magnesium_Ratio": float(data.get("Calcium", 900.0)) / (float(data.get("Magnesium", 300.0)) + 1e-5),
        "VitD_Calcium_Index": float(data.get("Vitamin_D", 25.0)) * float(data.get("Calcium", 900.0)),
        "Sugar_to_Fiber_Ratio": None,
        "Sat_Fat_Ratio": None,
        "Pulse_Pressure": pulse_pressure,
        "MAP": dia_bp + (pulse_pressure / 3.0),
        "BMI_Age_Interaction": bmi * float(data.get("Age", 40)),
        "Iron_to_Protein_Ratio": float(data.get("Iron", 15.0)) / (float(data.get("Protein", 65.0)) + 1e-5)
    }

    numeric_cols = [
        "Age", "Height", "Weight", "BMI_calc", "Waist", "Systolic_BP", "Diastolic_BP",
        "Vitamin_D", "Iron", "Calcium", "Vitamin_B12", "Omega_3", "Zinc", "Magnesium", "Protein",
        "Calories", "Sugars", "Fiber", "Total_Fat", "Sat_Fat", "Sodium", "Potassium",
        "Waist_to_Height_Ratio", "Sodium_to_Potassium_Ratio", "Calcium_to_Magnesium_Ratio",
        "VitD_Calcium_Index", "Sugar_to_Fiber_Ratio", "Sat_Fat_Ratio", "Pulse_Pressure",
        "MAP", "BMI_Age_Interaction", "Iron_to_Protein_Ratio"
    ]

    gender = str(data.get("Gender", "F")).upper()
    activity = str(data.get("Physical_Activity", "Moderate"))
    diet = str(data.get("Diet_Quality", "Average"))

    predictions = {}
    for t in TARGETS:
        t_key = t.lower()
        score = 0.20
        if t_key in LINEAR_WEIGHTS:
            model = LINEAR_WEIGHTS[t_key]
            x = []
            for i, col in enumerate(numeric_cols):
                val = row.get(col)
                if val is None or not math.isfinite(val):
                    val = model["num_medians"][i]
                scaled = (val - model["num_means"][i]) / (model["num_scales"][i] or 1.0)
                x.append(scaled)

            cat_vals = [gender, activity, diet, bmi_cat]
            for i, allowed in enumerate(model["cat_categories"]):
                cur = cat_vals[i]
                for cat in allowed:
                    x.append(1.0 if cat == cur else 0.0)

            z = model["intercept"]
            for i in range(min(len(x), len(model["coef"]))):
                z += model["coef"][i] * x[i]

            score = 1.0 / (1.0 + math.exp(-z))

        predictions[t] = {
            "target": t,
            "label": disease_label(t),
            "score": round(score, 4),
            "percentage": round(score * 100, 1),
            "state": score_state(score),
            "available": True
        }

    summary = {
        "bmi": round(bmi, 1),
        "bmi_category": bmi_cat,
        "activity": activity,
        "diet_quality": diet,
        "nutrients_count": 8
    }

    return {
        "success": True,
        "predictions": predictions,
        "summary": summary
    }


def find_static_file(path: str) -> Path | None:
    rel = "index.html" if path in ["", "/"] else path.lstrip("/")
    for base in [BASE_DIR, BASE_DIR / "public"]:
        candidate = base / rel
        if candidate.exists() and candidate.is_file():
            return candidate
    for base in [BASE_DIR, BASE_DIR / "public"]:
        fallback = base / "index.html"
        if fallback.exists() and fallback.is_file():
            return fallback
    return None


# EXPLICIT TOP-LEVEL CLASS DEF: handler (BaseHTTPRequestHandler)
class handler(BaseHTTPRequestHandler):
    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_GET(self):
        parsed = self.path.split("?")[0]
        if parsed.startswith("/api/"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.end_headers()
            payload = {
                "status": "healthy",
                "service": "Nutraceutical Disease Risk Prediction API",
                "version": "1.0.0",
                "targets": TARGETS
            }
            self.wfile.write(json.dumps(payload).encode("utf-8"))
            return

        target = find_static_file(parsed)
        if target:
            self.send_response(200)
            mime, _ = mimetypes.guess_type(str(target))
            self.send_header("Content-Type", mime or "text/html")
            self._send_cors()
            self.end_headers()
            with open(target, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        try:
            data = json.loads(post_data.decode("utf-8"))
            result = predict_profile(data)
            status_code = 200
        except Exception as e:
            result = {"success": False, "error": str(e)}
            status_code = 400

        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self._send_cors()
        self.end_headers()
        self.wfile.write(json.dumps(result).encode("utf-8"))


# EXPLICIT TOP-LEVEL FUNCTION DEF: app (WSGI Callable)
def app(environ, start_response):
    method = environ.get("REQUEST_METHOD", "GET").upper()
    path = environ.get("PATH_INFO", "/")

    cors = [
        ("Access-Control-Allow-Origin", "*"),
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
        ("Access-Control-Allow-Headers", "Content-Type, Authorization")
    ]

    if method == "OPTIONS":
        start_response("204 No Content", cors)
        return [b""]

    if path in ["/api/predict", "/api/index", "/api"] or path.startswith("/api/"):
        if method == "POST":
            try:
                length = int(environ.get("CONTENT_LENGTH", 0))
                body = environ["wsgi.input"].read(length) if length > 0 else b"{}"
                data = json.loads(body.decode("utf-8"))
                res = predict_profile(data)
                status = "200 OK"
            except Exception as e:
                res = {"success": False, "error": str(e)}
                status = "400 Bad Request"
            headers = cors + [("Content-Type", "application/json")]
            start_response(status, headers)
            return [json.dumps(res).encode("utf-8")]
        else:
            headers = cors + [("Content-Type", "application/json")]
            start_response("200 OK", headers)
            return [json.dumps({"status": "healthy", "service": "Nutraceutical Risk API"}).encode("utf-8")]

    target = find_static_file(path)
    if target:
        mime, _ = mimetypes.guess_type(str(target))
        headers = cors + [("Content-Type", mime or "text/html")]
        start_response("200 OK", headers)
        with open(target, "rb") as f:
            return [f.read()]

    start_response("404 Not Found", cors)
    return [b"Not Found"]


# EXPLICIT TOP-LEVEL ASSIGNMENT: application (WSGI Alias)
application = app

__all__ = ["app", "application", "handler", "predict_profile"]

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    server = HTTPServer(("0.0.0.0", port), handler)
    print(f"🚀 Nutraceutical Dashboard running at http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        server.server_close()
