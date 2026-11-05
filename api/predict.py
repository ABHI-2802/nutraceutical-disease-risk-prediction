from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import sys
import math
import mimetypes
from pathlib import Path

# Determine root path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Load linear weights for zero-dependency high-speed predictions
LINEAR_WEIGHTS = {}
weights_path = BASE_DIR / "models/linear_weights.json"
if not weights_path.exists():
    weights_path = BASE_DIR / "linear_weights.json"

if weights_path.exists():
    try:
        with open(weights_path, "r", encoding="utf-8") as f:
            LINEAR_WEIGHTS = json.load(f)
    except Exception as e:
        print(f"Note: Could not load linear weights: {e}", file=sys.stderr)

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
        score = 0.20  # default baseline
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


class handler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        payload = {
            "status": "healthy",
            "service": "Nutraceutical Disease Risk Prediction API",
            "version": "1.0.0",
            "targets_available": TARGETS
        }
        self.wfile.write(json.dumps(payload).encode("utf-8"))

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
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(result).encode("utf-8"))


class LocalDevHandler(handler):
    def do_GET(self):
        parsed_path = self.path.split("?")[0]
        if parsed_path.startswith("/api/"):
            super().do_GET()
            return

        # Check public/ then root
        target_file = None
        rel_path = "index.html" if parsed_path in ["", "/"] else parsed_path.lstrip("/")

        for base in [BASE_DIR / "public", BASE_DIR]:
            candidate = base / rel_path
            if candidate.exists() and candidate.is_file():
                target_file = candidate
                break

        if not target_file:
            for base in [BASE_DIR / "public", BASE_DIR]:
                candidate = base / "index.html"
                if candidate.exists() and candidate.is_file():
                    target_file = candidate
                    break

        if target_file and target_file.is_file():
            self.send_response(200)
            mime_type, _ = mimetypes.guess_type(str(target_file))
            self.send_header("Content-Type", mime_type or "text/html")
            self._send_cors_headers()
            self.end_headers()
            with open(target_file, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")


def wsgi_app(environ, start_response):
    method = environ.get("REQUEST_METHOD", "GET").upper()
    path = environ.get("PATH_INFO", "/")

    cors_headers = [
        ("Access-Control-Allow-Origin", "*"),
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
        ("Access-Control-Allow-Headers", "Content-Type, Authorization")
    ]

    if method == "OPTIONS":
        start_response("204 No Content", cors_headers)
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
            headers = cors_headers + [("Content-Type", "application/json")]
            start_response(status, headers)
            return [json.dumps(res).encode("utf-8")]
        else:
            headers = cors_headers + [("Content-Type", "application/json")]
            start_response("200 OK", headers)
            return [json.dumps({"status": "healthy", "service": "Nutraceutical Risk API"}).encode("utf-8")]

    # Static assets
    rel_path = "index.html" if path in ["", "/"] else path.lstrip("/")
    target_file = None
    for base in [BASE_DIR / "public", BASE_DIR]:
        candidate = base / rel_path
        if candidate.exists() and candidate.is_file():
            target_file = candidate
            break

    if not target_file:
        for base in [BASE_DIR / "public", BASE_DIR]:
            candidate = base / "index.html"
            if candidate.exists() and candidate.is_file():
                target_file = candidate
                break

    if target_file and target_file.is_file():
        mime, _ = mimetypes.guess_type(str(target_file))
        headers = cors_headers + [("Content-Type", mime or "text/html")]
        start_response("200 OK", headers)
        with open(target_file, "rb") as f:
            return [f.read()]

    start_response("404 Not Found", cors_headers)
    return [b"Not Found"]


# Dual exports for Vercel
app = wsgi_app
application = wsgi_app

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 3000))
    server = HTTPServer(("0.0.0.0", PORT), LocalDevHandler)
    print(f"🚀 Dashboard running at http://localhost:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        server.server_close()
