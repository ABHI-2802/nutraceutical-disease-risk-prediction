from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Determine root path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    from src.pipeline import TARGETS, engineer_features, DEFAULT_NUMERIC, DEFAULT_CATEGORICAL
except ImportError:
    TARGETS = ["Anemia", "Osteoporosis", "Diabetes", "Cardio"]
    DEFAULT_NUMERIC = [
        "Age", "Height", "Weight", "BMI_calc", "Waist", "Systolic_BP", "Diastolic_BP",
        "Vitamin_D", "Iron", "Calcium", "Vitamin_B12", "Omega_3", "Zinc", "Magnesium", "Protein",
        "Calories", "Sugars", "Fiber", "Total_Fat", "Sat_Fat", "Sodium", "Potassium",
        "Waist_to_Height_Ratio", "Sodium_to_Potassium_Ratio", "Calcium_to_Magnesium_Ratio",
        "VitD_Calcium_Index", "Sugar_to_Fiber_Ratio", "Sat_Fat_Ratio", "Pulse_Pressure",
        "MAP", "BMI_Age_Interaction", "Iron_to_Protein_Ratio"
    ]
    DEFAULT_CATEGORICAL = ["Gender", "Physical_Activity", "Diet_Quality", "BMI_Category"]

    def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        if "Height" in df.columns and "Weight" in df.columns:
            h = pd.to_numeric(df["Height"], errors="coerce")
            w = pd.to_numeric(df["Weight"], errors="coerce")
            height_m = h / 100.0
            df["BMI_calc"] = w / height_m.pow(2)
            df.loc[~np.isfinite(df["BMI_calc"]) | (height_m <= 0) | (w <= 0), "BMI_calc"] = np.nan
            df["BMI_Category"] = pd.cut(
                df["BMI_calc"],
                bins=[0, 18.5, 25, 30, np.inf],
                labels=["Underweight", "Normal", "Overweight", "Obesity"],
                right=False
            ).astype("object")
        if "Waist" in df.columns and "Height" in df.columns:
            df["Waist_to_Height_Ratio"] = df["Waist"] / (df["Height"] + 1e-5)
        if "Sodium" in df.columns and "Potassium" in df.columns:
            df["Sodium_to_Potassium_Ratio"] = df["Sodium"] / (df["Potassium"] + 1e-5)
        if "Calcium" in df.columns and "Magnesium" in df.columns:
            df["Calcium_to_Magnesium_Ratio"] = df["Calcium"] / (df["Magnesium"] + 1e-5)
        if "Vitamin_D" in df.columns and "Calcium" in df.columns:
            df["VitD_Calcium_Index"] = df["Vitamin_D"] * df["Calcium"]
        if "Sugars" in df.columns and "Fiber" in df.columns:
            df["Sugar_to_Fiber_Ratio"] = df["Sugars"] / (df["Fiber"] + 1e-5)
        if "Sat_Fat" in df.columns and "Total_Fat" in df.columns:
            df["Sat_Fat_Ratio"] = df["Sat_Fat"] / (df["Total_Fat"] + 1e-5)
        if "Systolic_BP" in df.columns and "Diastolic_BP" in df.columns:
            df["Pulse_Pressure"] = (df["Systolic_BP"] - df["Diastolic_BP"]).clip(lower=0)
            df["MAP"] = df["Diastolic_BP"] + (df["Pulse_Pressure"] / 3.0)
        if "BMI_calc" in df.columns and "Age" in df.columns:
            df["BMI_Age_Interaction"] = df["BMI_calc"] * df["Age"]
        if "Iron" in df.columns and "Protein" in df.columns:
            df["Iron_to_Protein_Ratio"] = df["Iron"] / (df["Protein"] + 1e-5)
        return df

# Global model cache to avoid re-reading disk on warm invocations
CACHED_MODELS = {}

def get_models():
    if not CACHED_MODELS:
        models_dir = BASE_DIR / "models"
        for t in TARGETS:
            path = models_dir / f"{t.lower()}_pipeline.joblib"
            if path.exists():
                try:
                    CACHED_MODELS[t] = joblib.load(path)
                except Exception as e:
                    print(f"Error loading {t}: {e}", file=sys.stderr)
    return CACHED_MODELS

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
    df = pd.DataFrame([data])
    df = engineer_features(df)
    
    # Ensure all pipeline expected columns are present with NaN for imputer
    for col in DEFAULT_NUMERIC + DEFAULT_CATEGORICAL:
        if col not in df.columns:
            df[col] = np.nan

    models = get_models()
    predictions = {}
    for t in TARGETS:
        label = disease_label(t)
        if t in models:
            try:
                score = float(models[t].predict_proba(df)[0, 1])
                predictions[t] = {
                    "target": t,
                    "label": label,
                    "score": round(score, 4),
                    "percentage": round(score * 100, 1),
                    "state": score_state(score),
                    "available": True
                }
            except Exception as exc:
                predictions[t] = {
                    "target": t,
                    "label": label,
                    "error": str(exc),
                    "available": False
                }
        else:
            predictions[t] = {
                "target": t,
                "label": label,
                "error": "Model file not found",
                "available": False
            }

    # Profile metrics
    height = float(data.get("Height", 170.0))
    weight = float(data.get("Weight", 70.0))
    bmi = weight / ((height / 100.0) ** 2) if height > 0 else 0.0

    bmi_cat = "Normal"
    if bmi < 18.5:
        bmi_cat = "Underweight"
    elif bmi < 25.0:
        bmi_cat = "Normal"
    elif bmi < 30.0:
        bmi_cat = "Overweight"
    else:
        bmi_cat = "Obesity"

    summary = {
        "bmi": round(bmi, 1),
        "bmi_category": bmi_cat,
        "activity": data.get("Physical_Activity", "Moderate"),
        "diet_quality": data.get("Diet_Quality", "Average"),
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
        models = get_models()
        payload = {
            "status": "healthy",
            "service": "Nutraceutical Disease Risk Prediction API",
            "version": "1.0.0",
            "targets_available": list(models.keys()),
            "total_targets": len(TARGETS)
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


# Exports for Vercel Python runtime
app = handler
application = handler


class LocalDevHandler(handler):
    def do_GET(self):
        parsed_path = self.path.split("?")[0]
        if parsed_path.startswith("/api/"):
            super().do_GET()
            return

        import mimetypes
        # Serve static files from public/
        public_dir = BASE_DIR / "public"
        if parsed_path == "/" or parsed_path == "":
            filepath = public_dir / "index.html"
        else:
            rel = parsed_path.lstrip("/")
            filepath = public_dir / rel

        if filepath.exists() and filepath.is_file():
            self.send_response(200)
            mime_type, _ = mimetypes.guess_type(str(filepath))
            self.send_header("Content-Type", mime_type or "application/octet-stream")
            self._send_cors_headers()
            self.end_headers()
            with open(filepath, "rb") as f:
                self.wfile.write(f.read())
        else:
            fallback = public_dir / "index.html"
            if fallback.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self._send_cors_headers()
                self.end_headers()
                with open(fallback, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not Found")


if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 3000))
    server = HTTPServer(("0.0.0.0", PORT), LocalDevHandler)
    print(f"Server started at http://localhost:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        server.server_close()
