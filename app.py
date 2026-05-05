import pickle
import random
from pathlib import Path
from typing import Dict, Optional

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
SUPPORTED_SITES = range(1, 8)


# ================= MODEL LOADER =================
class ModelRegistry:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model_no2 = None
        self.model_o3 = None

    def load(self):
        with open(self.model_dir / "model_no2.pkl", "rb") as f:
            self.model_no2 = pickle.load(f)

        with open(self.model_dir / "model_o3.pkl", "rb") as f:
            self.model_o3 = pickle.load(f)


model_registry = ModelRegistry(BASE_DIR)
model_registry.load()


# ================= SAFE FLOAT =================
def safe_float(value, default):
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except:
        return float(default)


# ================= UTILS =================
def calculate_aqi(no2, o3):
    return round((no2 * 0.6) + (o3 * 0.4), 2)


def get_status(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    else:
        return "Poor"


def get_advice(status):
    return {
        "Good": "Air quality is good.",
        "Moderate": "Limit outdoor exposure.",
        "Poor": "Avoid outdoor activities."
    }.get(status, "")


# ================= FEATURE BUILDER =================
def build_features(payload: Dict[str, Optional[str]]):
    return [[
        safe_float(payload.get("temp"), 30),
        safe_float(payload.get("humidity"), 50),
        safe_float(payload.get("wind"), 10),
        0.0,
        0.0,
        safe_float(payload.get("sat_no2"), 20),
        safe_float(payload.get("hcho"), 5),
    ]]


# ================= ROUTES =================
@app.get("/")
def home():
    return render_template("dashboard.html")


@app.post("/predict")
def predict():
    try:
        payload = request.form.to_dict()

        # MAIN INPUT
        features = build_features(payload)

        # MAIN PREDICTION
        no2 = float(model_registry.model_no2.predict(features)[0])
        o3 = float(model_registry.model_o3.predict(features)[0])

        aqi = calculate_aqi(no2, o3)
        status = get_status(aqi)

        # ================= SITE VARIATION =================
        site_profiles = {
            1: {"temp": 25, "humidity": 40, "wind": 5},
            2: {"temp": 30, "humidity": 50, "wind": 8},
            3: {"temp": 35, "humidity": 60, "wind": 10},
            4: {"temp": 40, "humidity": 70, "wind": 12},
            5: {"temp": 45, "humidity": 80, "wind": 15},
            6: {"temp": 28, "humidity": 55, "wind": 7},
            7: {"temp": 32, "humidity": 65, "wind": 9},
        }

        site_comparison = {}

        for site_id in SUPPORTED_SITES:
            base = site_profiles[site_id]

            modified = [[
                base["temp"] + random.uniform(-5, 5),
                base["humidity"] + random.uniform(-10, 10),
                base["wind"] + random.uniform(-3, 3),
                0.0,
                0.0,
                safe_float(payload.get("sat_no2"), 20),
                safe_float(payload.get("hcho"), 5),
            ]]

            no2_s = float(model_registry.model_no2.predict(modified)[0])
            o3_s = float(model_registry.model_o3.predict(modified)[0])

            aqi_s = calculate_aqi(no2_s, o3_s)

            site_comparison[f"Site {site_id}"] = round(aqi_s, 2)

        best_site = min(site_comparison, key=site_comparison.get)
        worst_site = max(site_comparison, key=site_comparison.get)

        return jsonify({
            "success": True,

            "prediction": {
                "aqi": aqi,
                "no2": round(no2, 2),
                "o3": round(o3, 2),
                "status": status,
                "advice": get_advice(status)
            },

            "comparison": {
                "aqi_by_site": site_comparison,
                "best_site": best_site,
                "worst_site": worst_site
            }
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)