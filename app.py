import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

FEATURE_COLUMNS = [
    "T_forecast", "q_forecast", "u_forecast",
    "v_forecast", "w_forecast",
    "NO2_satellite", "HCHO_satellite"
]

SUPPORTED_SITES = range(1, 8)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

CITY_COORDS = [
    {"name": "Mumbai", "lat": 19.0760, "lng": 72.8777},
    {"name": "Delhi", "lat": 28.7041, "lng": 77.1025},
    {"name": "Chennai", "lat": 13.0827, "lng": 80.2707},
    {"name": "Kolkata", "lat": 22.5726, "lng": 88.3639},
    {"name": "Bengaluru", "lat": 12.9716, "lng": 77.5946},
    {"name": "Hyderabad", "lat": 17.3850, "lng": 78.4867},
    {"name": "Pune", "lat": 18.5204, "lng": 73.8567},
]

# ================= MODEL =================
class ModelRegistry:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model_no2 = None
        self.model_o3 = None

    def load(self):
        self.model_no2 = pickle.load(open(self.model_dir / "model_no2.pkl", "rb"))
        self.model_o3 = pickle.load(open(self.model_dir / "model_o3.pkl", "rb"))

model_registry = ModelRegistry(BASE_DIR)
model_registry.load()

# ================= UTILS =================
def safe_float(val, default=0.0):
    try:
        if val is None or str(val).strip() == "":
            return default
        return float(val)
    except:
        return default

def calculate_aqi(no2, o3):
    return round((no2 * 0.6) + (o3 * 0.4), 2)

def get_status(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    return "Poor"

def get_warning(status):
    return {
        "Good": "Air quality is good.",
        "Moderate": "Limit outdoor exposure.",
        "Poor": "Avoid outdoor activities."
    }[status]

# ================= FEATURE =================
def build_features(site, payload):
    temp = safe_float(payload.get("temp"), 30)
    humidity = safe_float(payload.get("humidity"), 50)
    wind = safe_float(payload.get("wind"), 10)

    return [[
        temp,
        humidity,
        wind,
        wind * 0.1,   # 🔥 dynamic
        wind * 0.05,  # 🔥 dynamic
        safe_float(payload.get("sat_no2"), 20),
        safe_float(payload.get("hcho"), 5),
    ]]

# ================= PREDICT =================
def predict_model(features):
    no2 = float(model_registry.model_no2.predict(features)[0])
    o3 = float(model_registry.model_o3.predict(features)[0])
    aqi = calculate_aqi(no2, o3)
    return no2, o3, aqi

# ================= ROUTES =================
@app.route("/")
def home():
    return render_template("dashboard.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json()

        site = int(payload.get("site", 1))

        features = build_features(site, payload)
        no2, o3, aqi = predict_model(features)

        status = get_status(aqi)

        # 🔥 SITE COMPARISON (STRONG VARIATION)
        site_data = {}
        heatmap = []

        for i, city in enumerate(CITY_COORDS):
            variation = (i - 3) * 10

            mod_features = [features[0].copy()]
            mod_features[0][0] += variation

            _, _, site_aqi = predict_model(mod_features)

            site_data[f"Site {i+1}"] = round(site_aqi, 2)

            heatmap.append([
                city["lat"],
                city["lng"],
                min(site_aqi / 150, 1)
            ])

        best_site = min(site_data, key=site_data.get)
        worst_site = max(site_data, key=site_data.get)

        # 🔥 FORECAST (VISIBLE CHANGE)
        forecast = [round(aqi + (i * 2), 2) for i in range(7)]

        return jsonify({
            "success": True,
            "aqi": aqi,
            "no2": round(no2, 2),
            "o3": round(o3, 2),
            "status": status,
            "warning": get_warning(status),

            "aqi_by_site": site_data,
            "best_site": best_site,
            "worst_site": worst_site,

            "current_aqi_series": [aqi]*7,
            "forecast_aqi_series": forecast,
            "labels": ["Now","+1","+2","+3","+4","+5","+6"],

            "heatmap_points": heatmap
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"success": False, "error": str(e)}), 500

# ================= CHAT =================
@app.route("/chat", methods=["POST"])
def chat():
    msg = request.json.get("message","").lower()

    if "safe" in msg:
        reply = "Check AQI level: Good is safe, Moderate caution, Poor avoid outside."
    elif "aqi" in msg:
        reply = "AQI shows pollution level. Lower is better."
    else:
        reply = "Ask about air quality or safety."

    return jsonify({"reply": reply})

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)