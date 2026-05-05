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
    "T_forecast",
    "q_forecast",
    "u_forecast",
    "v_forecast",
    "w_forecast",
    "NO2_satellite",
    "HCHO_satellite",
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


class ModelRegistry:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.model_no2 = None
        self.model_o3 = None

    def load(self) -> None:
        no2_path = self.model_dir / "model_no2.pkl"
        o3_path = self.model_dir / "model_o3.pkl"

        if not no2_path.exists() or not o3_path.exists():
            raise FileNotFoundError("Model files not found. Expected model_no2.pkl and model_o3.pkl.")

        with no2_path.open("rb") as f:
            self.model_no2 = pickle.load(f)
        with o3_path.open("rb") as f:
            self.model_o3 = pickle.load(f)


model_registry = ModelRegistry(BASE_DIR)
model_registry.load()


def safe_float(value: Optional[str], default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def calculate_aqi(no2: float, o3: float) -> float:
    return round((no2 * 0.6) + (o3 * 0.4), 2)


def get_status(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    return "Poor"


def get_warning(status: str) -> str:
    warnings = {
        "Good": "Air quality is good. Outdoor activity is generally safe.",
        "Moderate": "Air quality is moderate. Sensitive groups should reduce prolonged outdoor exposure.",
        "Poor": "Air quality is poor. Avoid strenuous outdoor activity and use protection.",
    }
    return warnings.get(status, "Air quality data unavailable.")


def load_site_data(site: int) -> pd.DataFrame:
    if site not in SUPPORTED_SITES:
        raise ValueError(f"Invalid site '{site}'. Supported sites: 1-7.")

    data_path = BASE_DIR / f"site_{site}_unseen_input_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file missing for site {site}: {data_path.name}")

    df = pd.read_csv(data_path)
    missing_columns = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns for site {site}: {missing_columns}")

    return df.fillna(df.mean(numeric_only=True))


def build_features(site: int, payload: Dict[str, Optional[str]]) -> List[List[float]]:
    temp = payload.get("temp")
    humidity = payload.get("humidity")
    wind = payload.get("wind")
    custom_input_present = bool(temp and humidity and wind)

    if custom_input_present:
        return [[
            safe_float(temp),
            safe_float(humidity),
            safe_float(wind),
            0.0,
            0.0,
            safe_float(payload.get("sat_no2"), 0.0),
            safe_float(payload.get("hcho"), 0.0),
        ]]

    df = load_site_data(site)
    return [df[FEATURE_COLUMNS].iloc[0].astype(float).tolist()]


def predict_with_model(features: List[List[float]]) -> Tuple[float, float, float, str]:
    no2 = float(model_registry.model_no2.predict(features)[0])
    o3 = float(model_registry.model_o3.predict(features)[0])
    aqi = calculate_aqi(no2, o3)
    status = get_status(aqi)
    return round(no2, 2), round(o3, 2), aqi, status


def fetch_openweather_aqi(city_lat: float, city_lng: float) -> Optional[float]:
    if not OPENWEATHER_API_KEY:
        return None
    try:
        url = "https://api.openweathermap.org/data/2.5/air_pollution"
        params = {"lat": city_lat, "lon": city_lng, "appid": OPENWEATHER_API_KEY}
        response = requests.get(url, params=params, timeout=6)
        response.raise_for_status()
        payload = response.json()
        api_index = payload.get("list", [{}])[0].get("main", {}).get("aqi")
        if api_index is None:
            return None
        return round(float(api_index) * 40.0, 2)
    except Exception:
        return None


def build_forecast_series(base_aqi: float, points: int = 7) -> List[float]:
    offsets = [-5, -2, 0, 3, 6, 2, -1]
    return [round(max(0.0, base_aqi + offsets[i % len(offsets)]), 2) for i in range(points)]


@app.get("/")
def home():
    return render_template("dashboard.html")


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(silent=True) or request.form.to_dict() or {}
        site = int(payload.get("site", 1))

        features = build_features(site=site, payload=payload)
        no2, o3, model_aqi, status = predict_with_model(features)

        city_ref = CITY_COORDS[(site - 1) % len(CITY_COORDS)]
        api_aqi = fetch_openweather_aqi(city_ref["lat"], city_ref["lng"])
        final_aqi = api_aqi if api_aqi is not None else model_aqi
        status = get_status(final_aqi)

        site_comparison = {}
        heatmap_points = []

        for i, city in enumerate(CITY_COORDS, start=1):
            try:
                site_features = build_features(min(i, 7), {})
                _, _, site_model_aqi, _ = predict_with_model(site_features)
                site_api_aqi = fetch_openweather_aqi(city["lat"], city["lng"])
                site_final_aqi = site_api_aqi if site_api_aqi is not None else site_model_aqi
                site_comparison[f"Site {i}"] = round(site_final_aqi, 2)
                heatmap_points.append([city["lat"], city["lng"], round(min(site_final_aqi / 150.0, 1.0), 3)])
            except Exception:
                site_comparison[f"Site {i}"] = 60.0
                heatmap_points.append([city["lat"], city["lng"], 0.4])

        best_site = min(site_comparison, key=site_comparison.get)
        worst_site = max(site_comparison, key=site_comparison.get)

        current_series = [round(final_aqi, 2)] * 7
        forecast_series = build_forecast_series(final_aqi, 7)

        return jsonify({
            "success": True,
            "aqi": round(final_aqi, 2),
            "no2": no2,
            "o3": o3,
            "status": status,
            "warning": get_warning(status),
            "best_site": best_site,
            "worst_site": worst_site,
            "aqi_by_site": site_comparison,
            "current_aqi_series": current_series,
            "forecast_aqi_series": forecast_series,
            "labels": ["Now", "+1h", "+2h", "+3h", "+4h", "+5h", "+6h"],
            "heatmap_points": heatmap_points,
            "sites_geo": CITY_COORDS,
            "source": "openweather" if api_aqi is not None else "ml_fallback",
        })
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"success": False, "error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"success": False, "error": f"Internal server error: {exc}"}), 500


@app.post("/chat")
def chat():
    try:
        payload = request.get_json(silent=True) or {}
        message = (payload.get("message") or "").strip().lower()

        if not message:
            return jsonify({"success": True, "reply": "Please type your question about air quality."})

        if "safe" in message:
            reply = "Check AQI status: Good is generally safe, Moderate needs caution, Poor requires protection."
        elif "poor" in message or "warning" in message:
            reply = "For Poor AQI: avoid heavy outdoor activity, wear a mask, and stay hydrated."
        elif "aqi" in message:
            reply = "AQI summarizes pollution risk. Lower is better: Good (0-50), Moderate (51-100), Poor (>100)."
        else:
            reply = "I can help with AQI safety, pollution trends, and protective recommendations."

        return jsonify({"success": True, "reply": reply})
    except Exception as exc:
        return jsonify({"success": False, "reply": f"Chat error: {exc}"}), 500


@app.get("/metrics")
def metrics():
    try:
        metrics_path = BASE_DIR / "metrics.json"
        if not metrics_path.exists():
            return jsonify({"success": False, "error": "metrics.json not found."}), 404

        with metrics_path.open() as f:
            metrics_data = json.load(f)

        return jsonify({"success": True, "metrics": metrics_data})
    except Exception as exc:
        return jsonify({"success": False, "error": f"Failed to load metrics: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
