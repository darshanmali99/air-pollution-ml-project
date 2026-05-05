import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
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


def calculate_aqi(no2: float, o3: float) -> float:
    return round((no2 * 0.6) + (o3 * 0.4), 2)


def get_status(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    return "Poor"


def get_advice(status: str) -> str:
    advice_map = {
        "Good": "Air quality is good. Outdoor activities are generally safe.",
        "Moderate": "Air quality is moderate. Sensitive groups should limit prolonged outdoor exposure.",
        "Poor": "Air quality is poor. Limit outdoor activities and use protective measures.",
    }
    return advice_map.get(status, "Monitor local air quality before outdoor activities.")


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


def build_features(site: int, payload: Dict[str, Optional[float]]) -> List[List[float]]:
    custom_input_present = payload.get("temp") is not None and payload.get("humidity") is not None and payload.get("wind") is not None

    if custom_input_present:
        return [[
            float(payload["temp"]),
            float(payload["humidity"]),
            float(payload["wind"]),
            0.0,
            0.0,
            float(payload.get("sat_no2") or 0.0),
            float(payload.get("hcho") or 0.0),
        ]]

    df = load_site_data(site)
    return [df[FEATURE_COLUMNS].iloc[0].astype(float).tolist()]


def predict_for_site(site: int) -> Dict[str, float]:
    df = load_site_data(site)
    features = [df[FEATURE_COLUMNS].iloc[0].astype(float).tolist()]
    no2 = float(model_registry.model_no2.predict(features)[0])
    o3 = float(model_registry.model_o3.predict(features)[0])
    return {"no2": round(no2, 2), "o3": round(o3, 2), "aqi": calculate_aqi(no2, o3)}


@app.get("/")
def home():
    return render_template("dashboard.html")


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(silent=True) or request.form.to_dict()
        site = int(payload.get("site", 1))

        features = build_features(site=site, payload=payload)
        no2 = float(model_registry.model_no2.predict(features)[0])
        o3 = float(model_registry.model_o3.predict(features)[0])

        aqi = calculate_aqi(no2, o3)
        status = get_status(aqi)

        site_comparison = {}
        for site_id in SUPPORTED_SITES:
            try:
                result = predict_for_site(site_id)
                site_comparison[f"Site {site_id}"] = result["aqi"]
            except (FileNotFoundError, ValueError):
                continue

        if not site_comparison:
            raise RuntimeError("No site data available for comparison.")

        best_site = min(site_comparison, key=site_comparison.get)
        worst_site = max(site_comparison, key=site_comparison.get)

        return jsonify({
            "success": True,
            "input": {"site": site},
            "prediction": {
                "no2": round(no2, 2),
                "o3": round(o3, 2),
                "aqi": aqi,
                "status": status,
                "advice": get_advice(status),
            },
            "site_comparison": {
                "aqi_by_site": site_comparison,
                "best_site": best_site,
                "worst_site": worst_site,
            },
        })
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"success": False, "error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"success": False, "error": f"Internal server error: {exc}"}), 500


@app.get("/metrics")
def metrics():
    metrics_path = BASE_DIR / "metrics.json"
    if not metrics_path.exists():
        return jsonify({"success": False, "error": "metrics.json not found."}), 404

    with metrics_path.open() as f:
        metrics_data = json.load(f)

    return jsonify({"success": True, "metrics": metrics_data})


if __name__ == "__main__":
    app.run(debug=True)
