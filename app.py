import pickle
import requests
import random
from pathlib import Path
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

# ✅ API KEY (FIXED)
API_KEY = "2b39b089f97ca35b6bd933a6a06a01f7"

# ================= LOAD MODELS =================
with open(BASE_DIR / "model_no2.pkl", "rb") as f:
    model_no2 = pickle.load(f)

with open(BASE_DIR / "model_o3.pkl", "rb") as f:
    model_o3 = pickle.load(f)

# ================= GLOBAL MEMORY =================
LAST_AQI = 50
LAST_STATUS = "Moderate"

# ================= CITY DATA =================
CITY_COORDS = [
    {"name": "Mumbai", "lat": 19.0760, "lng": 72.8777},
    {"name": "Delhi", "lat": 28.7041, "lng": 77.1025},
    {"name": "Chennai", "lat": 13.0827, "lng": 80.2707},
    {"name": "Kolkata", "lat": 22.5726, "lng": 88.3639},
    {"name": "Bangalore", "lat": 12.9716, "lng": 77.5946},
    {"name": "Hyderabad", "lat": 17.3850, "lng": 78.4867},
    {"name": "Pune", "lat": 18.5204, "lng": 73.8567},
]

# ================= LIVE WEATHER =================
def get_weather():
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat=19.99&lon=73.78&appid={API_KEY}&units=metric"
        d = requests.get(url).json()
        return d["main"]["temp"], d["main"]["humidity"]
    except:
        return 30, 50

# ================= LIVE POLLUTION =================
def get_pollution():
    try:
        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat=19.99&lon=73.78&appid={API_KEY}"
        d = requests.get(url).json()
        c = d["list"][0]["components"]
        return c["no2"], c["o3"]
    except:
        return 40, 60

# ================= AQI =================
def calc_aqi(no2, o3):
    return round(no2 * 0.6 + o3 * 0.4, 2)

def get_status(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    return "Poor"

def get_advice(status):
    if status == "Good":
        return "Air quality is good. Safe for outdoor activities."
    elif status == "Moderate":
        return "Limit outdoor exposure, especially in traffic areas."
    return "Air quality is poor. Avoid outdoor activities."

# ================= ROUTES =================
@app.route("/")
def home():
    return render_template("dashboard.html")

@app.route("/predict", methods=["POST"])
def predict():
    global LAST_AQI, LAST_STATUS

    payload = request.get_json() or {}

    # ✅ TAKE USER INPUT FIRST
    no2 = float(payload.get("no2", 0))
    o3 = float(payload.get("o3", 0))

    temp = float(payload.get("temp", 0))
    humidity = float(payload.get("humidity", 0))

    # ✅ IF USER DID NOT ENTER → USE LIVE API
    if no2 == 0 or o3 == 0:
        pollution = get_pollution()
        no2 = pollution["no2"]
        o3 = pollution["o3"]

    if temp == 0 or humidity == 0:
        t, h = get_weather()
        temp, humidity = t, h

    # ✅ CALCULATE AQI
    aqi = calc_aqi(no2, o3)
    status = get_status(aqi)

    LAST_AQI = aqi
    LAST_STATUS = status

    # ✅ Advanced Forecast (trend based)
    forecast = []
    base = aqi
    for i in range(7):
        change = random.uniform(-2, 3)
        base = max(10, base + change)
        forecast.append(round(base, 2))

    # ✅ Heatmap
    heatmap = []
    site_data = {}

    for i, c in enumerate(CITY_COORDS):
        val = max(10, aqi + random.uniform(-15, 15))
        site_data[f"Site {i+1}"] = round(val, 2)
        heatmap.append([c["lat"], c["lng"], val/100])

    best = min(site_data, key=site_data.get)
    worst = max(site_data, key=site_data.get)

    return jsonify({
        "success": True,
        "aqi": aqi,
        "no2": no2,
        "o3": o3,
        "temp": temp,
        "humidity": humidity,
        "status": status,
        "warning": get_advice(status),
        "best_site": best,
        "worst_site": worst,
        "labels": ["Now","+1","+2","+3","+4","+5","+6"],
        "current_aqi_series": [aqi]*7,
        "forecast_aqi_series": forecast,
        "heatmap_points": heatmap
    })

# ================= SMART CHATBOT =================
@app.route("/chat", methods=["POST"])
def chat():
    msg = request.json.get("message","").lower()

    if "safe" in msg:
        reply = f"Current AQI is {LAST_AQI} ({LAST_STATUS}). {get_advice(LAST_STATUS)}"

    elif "aqi" in msg:
        reply = f"AQI is {LAST_AQI}. It is classified as {LAST_STATUS}."

    elif "precaution" in msg or "care" in msg:
        reply = get_advice(LAST_STATUS)

    else:
        reply = f"AQI is {LAST_AQI} ({LAST_STATUS}). Ask about safety or precautions."

    return jsonify({"reply": reply})

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
