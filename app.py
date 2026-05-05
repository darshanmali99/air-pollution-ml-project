import pandas as pd
import pickle
import random
import json
from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta

app = Flask(__name__)

# ---------- LOAD MODELS ----------
model_no2 = pickle.load(open("model_no2.pkl", "rb"))
model_o3 = pickle.load(open("model_o3.pkl", "rb"))

# ---------- AQI ----------
def calculate_aqi(no2, o3):
    return round((no2 * 0.6) + (o3 * 0.4), 2)

def get_status(aqi):
    if aqi <= 50:
        return "Good", "green"
    elif aqi <= 100:
        return "Moderate", "orange"
    else:
        return "Poor", "red"

# ---------- SYSTEM ADVICE ----------
def get_advice(status):
    if status == "Good":
        return """Air quality is GOOD.

✔ Safe for outdoor activities  
✔ Ideal for all age groups  
✔ No health risks  

Recommendation:
Enjoy outdoor activities freely."""
    
    elif status == "Moderate":
        return """Air quality is MODERATE.

⚠ Sensitive individuals may feel discomfort  
⚠ Avoid long outdoor exposure  

Safety:
- Reduce outdoor exercise  
- Stay hydrated  
- Use mask if needed"""
    
    else:
        return """Air quality is POOR.

🚨 High health risk  
🚨 Harmful for lungs  

Immediate actions:
- Avoid outdoor activities  
- Wear N95 mask  
- Stay indoors  
- Use air purifiers"""

# ---------- HOME ----------
@app.route('/')
def home():
    return render_template('home.html')

# ---------- PREDICT ----------
@app.route('/predict', methods=['GET','POST'])
def predict():

    if request.method == 'POST':

        site = request.form.get("site", "1")

        # LOAD DATA
        df = pd.read_csv(f"site_{site}_unseen_input_data.csv")
        df = df.fillna(df.mean())

        X = df[['T_forecast','q_forecast','u_forecast',
                'v_forecast','w_forecast',
                'NO2_satellite','HCHO_satellite']]

        # USER INPUT
        temp = request.form.get("temp")
        humidity = request.form.get("humidity")
        wind = request.form.get("wind")
        sat_no2 = request.form.get("sat_no2")
        hcho = request.form.get("hcho")

        # USE INPUT OR DEFAULT DATA
        if temp and humidity and wind:
            features = [[
                float(temp),
                float(humidity),
                float(wind),
                0, 0,
                float(sat_no2 or 0),
                float(hcho or 0)
            ]]
        else:
            features = X.iloc[0].values.reshape(1,-1)

        # ---------- PREDICTION ----------
        no2 = model_no2.predict(features)[0]
        o3 = model_o3.predict(features)[0]

        # ---------- AQI ----------
        aqi = calculate_aqi(no2, o3)
        status, color = get_status(aqi)
        advice = get_advice(status)

        # ---------- SITE COMPARISON ----------
        site_aqi = {}

        for i in range(1, 8):
            try:
                temp_df = pd.read_csv(f"site_{i}_unseen_input_data.csv")
                temp_df = temp_df.fillna(temp_df.mean())

                temp_X = temp_df[['T_forecast','q_forecast','u_forecast',
                                  'v_forecast','w_forecast',
                                  'NO2_satellite','HCHO_satellite']]

                temp_features = temp_X.iloc[0].values.reshape(1,-1)

                temp_no2 = model_no2.predict(temp_features)[0]
                temp_o3 = model_o3.predict(temp_features)[0]

                temp_aqi = calculate_aqi(temp_no2, temp_o3)

                site_aqi[f"Site {i}"] = round(temp_aqi, 2)

            except:
                continue

        best_site = min(site_aqi, key=site_aqi.get)
        worst_site = max(site_aqi, key=site_aqi.get)

        # ---------- FORECAST ----------
        trend = [round(no2 + random.uniform(-5,5),2) for _ in range(24)]
        o3_trend = [round(o3 + random.uniform(-5,5),2) for _ in range(24)]

        timestamps = [
            (datetime.now() + timedelta(hours=i)).strftime("%H:%M")
            for i in range(24)
        ]

        best_time = timestamps[trend.index(min(trend))]
        last_updated = datetime.now().strftime("%d %b %Y, %I:%M %p")

        # ---------- STORE FOR CHATBOT ----------
        global last_aqi, last_status, last_advice
        last_aqi = aqi
        last_status = status
        last_advice = advice

        return render_template(
            "predict.html",
            result=round(no2,2),
            o3=round(o3,2),
            aqi=aqi,
            status=status,
            color=color,
            advice=advice,
            trend=trend,
            o3_trend=o3_trend,
            timestamps=timestamps,
            best_time=best_time,
            last_updated=last_updated,
            site=site,

            # NEW FEATURE
            site_aqi=site_aqi,
            best_site=best_site,
            worst_site=worst_site
        )

    return render_template("predict.html")

# ---------- DASHBOARD ----------
@app.route('/dashboard')
def dashboard():
    with open("metrics.json") as f:
        metrics = json.load(f)

    with open("feature_importance.json") as f:
        features = json.load(f)

    return render_template('dashboard.html',
                           metrics=metrics,
                           features=features)

# ---------- CHATBOT ----------
@app.route('/chat', methods=['POST'])
def chat():
    msg = request.form.get("message","").lower()

    global last_aqi, last_status, last_advice

    if "aqi" in msg:
        reply = f"Current AQI is {last_aqi} ({last_status})."

    elif "safe" in msg:
        reply = last_advice

    elif "what should i do" in msg:
        reply = f"Based on AQI {last_aqi}, I recommend:\n{last_advice}"

    else:
        reply = "I analyze air quality and give health recommendations."

    return jsonify({"reply": reply})

# ---------- ABOUT ----------
@app.route('/about')
def about():
    return render_template('about.html')

# ---------- RUN ----------
if __name__ == "__main__":
    app.run(debug=True)