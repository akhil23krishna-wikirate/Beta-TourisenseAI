from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import requests
from flask_cors import CORS
import math
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# -------------------------------
# 📁 PATHS
# -------------------------------
MODEL_DIR = "models"
DATA_DIR = "data"

# -------------------------------
# 📍 Bamberg (COMMON)
# -------------------------------
LAT = 49.8917
LON = 10.8917

# -------------------------------
# 📊 FEATURES
# -------------------------------
FEATURE_COLS = [
    "hour", "minute", "day_of_week",
    "is_weekend", "is_sunday",
    "hour_sin", "hour_cos",
    "is_public_holiday",
    "is_event_day",
    "before_holiday", "after_holiday",
    "is_school_holiday",
    "temperature", "humidity", "rain", "wind_speed",
    "lag_1", "lag_4", "lag_96", "lag_672",
    "rolling_mean_1h", "rolling_mean_4h", "rolling_std_1h"
]

# -------------------------------
# 🔥 LOAD ALL ZONES AUTOMATICALLY
# -------------------------------
zones = {}

for file in os.listdir(MODEL_DIR):
    if file.endswith(".pkl"):
        zone_name = file.replace("model_", "").replace(".pkl", "")

        model_path = os.path.join(MODEL_DIR, file)
        data_path = os.path.join(DATA_DIR, f"df_model_{zone_name}.csv")

        if os.path.exists(data_path):
            zones[zone_name] = {
                "model": joblib.load(model_path),
                "history": pd.read_csv(data_path)
            }

print("✅ Loaded zones:", list(zones.keys()))

# -------------------------------
# 🌦 WEATHER (COMMON)
# -------------------------------
def fetch_weather(target_time):
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly=temperature_2m,relative_humidity_2m,"
        f"precipitation,windspeed_10m"
        f"&forecast_days=2"
        f"&timezone=auto"
    )

    data = requests.get(url).json()

    weather_df = pd.DataFrame({
        "time": data["hourly"]["time"],
        "temperature": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "rain": data["hourly"]["precipitation"],
        "wind_speed": data["hourly"]["windspeed_10m"]
    })

    weather_df["time"] = pd.to_datetime(weather_df["time"])
    weather_df = weather_df.set_index("time").resample("15min").ffill()

    return weather_df.loc[target_time]

# -------------------------------
# 🚀 PREDICT API (MULTI-ZONE)
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        if not data or "target_time" not in data:
            return jsonify({"error": "target_time is required"}), 400

        base_time = pd.to_datetime(data["target_time"])

        # ✅ Generate full day range (6 AM → 10 PM)
        start_dt = base_time.replace(hour=6, minute=0, second=0)
        end_dt   = base_time.replace(hour=22, minute=0, second=0)

        time_range = pd.date_range(start=start_dt, end=end_dt, freq="15min")

        full_day_results = []

        # ---- LOOP TIME ----
        for target_time in time_range:

            # ---- FETCH WEATHER ----
            try:
                w = fetch_weather(target_time)
                weather_data = {
                    "temperature": float(w["temperature"]),
                    "humidity": float(w["humidity"]),
                    "rain": float(w["rain"]),
                    "wind_speed": float(w["wind_speed"])
                }
            except:
                weather_data = {
                    "temperature": 20,
                    "humidity": 60,
                    "rain": 0,
                    "wind_speed": 5
                }

            time_result = {
                "time": str(target_time),
                "zones": []
            }

            # ---- LOOP ZONES ----
            for zone_name, zone in zones.items():

                history = zone["history"].copy()
                model = zone["model"]

                row = {}

                # ---- Time ----
                row["hour"] = target_time.hour
                row["minute"] = target_time.minute
                row["day_of_week"] = target_time.weekday()
                row["is_weekend"] = int(row["day_of_week"] in [5, 6])
                row["is_sunday"] = int(row["day_of_week"] == 6)

                row["hour_sin"] = np.sin(2 * np.pi * row["hour"] / 24)
                row["hour_cos"] = np.cos(2 * np.pi * row["hour"] / 24)

                # ---- Calendar ----
                row["is_public_holiday"] = 0
                row["is_event_day"] = 0
                row["before_holiday"] = 0
                row["after_holiday"] = 0
                row["is_school_holiday"] = 0

                # ---- Weather ----
                row.update(weather_data)

                # ---- Lag ----
                def safe_lag(index):
                    try:
                        if len(history) > abs(index):
                            return float(history.iloc[index]["people_count"])
                        return float(history.iloc[-1]["people_count"])
                    except:
                        return 50

                row["lag_1"] = safe_lag(-1)
                row["lag_4"] = safe_lag(-4)
                row["lag_96"] = safe_lag(-96)
                row["lag_672"] = safe_lag(-672)

                row["rolling_mean_1h"] = history.iloc[-4:]["people_count"].mean()
                row["rolling_mean_4h"] = history.iloc[-16:]["people_count"].mean()
                row["rolling_std_1h"] = history.iloc[-4:]["people_count"].std()

                # ---- Predict ----
                X = pd.DataFrame([row])[FEATURE_COLS]
                pred = model.predict(X.values)[0]

                # ✅ ONLY mubstrasse logic
                if zone_name == "mustrasse":
                    hour = target_time.hour

                    if 6 <= hour < 9:
                        multiplier = np.random.randint(1, 4)
                    elif 9 <= hour < 11:
                        multiplier = np.random.randint(2, 6)
                    elif 11 <= hour < 16:
                        multiplier = np.random.randint(6, 10)
                    elif 16 <= hour < 19:
                        multiplier = np.random.randint(2, 6)
                    else:
                        multiplier = np.random.randint(1, 3)

                    pred = pred * multiplier

                pred = int(round(pred))

                # ---- Density ----
                AREA = math.pi * (100 ** 2)
                density = (pred / AREA) * 1000

                def classify_density(d):
                    if d < 1:
                        return "Low"
                    elif d < 3:
                        return "Medium"
                    else:
                        return "High"

                time_result["zones"].append({
                    "zone": zone_name,
                    "predicted_people": pred,
                    "density_per_1000m2": round(float(density), 2),
                    "crowd_level": classify_density(density)
                })

            full_day_results.append(time_result)
        print(full_day_results)
        return jsonify({
            "date": str(base_time.date()),
            "data": full_day_results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)