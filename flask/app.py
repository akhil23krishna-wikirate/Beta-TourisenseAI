# ─────────────────────────────────────────────────────────────────
#  Bamberg Crowd Prediction — Flask API  (v5)
#
#  Expects per zone in models/:
#    model_{zone}.pkl          XGBoost model
#    calibration_{zone}.pkl    alpha, profile, seasonal_profile,
#                              scaling_table, monthly_hour_mean,
#                              active_start, active_end
#    history_{zone}.pkl        historical DataFrame (time_15min,
#                              people_count, + weather cols)
#
#  POST /predict  { "target_time": "2026-04-02T10:00:00" }
#  GET  /health
# ─────────────────────────────────────────────────────────────────
import os
import math
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import joblib
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    import ephem
    _EPHEM_OK = True
except ImportError:
    _EPHEM_OK = False

# ── App ───────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── Bamberg ───────────────────────────────────────────────────────
LAT, LON           = 49.8917, 10.8917
BAMBERG_LAT_STR    = "49.8917"
BAMBERG_LON_STR    = "10.8917"

# ── Density ───────────────────────────────────────────────────────
AREA_M2 = math.pi * (100 ** 2)   # 100 m radius

# ── v5 Feature columns — ORDER MUST NOT CHANGE ───────────────────
FEATURE_COLS = [
    # time
    "hour", "minute", "day_of_week", "is_weekend", "is_sunday",
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    # annual cycle
    "month", "week_of_year",
    "month_sin",  "month_cos",
    "doy_sin",    "doy_cos",
    "season",
    "is_summer_peak", "is_christmas_season",
    # calendar context
    "is_public_holiday", "is_event_day",
    "before_holiday", "after_holiday",
    "is_school_holiday", "is_bridge_day", "is_fasching",
    # market flags
    "is_market_day", "is_market_hour", "is_saturday_market",
    # weather
    "temperature", "humidity", "rain", "wind_speed", "feels_like",
    "rain_intensity", "wind_category",
    "is_good_weather", "is_bad_weather",
    # daylight / solar
    "daylight_hours", "solar_elevation",
    # seasonal reference
    "monthly_hour_mean",
    # lags
    "lag_1", "lag_4", "lag_96", "lag_672", "lag_35040",
    # rolling
    "rolling_mean_1h", "rolling_mean_4h", "rolling_std_1h",
    # interactions
    "hour_weekend", "hour_holiday",
    "rain_weekend", "temp_weekend",
    "temp_season",  "rain_market",
    # profiles
    "hist_profile",
    "seasonal_hist_profile",
]

# ── Zone name normalisation ───────────────────────────────────────
# Maps pkl filename → frontend-safe zone name (must match your frontend)
ZONE_NAME_MAP = {
    "Gabelmann":                        "gabelman",
    "Maxplatz":                         "maxplatz",
    "Mußstraße":                        "mustrasse",
    "New_Rathaus_Ost_also_Maxplatz":    "newRathawsOst",
    "New_Rathaus_West_also_Maxplatz":   "newRathawsWest",
    "Old_Rathaus":                      "oldRathaws",
    "Sandstraße":                       "sandstrasse",
    "Touristeninformation":             "touristinformation",
    "Domkranz":                         "domkranz",
}

# ═════════════════════════════════════════════════════════════════
#  CALENDAR SETS
# ═════════════════════════════════════════════════════════════════

def _build_public_holidays() -> dict:
    try:
        import holidays as hlib
        ph = {}
        for yr in range(2023, 2028):
            ph.update(hlib.Germany(state="BY", years=yr))
        return ph                        # date → name
    except ImportError:
        log.warning("'holidays' package missing — PH flags will be 0")
        return {}

def _build_school_holidays() -> set:
    ranges = [
        ("2023-02-20","2023-02-24"), ("2023-04-03","2023-04-14"),
        ("2023-05-30","2023-06-09"), ("2023-07-31","2023-09-11"),
        ("2023-10-30","2023-11-03"), ("2023-12-23","2024-01-05"),
        ("2024-02-12","2024-02-16"), ("2024-03-25","2024-04-06"),
        ("2024-05-21","2024-05-31"), ("2024-07-29","2024-09-09"),
        ("2024-10-28","2024-11-01"), ("2024-12-23","2025-01-04"),
        ("2025-02-24","2025-02-28"), ("2025-04-14","2025-04-25"),
        ("2025-06-03","2025-06-13"), ("2025-07-28","2025-09-08"),
        ("2025-10-27","2025-10-31"), ("2025-12-22","2026-01-04"),
        ("2026-02-23","2026-02-27"), ("2026-04-06","2026-04-17"),
        ("2026-06-02","2026-06-12"), ("2026-07-27","2026-09-07"),
        ("2026-10-26","2026-10-30"), ("2026-12-23","2027-01-05"),
    ]
    s = set()
    for a, b in ranges:
        d = pd.to_datetime(a).date()
        while d <= pd.to_datetime(b).date():
            s.add(d); d += timedelta(days=1)
    return s

def _build_fasching() -> set:
    """Fasching in Bavaria: Thursday before Ash Wednesday through Shrove Tuesday."""
    s = set()
    for yr in range(2023, 2028):
        # Easter Sunday (Gauss algorithm)
        a = yr % 19; b = yr // 100; c = yr % 100
        d = b // 4;  e = b % 4;    f = (b + 8) // 25
        g = (b - f + 1) // 3;     h = (19*a + b - d - g + 15) % 30
        i = c // 4;  k = c % 4;   l = (32 + 2*e + 2*i - h - k) % 7
        m = (a + 11*h + 22*l) // 451
        month = (h + l - 7*m + 114) // 31
        day   = ((h + l - 7*m + 114) % 31) + 1
        easter = pd.Timestamp(yr, month, day).date()
        ash_wed   = easter - timedelta(days=46)
        fat_thurs = ash_wed - timedelta(days=3)
        shrove_tue = ash_wed - timedelta(days=1)
        d = fat_thurs
        while d <= shrove_tue:
            s.add(d); d += timedelta(days=1)
    return s

PH           = _build_public_holidays()
SCHOOL_SET   = _build_school_holidays()
FASCHING_SET = _build_fasching()

def is_bridge_day(dt: pd.Timestamp) -> bool:
    d      = dt.date()
    prev_d = d - timedelta(days=1)
    next_d = d + timedelta(days=1)
    if d in PH or d.weekday() >= 5:
        return False
    return ((prev_d in PH or prev_d.weekday() >= 5) and
            (next_d in PH or next_d.weekday() >= 5))

def calendar_ctx(ts: pd.Timestamp) -> dict:
    d = ts.date()
    return {
        "is_public_holiday": int(d in PH),
        "is_school_holiday": int(d in SCHOOL_SET),
        "is_event_day":      0,
        "before_holiday":    int((d + timedelta(1)) in PH),
        "after_holiday":     int((d - timedelta(1)) in PH),
        "is_bridge_day":     int(is_bridge_day(ts)),
        "is_fasching":       int(d in FASCHING_SET),
    }

def day_category(ts: pd.Timestamp) -> str:
    d = ts.date()
    if d in PH:               return "Holiday"
    if ts.weekday() in (5,6): return "Weekend"
    return "Weekday"

# ═════════════════════════════════════════════════════════════════
#  SOLAR  (ephem optional — falls back gracefully)
# ═════════════════════════════════════════════════════════════════

def solar_info(ts: pd.Timestamp):
    """Returns (elevation_deg, daylight_hours). Falls back to (0, 12)."""
    if not _EPHEM_OK:
        return 0.0, 12.0
    try:
        obs       = ephem.Observer()
        obs.lat   = BAMBERG_LAT_STR
        obs.lon   = BAMBERG_LON_STR
        obs.date  = ts.strftime("%Y/%m/%d %H:%M:%S")
        sun       = ephem.Sun(obs)
        sun.compute(obs)
        elev      = float(sun.alt) * (180 / math.pi)

        # Daylight: compute once per day (cache on date string)
        obs2 = ephem.Observer()
        obs2.lat = BAMBERG_LAT_STR; obs2.lon = BAMBERG_LON_STR
        obs2.horizon = "0"
        obs2.date    = ts.strftime("%Y/%m/%d")
        try:
            rise   = obs2.next_rising(ephem.Sun())
            sett   = obs2.next_setting(ephem.Sun())
            dlight = (sett - rise) * 24.0
        except Exception:
            dlight = 12.0
        return max(elev, 0.0), float(dlight)
    except Exception:
        return 0.0, 12.0

# ═════════════════════════════════════════════════════════════════
#  WEATHER
# ═════════════════════════════════════════════════════════════════

_weather_cache: dict = {}

def fetch_weather_day(date_str: str) -> pd.DataFrame:
    if date_str in _weather_cache:
        return _weather_cache[date_str]
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={LAT}&longitude={LON}"
            "&hourly=temperature_2m,relative_humidity_2m,"
            "precipitation,windspeed_10m,apparent_temperature"
            "&forecast_days=3&timezone=auto"
        )
        raw = requests.get(url, timeout=6).json()["hourly"]
        df  = pd.DataFrame({
            "time":        raw["time"],
            "temperature": raw["temperature_2m"],
            "humidity":    raw["relative_humidity_2m"],
            "rain":        raw["precipitation"],
            "wind_speed":  raw["windspeed_10m"],
            "feels_like":  raw["apparent_temperature"],
        })
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").resample("15min").ffill()
        _weather_cache[date_str] = df
        return df
    except Exception as exc:
        log.warning("Weather fetch failed: %s", exc)
        return pd.DataFrame()

_WX_FALLBACK = {
    "temperature": 15.0, "humidity": 60.0,
    "rain": 0.0, "wind_speed": 5.0, "feels_like": 15.0,
}

def weather_at(ts: pd.Timestamp, history: pd.DataFrame) -> dict:
    wdf = fetch_weather_day(ts.strftime("%Y-%m-%d"))
    if not wdf.empty and ts in wdf.index:
        row = wdf.loc[ts]
        return {k: float(row[k])
                for k in ("temperature","humidity","rain","wind_speed","feels_like")}
    # fallback to historical means
    w = {}
    for col in ("temperature","humidity","rain","wind_speed","feels_like"):
        w[col] = (float(history[col].mean())
                  if col in history.columns else _WX_FALLBACK[col])
    return w

# ═════════════════════════════════════════════════════════════════
#  ZONE LOADER
# ═════════════════════════════════════════════════════════════════

def load_zones() -> dict:
    zones = {}
    for fname in sorted(os.listdir(MODEL_DIR)):
        if not (fname.startswith("model_") and fname.endswith(".pkl")):
            continue
        zone_name = fname[len("model_"):-len(".pkl")]
        model_path = os.path.join(MODEL_DIR, fname)
        cal_path   = os.path.join(MODEL_DIR, f"calibration_{zone_name}.pkl")
        hist_path  = os.path.join(MODEL_DIR, f"history_{zone_name}.pkl")

        missing = [p for p in (cal_path, hist_path) if not os.path.exists(p)]
        if missing:
            log.warning("Zone '%s' skipped — missing: %s", zone_name,
                        [os.path.basename(p) for p in missing])
            continue

        cal = joblib.load(cal_path)
        zones[zone_name] = {
            "model":           joblib.load(model_path),
            "history":         joblib.load(hist_path),
            "alpha":           float(cal["alpha"]),
            "profile":         cal["profile"],
            "seasonal_profile": cal["seasonal_profile"],
            "scaling_table":   cal["scaling_table"],
            "monthly_hour_mean": cal["monthly_hour_mean"],
            "active_start":    int(cal.get("active_start", 6)),
            "active_end":      int(cal.get("active_end",   22)),
        }
        log.info("Loaded zone: %s  (alpha=%.2f)", zone_name, cal["alpha"])

    if not zones:
        raise RuntimeError("No zones loaded — check models/ directory.")
    return zones

zones = load_zones()
log.info("✅  %d zone(s) ready: %s", len(zones), list(zones.keys()))

# ═════════════════════════════════════════════════════════════════
#  FEATURE BUILDER  (mirrors build_row() from v5 notebook)
# ═════════════════════════════════════════════════════════════════

def build_feature_row(
    ts:       pd.Timestamp,
    zone:     dict,
    w:        dict,
) -> pd.DataFrame:

    history          = zone["history"]
    monthly_hour_mean = zone["monthly_hour_mean"]
    profile          = zone["profile"]
    seasonal_profile = zone["seasonal_profile"]

    h   = ts.hour
    m   = ts.month
    dow = ts.weekday()
    doy = ts.timetuple().tm_yday
    woy = ts.isocalendar()[1]

    season_val = (0 if m in (12,1,2) else
                  1 if m in (3,4,5)  else
                  2 if m in (6,7,8)  else 3)
    is_we = int(dow in (5,6))
    is_ph = int(ts.date() in PH)
    dc    = day_category(ts)

    # ── Time ─────────────────────────────────────────────────
    row = {
        "hour": h, "minute": ts.minute,
        "day_of_week": dow, "is_weekend": is_we, "is_sunday": int(dow==6),
        "hour_sin": np.sin(2*np.pi*h/24), "hour_cos": np.cos(2*np.pi*h/24),
        "dow_sin":  np.sin(2*np.pi*dow/7),"dow_cos":  np.cos(2*np.pi*dow/7),
        # annual cycle
        "month": m, "week_of_year": woy,
        "month_sin": np.sin(2*np.pi*m/12),  "month_cos": np.cos(2*np.pi*m/12),
        "doy_sin":   np.sin(2*np.pi*doy/365),"doy_cos":  np.cos(2*np.pi*doy/365),
        "season":    season_val,
        "is_summer_peak":      int(m in (6,7,8)),
        "is_christmas_season": int(m in (11,12)),
        # market flags
        "is_market_day":      int(dow in (0,1,2,3,4,5)),
        "is_saturday_market": int(dow==5),
        "is_market_hour":     int((h>=7) and (h<13) and (dow in (0,1,2,3,4,5))),
    }

    # ── Calendar ──────────────────────────────────────────────
    row.update(calendar_ctx(ts))

    # ── Weather ───────────────────────────────────────────────
    temp = w["temperature"];  rain = w["rain"];  wind = w["wind_speed"]
    row.update(w)
    row["rain_intensity"]  = (0 if rain<=0   else 1 if rain<2.5 else 2 if rain<7.5 else 3)
    row["wind_category"]   = (0 if wind<5    else 1 if wind<15  else 2 if wind<25  else 3)
    row["is_good_weather"] = int((temp>=15)  and (rain<1.0) and (wind<20))
    row["is_bad_weather"]  = int((temp<5)    or  (rain>=5)  or  (wind>=25))

    # ── Solar ─────────────────────────────────────────────────
    elev, dlight = solar_info(ts)
    row["solar_elevation"] = elev
    row["daylight_hours"]  = dlight

    # ── Lag features ─────────────────────────────────────────
    pc = (history["people_count"]
          if "people_count" in history.columns else pd.Series([50.0]))

    def safe_lag(n):
        return float(pc.iloc[-n]) if len(pc)>=n else float(pc.mean())

    row["lag_1"]   = safe_lag(1)
    row["lag_4"]   = safe_lag(4)
    row["lag_96"]  = safe_lag(96)
    row["lag_672"] = safe_lag(672)

    # lag_35040: same slot 1 year ago
    if "time_15min" in history.columns:
        year_ago = ts - pd.Timedelta(days=365)
        hist_ya  = history[history["time_15min"].between(
            year_ago - pd.Timedelta(minutes=15),
            year_ago + pd.Timedelta(minutes=15))]
        if len(hist_ya):
            row["lag_35040"] = float(hist_ya["people_count"].mean())
        else:
            mhm = monthly_hour_mean[
                (monthly_hour_mean["month"]==m) &
                (monthly_hour_mean["day_category"]==dc) &
                (monthly_hour_mean["hour"]==h)]
            row["lag_35040"] = (float(mhm["monthly_hour_mean"].values[0])
                                if len(mhm) else float(pc.mean()))
    else:
        row["lag_35040"] = float(pc.mean())

    # ── Rolling ───────────────────────────────────────────────
    last4  = pc.iloc[-4:]  if len(pc)>=4  else pc
    last16 = pc.iloc[-16:] if len(pc)>=16 else pc
    row["rolling_mean_1h"] = float(last4.mean())
    row["rolling_mean_4h"] = float(last16.mean())
    row["rolling_std_1h"]  = float(last4.std()) if len(last4)>1 else 0.0

    # ── Interactions ─────────────────────────────────────────
    row["hour_weekend"] = h   * is_we
    row["hour_holiday"] = h   * is_ph
    row["rain_weekend"] = rain * is_we
    row["temp_weekend"] = temp * is_we
    row["temp_season"]  = temp * season_val
    row["rain_market"]  = rain * row["is_market_day"]

    # ── Seasonal reference ────────────────────────────────────
    mhm_row = monthly_hour_mean[
        (monthly_hour_mean["month"]==m) &
        (monthly_hour_mean["day_category"]==dc) &
        (monthly_hour_mean["hour"]==h)]
    row["monthly_hour_mean"] = (float(mhm_row["monthly_hour_mean"].values[0])
                                if len(mhm_row) else float(pc.mean()))

    # ── Profile lookups ───────────────────────────────────────
    op = profile[
        (profile["day_category"]==dc) &
        (profile["hour"]==h) &
        (profile["minute"]==ts.minute)]["hist_profile"]
    row["hist_profile"] = float(op.values[0]) if len(op) else float(pc.mean())

    sp = seasonal_profile[
        (seasonal_profile["month"]==m) &
        (seasonal_profile["day_category"]==dc) &
        (seasonal_profile["hour"]==h) &
        (seasonal_profile["minute"]==ts.minute)]["seasonal_hist_profile"]
    row["seasonal_hist_profile"] = (float(sp.values[0]) if len(sp)
                                    else row["hist_profile"])

    return pd.DataFrame([row])[FEATURE_COLS]

# ═════════════════════════════════════════════════════════════════
#  CALIBRATE  (mirrors calibrate() from v5 notebook exactly)
# ═════════════════════════════════════════════════════════════════

def calibrate(pred_arr, time_arr, zone: dict) -> np.ndarray:
    alpha            = zone["alpha"]
    profile          = zone["profile"]
    seasonal_profile = zone["seasonal_profile"]
    scaling_table    = zone["scaling_table"]

    sp_cache = {}
    out      = []

    for raw, ts in zip(pred_arr, time_arr):
        ts = pd.Timestamp(ts)
        m  = ts.month
        dc = day_category(ts)

        sp_row = seasonal_profile[
            (seasonal_profile["month"]==m) &
            (seasonal_profile["day_category"]==dc) &
            (seasonal_profile["hour"]==ts.hour) &
            (seasonal_profile["minute"]==ts.minute)]["seasonal_hist_profile"]

        if len(sp_row) and not pd.isna(sp_row.values[0]):
            pv = float(sp_row.values[0])
        else:
            op = profile[
                (profile["day_category"]==dc) &
                (profile["hour"]==ts.hour) &
                (profile["minute"]==ts.minute)]["hist_profile"]
            pv = float(op.values[0]) if len(op) else float(raw)

        ck = (dc, m)
        if ck not in sp_cache:
            sm = seasonal_profile[
                (seasonal_profile["day_category"]==dc) &
                (seasonal_profile["month"]==m)]["seasonal_hist_profile"]
            sp_cache[ck] = (float(sm.mean()) if len(sm)
                            else float(profile[
                                profile["day_category"]==dc]["hist_profile"].mean()))

        shape_factor = pv / (sp_cache[ck] + 1e-9)
        shaped       = float(raw) * ((1-alpha) + alpha*shape_factor)
        shaped       = shaped * scaling_table.get(m, 1.0)
        out.append(max(5.0, round(shaped)))

    return np.array(out)

# ═════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════

def classify_density(people: int):
    d     = round((people / AREA_M2) * 1000, 2)
    level = "Low" if d < 1 else ("Medium" if d < 3 else "High")
    return d, level

def _is_mussstrasse(name: str) -> bool:
    n = name.lower().replace("ß","ss").replace("ü","u")
    return "mussstrasse" in n or "mustrasse" in n or "mußstraße" in n.replace("ß","ss")

def mustrasse_multiplier(hour: int) -> int:
    if   6  <= hour < 9:  return int(np.random.randint(1, 4))
    elif 9  <= hour < 11: return int(np.random.randint(2, 6))
    elif 11 <= hour < 16: return int(np.random.randint(6, 10))
    elif 16 <= hour < 19: return int(np.random.randint(2, 6))
    else:                  return int(np.random.randint(1, 3))

# ═════════════════════════════════════════════════════════════════
#  ZONE NAME MAP
# ═════════════════════════════════════════════════════════════════
ZONE_NAME_MAP = {
    "Gabelmann":                        "gabelman",
    "Maxplatz":                         "maxplatz",
    "Mußstraße":                        "mustrasse",
    "New_Rathaus_Ost_also_Maxplatz":    "newRathawsOst",
    "New_Rathaus_West_also_Maxplatz":   "newRathawsWest",
    "Old_Rathaus":                      "oldRathaws",
    "Sandstraße":                       "sandstrasse",
    "Touristeninformation":             "touristinformation",
    "Domkranz":                         "domkranz",
}

# ═════════════════════════════════════════════════════════════════
#  ROUTES
# ═════════════════════════════════════════════════════════════════
@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True) or {}
    if "target_time" not in body:
        return jsonify({"error": "target_time is required"}), 400
    try:
        base_time = pd.to_datetime(body["target_time"])
    except Exception:
        return jsonify({"error": "Invalid target_time — use ISO format e.g. 2026-04-02T10:00:00"}), 400
    start_dt   = base_time.replace(hour=6,  minute=0, second=0, microsecond=0)
    end_dt     = base_time.replace(hour=22, minute=0, second=0, microsecond=0)
    time_range = pd.date_range(start=start_dt, end=end_dt, freq="15min")
    # Pre-warm weather cache once
    fetch_weather_day(base_time.strftime("%Y-%m-%d"))
    full_day = []
    for ts in time_range:
        time_entry = {"time": str(ts), "zones": []}
        for zone_name, zone in zones.items():
            w    = weather_at(ts, zone["history"])
            X    = build_feature_row(ts, zone, w)
            pred = float(np.maximum(zone["model"].predict(X.values)[0], 0))
            # Calibrate using v5 calibration package
            cal_pred = calibrate(
                np.array([pred]),
                np.array([ts]),
                zone
            )[0]
            # Mußstraße low-signal multiplier
            if _is_mussstrasse(zone_name):
                cal_pred *= mustrasse_multiplier(ts.hour)
            cal_pred = max(0, int(round(cal_pred)))
            density, level = classify_density(cal_pred)
            time_entry["zones"].append({
                "zone":               ZONE_NAME_MAP.get(zone_name, zone_name.lower()),
                "predicted_people":   cal_pred,
                "density_per_1000m2": density,
                "crowd_level":        level,
            })
        full_day.append(time_entry)
    return jsonify({"date": str(base_time.date()), "data": full_day})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "zones": list(zones.keys())})
# ═════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ═════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════
#  DEBUG — hardcode a date here to test predictions
# ═════════════════════════════════════════════════════════════════
DEBUG_DATE = "2026-05-05"   # ← change this to any date 

@app.route("/debug", methods=["GET"])
def debug_predict():
    results = {}
    time_range = pd.date_range(
        start=f"{DEBUG_DATE} 06:00",
        end=f"{DEBUG_DATE} 22:00",
        freq="15min"
    )
    fetch_weather_day(DEBUG_DATE)

    for zone_name, zone in zones.items():
        preds = []
        for ts in time_range:
            w    = weather_at(ts, zone["history"])
            X    = build_feature_row(ts, zone, w)
            pred = float(np.maximum(zone["model"].predict(X.values)[0], 0))
            cal_pred = calibrate(np.array([pred]), np.array([ts]), zone)[0]
            if _is_mussstrasse(zone_name):
                cal_pred *= mustrasse_multiplier(ts.hour)
            cal_pred = max(0, int(round(cal_pred)))
            preds.append({"time": ts.strftime("%H:%M"), "predicted_people": cal_pred})

        results[ZONE_NAME_MAP.get(zone_name, zone_name.lower())] = {
            "peak": max(p["predicted_people"] for p in preds),
            "peak_time": max(preds, key=lambda x: x["predicted_people"])["time"],
            "average": int(sum(p["predicted_people"] for p in preds) / len(preds)),
            "slots": preds
        }

    return jsonify({"debug_date": DEBUG_DATE, "zones": results})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5001)
