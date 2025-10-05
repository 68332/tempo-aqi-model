import json
import os
import requests
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from openaq import OpenAQ

# AQI brakpoints definition
aqi_breakpoints = {
    "pm25": [
        {"lo": 0.0,   "hi": 12.0,  "aqiLo": 0,   "aqiHi": 50},
        {"lo": 12.1,  "hi": 35.4,  "aqiLo": 51,  "aqiHi": 100},
        {"lo": 35.5,  "hi": 55.4,  "aqiLo": 101, "aqiHi": 150},
        {"lo": 55.5,  "hi": 150.4, "aqiLo": 151, "aqiHi": 200},
        {"lo": 150.5, "hi": 250.4, "aqiLo": 201, "aqiHi": 300},
        {"lo": 250.5, "hi": 350.4, "aqiLo": 301, "aqiHi": 400},
        {"lo": 350.5, "hi": 500.4, "aqiLo": 401, "aqiHi": 500},
    ],
    "pm10": [
        {"lo": 0,   "hi": 54,   "aqiLo": 0,   "aqiHi": 50},
        {"lo": 55,  "hi": 154,  "aqiLo": 51,  "aqiHi": 100},
        {"lo": 155, "hi": 254,  "aqiLo": 101, "aqiHi": 150},
        {"lo": 255, "hi": 354,  "aqiLo": 151, "aqiHi": 200},
        {"lo": 355, "hi": 424,  "aqiLo": 201, "aqiHi": 300},
        {"lo": 425, "hi": 504,  "aqiLo": 301, "aqiHi": 400},
        {"lo": 505, "hi": 604,  "aqiLo": 401, "aqiHi": 500},
    ],
    "co": [
        {"lo": 0.0, "hi": 4.4,  "aqiLo": 0,   "aqiHi": 50},
        {"lo": 4.5, "hi": 9.4,  "aqiLo": 51,  "aqiHi": 100},
        {"lo": 9.5, "hi": 12.4, "aqiLo": 101, "aqiHi": 150},
        {"lo": 12.5,"hi": 15.4, "aqiLo": 151, "aqiHi": 200},
        {"lo": 15.5,"hi": 30.4, "aqiLo": 201, "aqiHi": 300},
        {"lo": 30.5,"hi": 40.4, "aqiLo": 301, "aqiHi": 400},
        {"lo": 40.5,"hi": 50.4, "aqiLo": 401, "aqiHi": 500},
    ],
    "so2": [
        {"lo": 0,   "hi": 35,   "aqiLo": 0,   "aqiHi": 50},
        {"lo": 36,  "hi": 75,   "aqiLo": 51,  "aqiHi": 100},
        {"lo": 76,  "hi": 185,  "aqiLo": 101, "aqiHi": 150},
        {"lo": 186, "hi": 304,  "aqiLo": 151, "aqiHi": 200},
        {"lo": 305, "hi": 604,  "aqiLo": 201, "aqiHi": 300},
        {"lo": 605, "hi": 804,  "aqiLo": 301, "aqiHi": 400},
        {"lo": 805, "hi": 1004, "aqiLo": 401, "aqiHi": 500},
    ],
    "no2": [
        {"lo": 0,    "hi": 53,   "aqiLo": 0,   "aqiHi": 50},
        {"lo": 54,   "hi": 100,  "aqiLo": 51,  "aqiHi": 100},
        {"lo": 101,  "hi": 360,  "aqiLo": 101, "aqiHi": 150},
        {"lo": 361,  "hi": 649,  "aqiLo": 151, "aqiHi": 200},
        {"lo": 650,  "hi": 1249, "aqiLo": 201, "aqiHi": 300},
        {"lo": 1250, "hi": 1649, "aqiLo": 301, "aqiHi": 400},
        {"lo": 1650, "hi": 2049, "aqiLo": 401, "aqiHi": 500},
    ]
}

def load_model_bundle(model_path, scaler_path, config_path):
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return model, scaler, cfg

def predict_next(var, recent_series, model_path, scaler_path, config_path):
    model, scaler, cfg = load_model_bundle(model_path, scaler_path, config_path)
    n_steps = int(cfg["n_steps"])

    arr = np.asarray(recent_series, dtype=float).reshape(-1, 1)
    if arr.shape[0] < n_steps:
        raise ValueError(f"{var}: recent_series 長度({arr.shape[0]})不足 n_steps({n_steps})")

    arr_norm = scaler.transform(arr)
    window = arr_norm[-n_steps:].reshape(1, n_steps, 1)  # (1, timesteps, features)
    yhat_norm = model.predict(window, verbose=0)
    yhat = scaler.inverse_transform(yhat_norm.reshape(-1, 1))[0, 0]
    print(f"{var} inference end")
    return float(yhat)

def predict_sequence(name):
    # Get Input Vector
    gas = fetchgas()
    
    pm25_intput = gas["pm25"]
    pm25_model   = "models/pm25_lstm.keras"
    pm25_scaler  = "models/pm25_lstm.scaler"
    pm25_config  = "models/pm25_lstm.json"

    pm10_intput = gas["pm10"]
    pm10_model   = "models/pm10_lstm.keras"
    pm10_scaler  = "models/pm10_lstm.scaler"
    pm10_config  = "models/pm10_lstm.json"

    o3_intput = gas["o3"]
    o3_model    = "models/o3_lstm.keras"
    o3_scaler   = "models/o3_lstm.scaler"
    o3_config   = "models/o3_lstm.json"

    co_intput = gas["co"]
    co_model    = "models/co_lstm.keras"
    co_scaler   = "models/co_lstm.scaler"
    co_config   = "models/co_lstm.json"

    so2_intput = gas["so2"]
    so2_model   = "models/so2_lstm.keras"
    so2_scaler  = "models/so2_lstm.scaler"
    so2_config  = "models/so2_lstm.json"

    no2_intput = gas["no2"]
    no2_model   = "models/no2_lstm.keras"
    no2_scaler  = "models/no2_lstm.scaler"
    no2_config  = "models/no2_lstm.json"

    pm25_pred = predict_next(
        "pm25",
        recent_series=[10.3, 10.0, 12.3],
        model_path=pm25_model,
        scaler_path=pm25_scaler,
        config_path=pm25_config,
    )

    pm10_pred = predict_next(
        "pm10",
        recent_series=[25, 15, 15],
        model_path=pm10_model,
        scaler_path=pm10_scaler,
        config_path=pm10_config,
    )

    o3_pred = predict_next(
        "o3",
        recent_series=[0.020, 0.025, 0.018],
        model_path=o3_model,
        scaler_path=o3_scaler,
        config_path=o3_config,
    )

    so2_pred = predict_next(
        "so2",
        recent_series=[0.020, 0.025, 0.018],
        model_path=so2_model,
        scaler_path=so2_scaler,
        config_path=so2_config,
    )

    co_pred = predict_next(
        "co",
        recent_series=[0.2, 0.3, 0.25],
        model_path=co_model,
        scaler_path=co_scaler,
        config_path=co_config
    )   

    no2_pred = predict_next(
        "no2",
        recent_series=[0.2, 0.3, 0.25],
        model_path=no2_model,
        scaler_path=no2_scaler,
        config_path=no2_config
    )   

    sample = {"pm25": pm25_pred, "pm10": pm10_pred, "co": co_pred, "so2": so2_pred, "no2": no2_pred, "o3": o3_pred} 
    result = calculate_overall_aqi(sample)
    return result    

def calculate_single_aqi(concentration, pollutant):
    """計算單一污染物 AQI"""
    if pollutant not in aqi_breakpoints:
        raise ValueError(f"未知污染物: {pollutant}")

    for bp in aqi_breakpoints[pollutant]:
        if bp["lo"] <= concentration <= bp["hi"]:
            aqi = ((bp["aqiHi"] - bp["aqiLo"]) / (bp["hi"] - bp["lo"])) * (concentration - bp["lo"]) + bp["aqiLo"]
            return round(aqi)
    return None  # 超出範圍

def calculate_overall_aqi(values):
    """
    values = dict, e.g. {"pm25": 35.0, "pm10": 80, "co": 5.0, "so2": 20, "no2": 40}
    """
    individual_aqis = {}
    max_aqi, dominant = 0, None

    for pollutant, conc in values.items():
        if conc is not None:
            aqi = calculate_single_aqi(conc, pollutant)
            if aqi is not None:
                individual_aqis[pollutant] = aqi
                if aqi > max_aqi:
                    max_aqi, dominant = aqi, pollutant

    return {
        "AQI": max_aqi,
        "Dominant": dominant,
        "Detail": individual_aqis
    }

def fetchgas():
    vars_to_fetch = ["pm25", "pm10", "o3", "no2", "so2", "co"]
    results = {}
    for v in vars_to_fetch:
        results[v] = fetchsiglegas(v)   # 假設 fetchsiglegas 需要帶入 v
    return results

def fetchsiglegas(sensor_id):
    api_key = "41e704bb85ef8f83cf8a5723210056ed6bd5cdbc4c4bcb2e46df5f746fa3475f"
    url = f"https://api.openaq.org/v3/sensors/{sensor_id}/hours"
    print(url)
    headers = {}
    headers["X-API-Key"] = api_key
    resp = requests.get(url, headers=headers, timeout=30)

    # Error handling
    if resp.status_code != 200:
        print("Error:", resp.status_code, resp.text)
        return None
    
    # Get the last three hours' values
    data = resp.json()
    detail = data.get("results")
    last_three = detail[-3:]
    values = [item["value"] for item in last_three]
    return values

def fetch_vars_to_fetch(station_id):
    api_key = "41e704bb85ef8f83cf8a5723210056ed6bd5cdbc4c4bcb2e46df5f746fa3475f"
    url = f"https://api.openaq.org/v3/sensors/{sensor_id}/hours"


if __name__ == "__main__":
    print(fetchsiglegas(273))
    # print("AQI is", predict_sequence('test'))