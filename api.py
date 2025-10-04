from fastapi import FastAPI
from pydantic import BaseModel
import inference

app = FastAPI(title="Air Quality Prediction API", version="1.0")

# 輸入資料格式
class InferenceRequest(BaseModel):
    pm25: list[float]
    pm10: list[float]
    o3: list[float]
    no2: list[float]
    so2: list[float]
    co: list[float]

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/predict_aqi")
def predict_aqi():
    return inference.predict_sequence('test')
