from fastapi import FastAPI, Query
from pydantic import BaseModel
import inference

app = FastAPI(title="Air Quality Prediction API", version="1.0")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/predict_aqi")
def predict_aqi(station_id: int = Query(..., description="Station ID to predict AQI")):
    print(f"Received request for station_id: {station_id}")
    result = inference.predict_sequence(station_id)
    return {"station_id": station_id, "result": result}
