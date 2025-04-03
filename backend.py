from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

# Load the trained models
model_with_lag = joblib.load("./Model/prediction_model.joblib")
model_without_lag = joblib.load("./Model/prediction_model_without_lag.joblib")

# Initialize FastAPI app
app = FastAPI()

# Define request model for model with lag features
class PredictionRequestWithLag(BaseModel):
    pickup_cluster: int
    mins: int
    hour: int
    month: int
    quarter: int
    dayofweek: int
    lag_1: float
    lag_2: float
    lag_3: float
    rolling_mean: float

# Define request model for model without lag features
class PredictionRequestWithoutLag(BaseModel):
    pickup_cluster: int
    mins: int
    hour: int
    month: int
    quarter: int
    dayofweek: int

# Prediction endpoint for model with lag
@app.post("/predict_with_lag")
def predict_with_lag(data: PredictionRequestWithLag):
    features = np.array([
        data.pickup_cluster, data.mins, data.hour, data.month,
        data.quarter, data.dayofweek, data.lag_1,
        data.lag_2, data.lag_3, data.rolling_mean
    ]).reshape(1, -1)
    
    prediction = model_with_lag.predict(features)
    return {"prediction": prediction.tolist()}

# Prediction endpoint for model without lag
@app.post("/predict_without_lag")
def predict_without_lag(data: PredictionRequestWithoutLag):
    features = np.array([
        data.pickup_cluster, data.mins, data.hour, data.month,
        data.quarter, data.dayofweek
    ]).reshape(1, -1)
    
    prediction = model_without_lag.predict(features)
    return {"prediction": prediction.tolist()}

# Root endpoint
@app.get("/")
def home():
    return {"message": "Prediction API is running!"}
