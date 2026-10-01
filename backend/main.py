from fastapi import FastAPI
from pydantic import BaseModel

from backend.predictor import predict


app = FastAPI(
    title="Heart Risk Prediction API",
    version="1.0.0"
)

# input schema matching training features
class HeartRiskInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int
    
@app.get("/health")
def health_check():
    return {"status": "ok"}


# Heart disease prediction endpoint
@app.post("/predict-heart-risk")
def predict_heart_disease(input_data: HeartRiskInput):
    
    
    input_data = input_data.model_dump()
    result = predict(input_data=input_data)
    return {
        "prediction": result["prediction"],
        "probability": result["probability"],
        "diagnosis": (
            "High Risk of Heart Disease detected"
            if result["prediction"] == 1
            else "Low Risk of Heart Disease detected"
        )
    }
    
    
