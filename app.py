from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")

# Create FastAPI app
app = FastAPI(title="Diabetes Prediction API")

# Input schema
class PatientData(BaseModel):
    age: int
    gender: int
    bmi: float
    hypertension: int
    heart_disease: int
    smoking_history: int
    physical_activity_level: int
    alcohol_intake: int
    diet_score: int
    sleep_hours: float
    stress_level: int
    glucose_level: float
    blood_pressure_systolic: float
    blood_pressure_diastolic: float
    insulin_level: float
    cholesterol: float
    triglycerides: float
    waist_circumference: float
    family_history: int
    exercise_frequency: int
    fasting_blood_sugar: float
    hba1c: float
    skin_thickness: float
    pregnancies: int
    income_level: int
    region: int


# Home route
@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running"}

# Prediction route
@app.post("/predict")
def predict(data: PatientData):

    # Convert input to dataframe
    input_data = pd.DataFrame([data.dict()])

    # Predict
    prediction = model.predict(input_data)[0]

    # Probability
    probability = model.predict_proba(input_data)[0][1]

    result = "Diabetic" if prediction == 1 else "Not Diabetic"

    return {
        "prediction": result,
        "probability": round(float(probability), 2)
    }