from fastapi import FastAPI, Request, HTTPException
import pickle
import pandas as pd
from pydantic import BaseModel, Field
from catboost import CatBoostClassifier

from data_pipeline import preprocess_dataframe

app = FastAPI()

model = CatBoostClassifier()
model.load_model('model.cbm')

# Счетчик запросов
request_count = 0

# Модель для валидации входных данных
class PredictionInput(BaseModel):
    customerID: str
    gender: str = Field(..., pattern="^(Male|Female)$")
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str = Field(..., pattern="^(Yes|No)$")
    Dependents: str = Field(..., pattern="^(Yes|No)$")
    tenure: int = Field(..., ge=0, le=100)
    PhoneService: str = Field(..., pattern="^(Yes|No)$")
    MultipleLines: str = Field(..., pattern="^(Yes|No|No phone service)$")
    InternetService: str = Field(..., pattern="^(DSL|Fiber optic|No)$")
    OnlineSecurity: str = Field(..., pattern="^(Yes|No|No internet service)$")
    OnlineBackup: str = Field(..., pattern="^(Yes|No|No internet service)$")
    DeviceProtection: str = Field(..., pattern="^(Yes|No|No internet service)$")
    TechSupport: str = Field(..., pattern="^(Yes|No|No internet service)$")
    StreamingTV: str = Field(..., pattern="^(Yes|No|No internet service)$")
    StreamingMovies: str = Field(..., pattern="^(Yes|No|No internet service)$")
    Contract: str = Field(..., pattern="^(Month-to-month|One year|Two year)$")
    PaperlessBilling: str = Field(..., pattern="^(Yes|No)$")
    PaymentMethod: str = Field(..., pattern="^(Electronic check|Mailed check|Bank transfer \(automatic\)|Credit card \(automatic\))$")
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: str 

@app.get("/stats")
def stats():
    return {"request_count": request_count}

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/predict_model")
def predict_model(input_data: PredictionInput):
    global request_count
    request_count += 1

    # Создание DataFrame из данных
    new_data = pd.DataFrame({
        'customerID': [input_data.customerID],
        'gender': [input_data.gender],
        'SeniorCitizen': [input_data.SeniorCitizen],
        'Partner': [input_data.Partner],
        'Dependents': [input_data.Dependents],
        'tenure': [input_data.tenure],
        'PhoneService': [input_data.PhoneService],
        'MultipleLines': [input_data.MultipleLines],
        'InternetService': [input_data.InternetService],
        'OnlineSecurity': [input_data.OnlineSecurity],
        'OnlineBackup': [input_data.OnlineBackup],
        'DeviceProtection': [input_data.DeviceProtection],
        'TechSupport': [input_data.TechSupport],
        'StreamingTV': [input_data.StreamingTV],
        'StreamingMovies': [input_data.StreamingMovies],
        'Contract': [input_data.Contract],
        'PaperlessBilling': [input_data.PaperlessBilling],
        'PaymentMethod': [input_data.PaymentMethod],
        'MonthlyCharges': [input_data.MonthlyCharges],
        'TotalCharges': [input_data.TotalCharges]
    })

    X = preprocess_dataframe(new_data)

    # Предсказание
    predictions = model.predict(X)

    result = "Churn" if predictions[0] == 1 else "No Churn"

    return {"prediction": result}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)