# ml-api/app.py
"""
FastAPI Banking Customer Churn Prediction Service.
Integrates with SageMaker Runtime to query real-time endpoint 'bank-churn-endpoint'.
"""
import os
import json
import time
import boto3
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "bank-churn-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-2")

runtime = boto3.client("sagemaker-runtime", region_name=REGION)

app = FastAPI(
    title="Apex Bank Churn Intelligence API",
    description="Real-time inferencing connected to AWS SageMaker Endpoints"
)

templates = Jinja2Templates(directory="templates")

class CustomerProfile(BaseModel):
    CreditScore: float
    Geography: int
    Gender: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(customer: CustomerProfile):
    payload = customer.model_dump()
    
    t0 = time.time()
    try:
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(payload)
        )
        latency_ms = round((time.time() - t0) * 1000, 1)

        result_bytes = response["Body"].read().decode("utf-8")
        result = json.loads(result_bytes)
        result["latency_ms"] = latency_ms
        return result
    except Exception as e:
        latency_ms = round((time.time() - t0) * 1000, 1)
        return {
            "error": str(e),
            "churn_prediction": -1,
            "churn_probability": 0.0,
            "churn_probability_pct": "0.0%",
            "risk_tier": "Unavailable",
            "recommended_retention_action": f"Endpoint Error: {str(e)}",
            "latency_ms": latency_ms
        }
