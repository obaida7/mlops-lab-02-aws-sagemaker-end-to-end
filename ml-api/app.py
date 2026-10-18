# ml-api/app.py
"""
FastAPI Banking Customer Churn Prediction Service.
- Integrates with SageMaker Runtime to query real-time endpoint 'bank-churn-endpoint'.
- Connects to DynamoDB Online Feature Store ('bank_customer_features') for <10ms feature retrieval.
- Streams real-time inference data capture payloads to S3 ('datacapture/') for continuous drift auditing.
"""
import os
import sys
import json
import time
import uuid
import boto3
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "bank-churn-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-2")
S3_BUCKET = os.environ.get("S3_BUCKET", "sagemaker-us-east-2-227246780492-bank-churn")
DYNAMODB_TABLE = "bank_customer_features"

if "AWS_PROFILE" not in os.environ and "AWS_ACCESS_KEY_ID" not in os.environ:
    os.environ["AWS_PROFILE"] = "agent-toolkit"

boto_session = boto3.Session(region_name=REGION)
runtime = boto_session.client("sagemaker-runtime")
s3 = boto_session.client("s3")
dynamodb = boto_session.resource("dynamodb")
feature_table = dynamodb.Table(DYNAMODB_TABLE)

app = FastAPI(
    title="Apex Bank Churn Intelligence API",
    description="Real-time inferencing connected to AWS SageMaker Endpoints, DynamoDB Feature Store, and S3 Data Capture"
)

templates = Jinja2Templates(directory="templates")

class CustomerProfile(BaseModel):
    CreditScore: float = 650.0
    Geography: int = 0
    Gender: int = 1
    Age: int = 40
    Tenure: int = 3
    Balance: float = 0.0
    NumOfProducts: int = 1
    HasCrCard: int = 1
    IsActiveMember: int = 1           # Default fallback prevents missing field crash
    EstimatedSalary: float = 50000.0

    @field_validator("Balance", mode="before")
    @classmethod
    def clean_currency(cls, v):
        """Strips '$' and commas from currency strings like '$124,500.00'."""
        if isinstance(v, str):
            clean_str = v.replace("$", "").replace(",", "").strip()
            return float(clean_str)
        return float(v)

    @field_validator("CreditScore", "Age", "Tenure", mode="before")
    @classmethod
    def coerce_numeric(cls, v):
        """Coerces numeric strings like '45' to numbers."""
        return float(v) if v is not None else 0.0

def save_data_capture_to_s3(payload: dict, result: dict, latency_ms: float):
    """Asynchronously logs production inference payloads and model predictions to S3 Data Capture."""
    try:
        req_id = str(uuid.uuid4())[:8]
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        date_prefix = time.strftime("year=%Y/month=%m/day=%d")
        
        capture_record = {
            "request_id": req_id,
            "timestamp": timestamp,
            "endpoint_name": ENDPOINT_NAME,
            "features": payload,
            "prediction": result,
            "latency_ms": latency_ms
        }
        
        s3_key = f"datacapture/{date_prefix}/inference_{timestamp}_{req_id}.json"
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=json.dumps(capture_record, indent=2),
            ContentType="application/json"
        )
    except Exception as e:
        print(f"[WARN] DataCapture S3 logging error: {e}", file=sys.stderr)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/features/{customer_id}")
def get_customer_from_feature_store(customer_id: int):
    """Retrieves online features for a customer directly from the DynamoDB Feature Store."""
    try:
        resp = feature_table.get_item(Key={"CustomerId": int(customer_id)})
        if "Item" in resp:
            item = resp["Item"]
            return {
                "found": True,
                "customer_id": customer_id,
                "features": {
                    "CreditScore": int(item["CreditScore"]),
                    "Geography": int(item["Geography"]),
                    "Gender": int(item["Gender"]),
                    "Age": int(item["Age"]),
                    "Tenure": int(item["Tenure"]),
                    "Balance": float(item["Balance"]),
                    "NumOfProducts": int(item["NumOfProducts"]),
                    "HasCrCard": int(item["HasCrCard"]),
                    "IsActiveMember": int(item["IsActiveMember"]),
                    "EstimatedSalary": float(item["EstimatedSalary"])
                }
            }
        return {"found": False, "message": f"Customer {customer_id} not in online feature store."}
    except Exception as e:
        return {"found": False, "error": str(e)}

@app.post("/predict")
def predict(customer: CustomerProfile, background_tasks: BackgroundTasks):
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

        # Trigger S3 Data Capture in the background
        background_tasks.add_task(save_data_capture_to_s3, payload, result, latency_ms)

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
