# scripts/inference.py
"""
SageMaker Inference Serving Script for Bank Customer Churn Prediction.
Complies with SageMaker PyTorch / Scikit-Learn / XGBoost serving contract:
- model_fn: Load model into memory
- input_fn: Deserialize incoming request
- predict_fn: Execute inference and risk scoring
- output_fn: Serialize response
"""
import os
import json
import joblib
import pandas as pd
import numpy as np

FEATURE_NAMES = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]

def model_fn(model_dir):
    """Loads the trained model from the model directory."""
    xgb_path = os.path.join(model_dir, "model.xgb")
    if os.path.exists(xgb_path):
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(xgb_path)
        return model

    model_path = os.path.join(model_dir, "model.joblib")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model

    raise FileNotFoundError(f"No valid model found in {model_dir}")


def input_fn(request_body, content_type):
    """Deserializes request payload into structured feature format."""
    if content_type == "application/json":
        data = json.loads(request_body)
        if isinstance(data, dict):
            # Single customer prediction
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            # Batch of customers
            df = pd.DataFrame(data)
        else:
            raise ValueError("JSON request must be a dictionary or list of dictionaries.")
        
        # Ensure correct column ordering
        return df[FEATURE_NAMES]

    elif content_type == "text/csv":
        lines = [line.strip() for line in request_body.strip().split("\n") if line.strip()]
        if not lines:
            raise ValueError("Empty CSV payload")

        # Check if first line contains headers
        first_line_parts = lines[0].split(",")
        has_headers = any(h.strip() in FEATURE_NAMES for h in first_line_parts)

        if has_headers:
            from io import StringIO
            df = pd.read_csv(StringIO(request_body))
            return df[FEATURE_NAMES]
        else:
            rows = [[float(v) for v in line.split(",")] for line in lines]
            df = pd.DataFrame(rows, columns=FEATURE_NAMES)
            return df

    else:
        raise ValueError(f"Unsupported content type: {content_type}. Expected application/json or text/csv.")


def predict_fn(input_data, model):
    """Generates churn prediction, calibrated probabilities, and banking risk categorization."""
    probs = model.predict_proba(input_data)[:, 1]

    # Check for calibrated fairness thresholds
    th_female = 0.50
    th_male = 0.50
    for candidate_path in [
        "artifacts/compliance/fairness_thresholds.json",
        "fairness_thresholds.json",
        os.path.join(os.path.dirname(__file__), "../artifacts/compliance/fairness_thresholds.json")
    ]:
        if os.path.exists(candidate_path):
            try:
                with open(candidate_path) as f:
                    cfg = json.load(f)
                th_female = cfg.get("threshold_female", 0.50)
                th_male = cfg.get("threshold_male", 0.50)
                break
            except Exception:
                pass

    if "Gender" in input_data.columns:
        thresholds = np.where(input_data["Gender"].values == 0, th_female, th_male)
        preds = (probs >= thresholds).astype(int)
    else:
        preds = (probs >= 0.5).astype(int)

    results = []
    for prob, pred in zip(probs, preds):
        prob_pct = round(float(prob) * 100, 2)
        
        # Financial services risk tiering
        if prob >= 0.60:
            risk_tier = "High"
            action = "Immediate proactive outreach by Senior Relationship Manager. Offer preferential deposit rate or fee-free banking suite."
        elif prob >= 0.35:
            risk_tier = "Medium"
            action = "Targeted digital engagement: Cross-sell secondary account with welcome rewards or cashback incentives."
        else:
            risk_tier = "Low"
            action = "Maintain standard relationship. Customer is highly engaged and loyal."

        results.append({
            "churn_prediction": int(pred),
            "churn_probability": round(float(prob), 4),
            "churn_probability_pct": f"{prob_pct}%",
            "risk_tier": risk_tier,
            "recommended_retention_action": action
        })

    return results


def output_fn(prediction, accept):
    """Serializes prediction results into requested accept format."""
    if accept in ["application/json", "*/*", None]:
        # Return single object if 1 customer, list if batch
        if len(prediction) == 1:
            return json.dumps(prediction[0]), "application/json"
        return json.dumps(prediction), "application/json"

    elif accept == "text/csv":
        # Returns: churn_prediction,churn_probability,risk_tier
        csv_lines = [
            f"{p['churn_prediction']},{p['churn_probability']},{p['risk_tier']}"
            for p in prediction
        ]
        return "\n".join(csv_lines), "text/csv"

    else:
        raise ValueError(f"Unsupported accept type: {accept}")
