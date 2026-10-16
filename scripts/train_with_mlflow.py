# scripts/train_with_mlflow.py
"""
Bank Customer Churn Model Training with SageMaker Managed MLflow Tracking.
- Connects directly to SageMaker Managed MLflow Tracking Server.
- Logs hyperparameters, evaluation metrics (PR-AUC, ROC-AUC, F1, Confusion Matrix).
- Logs SHAP explainability plots as visual artifacts.
- Automatically synchronizes with SageMaker Model Registry via --automatic-model-registration.
"""
import os
import sys
import json
import joblib
import boto3
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

# SageMaker Managed MLflow Tracking Server ARN
REGION = os.environ.get("AWS_REGION", "us-east-2")
TRACKING_SERVER_NAME = os.environ.get("TRACKING_SERVER_NAME", "bank-churn-mlflow")
ACCOUNT_ID = "227246780492"
DEFAULT_TRACKING_ARN = f"arn:aws:sagemaker:{REGION}:{ACCOUNT_ID}:mlflow-tracking-server/{TRACKING_SERVER_NAME}"

if "AWS_PROFILE" not in os.environ and "AWS_ACCESS_KEY_ID" not in os.environ:
    os.environ["AWS_PROFILE"] = "agent-toolkit"

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_ARN)
EXPERIMENT_NAME = "bank-customer-churn-experiment"

print("=" * 60)
print(f"TRAINING BANK CHURN MODEL WITH MANAGED SAGEMAKER MLFLOW")
print(f"Tracking Server: {TRACKING_URI}")
print(f"Experiment:      {EXPERIMENT_NAME}")
print("=" * 60)

# 1. Setup MLflow with SageMaker plugin
import mlflow
import mlflow.xgboost

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# 2. Load Dataset
data_path = "data/bank_churn.csv"
if not os.path.exists(data_path):
    data_path = "terraform/bank_churn.csv"

print(f"\n[1/5] Loading training data from: {data_path}")
df = pd.read_csv(data_path)
print(f"  Dataset shape: {df.shape} (Customers: {len(df)})")

feature_cols = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]
target_col = "Churn"

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Handle Class Imbalance (80% Non-Churn / 20% Churn)
num_neg = (y_train == 0).sum()
num_pos = (y_train == 1).sum()
scale_pos_weight = float(num_neg / num_pos)

params = {
    "n_estimators": 150,
    "max_depth": 4,
    "learning_rate": 0.08,
    "scale_pos_weight": round(scale_pos_weight, 2),
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "random_state": 42,
    "eval_metric": "aucpr"
}

# 3. Train with MLflow Tracking
print("\n[2/5] Training XGBoost Classifier and logging to MLflow...")
with mlflow.start_run(run_name="xgboost-churn-baseline") as run:
    run_id = run.info.run_id
    print(f"  MLflow Run ID: {run_id}")

    # Log Parameters
    mlflow.log_params(params)
    mlflow.log_param("dataset_size", len(df))
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    mlflow.log_param("positive_class_ratio", f"{round(num_pos / len(y_train) * 100, 1)}%")

    # Fit Model
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    # Predictions & Metrics
    print("\n[3/5] Evaluating Model & Logging Banking Metrics...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.50).astype(int)

    roc_auc = float(roc_auc_score(y_test, y_pred_proba))
    pr_auc = float(average_precision_score(y_test, y_pred_proba))
    f1 = float(f1_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))

    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  PR-AUC:    {pr_auc:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")

    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("pr_auc", pr_auc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # 4. SHAP Explainability & Visual Artifacts
    print("\n[4/5] Generating SHAP Explainability Plot for MLflow UI...")
    os.makedirs("artifacts/mlflow", exist_ok=True)
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        shap_path = "artifacts/mlflow/shap_summary.png"
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig(shap_path, bbox_inches="tight")
        plt.close()

        mlflow.log_artifact(shap_path, artifact_path="explainability")
        print("  SHAP summary plot successfully logged to MLflow artifacts!")
    except Exception as e:
        print(f"  [WARN] SHAP generation failed: {e}")

    # Log Evaluation Report JSON
    cm = confusion_matrix(y_test, y_pred).tolist()
    eval_report = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "confusion_matrix": {"TN": cm[0][0], "FP": cm[0][1], "FN": cm[1][0], "TP": cm[1][1]}
    }
    report_path = "artifacts/mlflow/evaluation.json"
    with open(report_path, "w") as f:
        json.dump(eval_report, f, indent=2)
    mlflow.log_artifact(report_path, artifact_path="metrics")

    # 5. Log & Register Model
    print("\n[5/5] Registering Model in MLflow (Syncs with SageMaker Model Registry)...")
    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path="model",
        registered_model_name="bank-customer-churn"
    )

print("\n" + "=" * 60)
print("TRAINING & MLFLOW LOGGING COMPLETE!")
print(f"Run ID: {run_id}")
print("=" * 60)
