# scripts/train_and_evaluate.py
"""
Training and Evaluation script for SageMaker XGBoost Pipeline.
- Trains an XGBoost classifier on banking customer churn.
- Optimizes for PR-AUC and ROC-AUC on imbalanced financial data.
- Generates SHAP explainability artifacts to comply with SR 11-7 model governance.
- Packages model weights and inference.py into /opt/ml/model.
"""
import os
import sys
import json
import shutil
import argparse
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=150)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.08)
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    return parser.parse_args()


def load_dataset(train_dir):
    csv_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if "churn" in f.lower() and f.endswith(".csv")]
    if not csv_files:
        csv_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in training directory: {train_dir}")
    
    print(f"Loading training data from: {csv_files[0]}")
    df = pd.read_csv(csv_files[0])
    print(f"Dataset Shape: {df.shape}")
    return df


def main():
    args = parse_args()
    print("=" * 60)
    print("STARTING BANK CHURN MODEL TRAINING")
    print(f"Hyperparameters: n_estimators={args.n_estimators}, max_depth={args.max_depth}, lr={args.learning_rate}")
    print("=" * 60)

    # 1. Load data
    df = load_dataset(args.train)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    print(f"Class Distribution: Retained (0) = {(y == 0).sum()}, Churned (1) = {(y == 1).sum()}")
    churn_rate = y.mean() * 100
    print(f"Overall Churn Rate: {churn_rate:.2f}%")

    # 2. Train/Test Split (Stratified to preserve class imbalance ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Calculate scale_pos_weight for handling banking churn class imbalance
    scale_pos_weight = float((len(y_train) - sum(y_train)) / sum(y_train))
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    # 3. Train XGBoost Classifier
    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42
    )

    print("Fitting model...")
    model.fit(X_train, y_train)

    # 4. Evaluation
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    roc_auc = float(roc_auc_score(y_test, test_probs))
    pr_auc = float(average_precision_score(y_test, test_probs))
    f1 = float(f1_score(y_test, test_preds))
    precision = float(precision_score(y_test, test_preds))
    recall = float(recall_score(y_test, test_preds))
    cm = confusion_matrix(y_test, test_preds).tolist()

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS (TEST SET)")
    print("=" * 60)
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}  <-- Primary Banking Retention Metric")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Confusion Matrix (TN, FP / FN, TP):\n{cm}")
    print("\nFull Classification Report:")
    print(classification_report(y_test, test_preds, target_names=["Retained", "Churned"]))

    feature_importances = {
        col: float(imp) for col, imp in zip(X.columns, model.feature_importances_)
    }
    sorted_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))
    print("Feature Importances:\n", json.dumps(sorted_importances, indent=2))

    # 5. SHAP Model Explainability (Banking Compliance: SR 11-7)
    try:
        import shap
        import matplotlib.pyplot as plt

        print("\nCalculating SHAP values for model explainability...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        os.makedirs(args.model_dir, exist_ok=True)
        shap_plot_path = os.path.join(args.model_dir, "shap_summary.png")
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig(shap_plot_path, bbox_inches="tight")
        plt.close()
        print(f"Saved SHAP summary plot to: {shap_plot_path}")
    except Exception as e:
        print(f"[WARN] SHAP generation skipped: {e}")

    # 6. Save Model Artifacts & Evaluation Report
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True)

    # Save model weights
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    model.save_model(os.path.join(args.model_dir, "model.xgb"))

    # Copy inference.py into code/ subdirectory inside model directory for SageMaker serving
    code_dir = os.path.join(args.model_dir, "code")
    os.makedirs(code_dir, exist_ok=True)
    
    # Check source locations for inference.py
    src_inference = os.path.join(os.path.dirname(__file__), "inference.py")
    if os.path.exists(src_inference):
        shutil.copyfile(src_inference, os.path.join(code_dir, "inference.py"))
    else:
        # Fallback if executed from root
        shutil.copyfile("scripts/inference.py", os.path.join(code_dir, "inference.py"))

    with open(os.path.join(code_dir, "requirements.txt"), "w") as f:
        f.write("xgboost>=1.7.0\njoblib>=1.2.0\npandas>=1.5.0\nscikit-learn>=1.2.0\n")

    # Prepare standard SageMaker evaluation report JSON
    eval_report = {
        "binary_classification_metrics": {
            "roc_auc": {"value": roc_auc},
            "pr_auc": {"value": pr_auc},
            "f1": {"value": f1},
            "precision": {"value": precision},
            "recall": {"value": recall},
            "confusion_matrix": {
                "0": {"0": cm[0][0], "1": cm[0][1]},
                "1": {"0": cm[1][0], "1": cm[1][1]}
            }
        },
        "feature_importances": sorted_importances,
        "sample_size": len(df),
        "test_size": len(X_test)
    }

    report_path = os.path.join(args.model_dir, "evaluation.json")
    with open(report_path, "w") as f:
        json.dump(eval_report, f, indent=2)

    with open(os.path.join(args.output_data_dir, "evaluation.json"), "w") as f:
        json.dump(eval_report, f, indent=2)

    print(f"Saved model and evaluation report to: {args.model_dir}")
    print("TRAINING AND EVALUATION COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
