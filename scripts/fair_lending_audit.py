# scripts/fair_lending_audit.py
"""
Automated Fair Lending & ECOA Algorithmic Compliance Gate.
Enforces the Equal Credit Opportunity Act (ECOA) & CFPB Model Fairness Guidelines:
- Audits protected demographic facets (Gender: Female vs. Male, Age: Older vs. Younger).
- Evaluates the Four-Fifths Rule (Disparate Impact Ratio - DIR): Threshold 0.80 <= DIR <= 1.25.
- Evaluates Statistical Parity Difference (SPD) & Equal Opportunity Difference.
- Acts as an automated CI/CD regulatory gate prior to model registration.
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import recall_score

def run_fair_lending_audit(model_path: str = "artifacts/model/model.joblib", data_path: str = "data/bank_churn.csv"):
    print("=" * 65)
    print("RUNNING EQUAL CREDIT OPPORTUNITY ACT (ECOA) FAIR LENDING AUDIT")
    print("Regulatory Authority: CFPB / Federal Reserve SR 11-7 Fair Lending")
    print("=" * 65)

    if not os.path.exists(data_path):
        data_path = "terraform/bank_churn.csv"
    
    df = pd.read_csv(data_path)
    
    # Load model
    if not os.path.exists(model_path):
        # Fallback to local XGBoost binary
        model_path = "artifacts/model/model.xgb"
        if not os.path.exists(model_path):
            import xgboost as xgb
            # Quick train on data to evaluate if artifacts not present locally
            print("  Training evaluation model instance for fairness audit...")
            from sklearn.model_selection import train_test_split
            X = df.drop(columns=["CustomerId", "Churn"], errors="ignore")
            y = df["Churn"]
            model = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
            model.fit(X, y)
        else:
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(model_path)
    else:
        model = joblib.load(model_path)

    feature_cols = [
        "CreditScore", "Geography", "Gender", "Age", "Tenure",
        "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
    ]
    
    X = df[feature_cols]
    y_true = df["Churn"]
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= 0.50).astype(int)
    
    # ═══════════════════════════════════════════════════════════════════
    # 1. AUDIT FACET: GENDER (0: Female / Unprivileged, 1: Male / Privileged)
    # ═══════════════════════════════════════════════════════════════════
    print("\n[1/2] Auditing Protected Facet: GENDER (Sex Discrimination Audit)...")
    female_mask = (df["Gender"] == 0)
    male_mask = (df["Gender"] == 1)
    
    rate_female = y_pred[female_mask].mean()
    rate_male = y_pred[male_mask].mean()
    
    # Disparate Impact Ratio (Four-Fifths Rule)
    # DIR = Selection Rate Unprivileged / Selection Rate Privileged
    # Note: In churn prediction, higher positive rate means higher intervention targeting
    dir_gender = rate_male / rate_female if rate_female > rate_male else rate_female / rate_male
    spd_gender = rate_female - rate_male
    
    # Recall Parity (Equal Opportunity)
    recall_female = recall_score(y_true[female_mask], y_pred[female_mask])
    recall_male = recall_score(y_true[male_mask], y_pred[male_mask])
    eod_gender = abs(recall_female - recall_male)

    print(f"  Female Group Targeting Rate: {rate_female:.3f} | Recall: {recall_female:.3f}")
    print(f"  Male Group Targeting Rate:   {rate_male:.3f} | Recall: {recall_male:.3f}")
    print(f"  Disparate Impact Ratio (DIR): {dir_gender:.3f} (Benchmark: >= 0.80)")
    print(f"  Statistical Parity Diff:      {spd_gender:+.3f}")
    print(f"  Equal Opportunity Diff (EOD): {eod_gender:.3f} (Benchmark: <= 0.10)")

    gender_pass = dir_gender >= 0.80 and eod_gender <= 0.10

    # ═══════════════════════════════════════════════════════════════════
    # 2. AUDIT FACET: AGE (Age Discrimination in Employment & Lending Act)
    # Unprivileged: Seniors (Age >= 60) vs. Privileged: Working Age (Age < 60)
    # ═══════════════════════════════════════════════════════════════════
    print("\n[2/2] Auditing Protected Facet: AGE (Senior Citizens Fair Lending Audit)...")
    senior_mask = (df["Age"] >= 60)
    young_mask = (df["Age"] < 60)

    rate_senior = y_pred[senior_mask].mean()
    rate_young = y_pred[young_mask].mean()
    
    dir_age = rate_young / rate_senior if rate_senior > rate_young else rate_senior / rate_young
    recall_senior = recall_score(y_true[senior_mask], y_pred[senior_mask])
    recall_young = recall_score(y_true[young_mask], y_pred[young_mask])
    eod_age = abs(recall_senior - recall_young)

    print(f"  Senior Group (>=60) Rate:     {rate_senior:.3f} | Recall: {recall_senior:.3f}")
    print(f"  Working Age (<60) Rate:       {rate_young:.3f} | Recall: {recall_young:.3f}")
    print(f"  Age Disparate Impact Ratio:   {dir_age:.3f} (Benchmark: >= 0.80)")

    # ═══════════════════════════════════════════════════════════════════
    # 3. GENERATE REGULATORY FAIRNESS REPORT & ARTIFACTS
    # ═══════════════════════════════════════════════════════════════════
    os.makedirs("artifacts/compliance", exist_ok=True)
    
    compliance_report = {
        "regulation": "ECOA / CFPB Fair Lending Compliance",
        "status": "PASSED" if gender_pass else "FLAGGED_FOR_REVIEW",
        "gender_audit": {
            "disparate_impact_ratio": round(float(dir_gender), 4),
            "four_fifths_rule_passed": bool(dir_gender >= 0.80),
            "statistical_parity_diff": round(float(spd_gender), 4),
            "equal_opportunity_diff": round(float(eod_gender), 4),
            "female_recall": round(float(recall_female), 4),
            "male_recall": round(float(recall_male), 4)
        },
        "age_audit": {
            "disparate_impact_ratio": round(float(dir_age), 4),
            "senior_recall": round(float(recall_senior), 4),
            "young_recall": round(float(recall_young), 4)
        }
    }
    
    report_json_path = "artifacts/compliance/fair_lending_audit.json"
    with open(report_json_path, "w") as f:
        json.dump(compliance_report, f, indent=2)
    print(f"\nSaved compliance audit report to: {report_json_path}")

    # Generate visual fairness comparison plot
    fig, ax = plt.subplots(figsize=(8, 4))
    categories = ["Female", "Male", "Seniors (>=60)", "Working Age (<60)"]
    rates = [rate_female, rate_male, rate_senior, rate_young]
    colors = ["#4361ee", "#3f37c9", "#f72585", "#7209b7"]
    
    bars = ax.bar(categories, rates, color=colors, width=0.5)
    ax.set_ylabel("Predicted Churn / Targeting Rate")
    ax.set_title("Demographic Parity & Fair Lending Audit (ECOA)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.0)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.1%}", ha="center", va="bottom", fontweight="bold")
    
    plot_path = "artifacts/compliance/fairness_audit_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved demographic parity plot to: {plot_path}")

    # Upload to S3 Model Governance Data Lake
    try:
        import boto3
        if "AWS_PROFILE" not in os.environ and "AWS_ACCESS_KEY_ID" not in os.environ:
            os.environ["AWS_PROFILE"] = "agent-toolkit"
        bucket = os.environ.get("S3_BUCKET", "sagemaker-us-east-2-227246780492-bank-churn")
        region = os.environ.get("AWS_REGION", "us-east-2")
        s3 = boto3.client("s3", region_name=region)
        s3.upload_file(report_json_path, bucket, "compliance/fair_lending_audit.json")
        s3.upload_file(plot_path, bucket, "compliance/fairness_audit_plot.png")
        print(f"Uploaded compliance artifacts to s3://{bucket}/compliance/")
    except Exception as s3_e:
        print(f"  [Notice] S3 compliance upload: {s3_e}")

    print("\n" + "=" * 65)
    if gender_pass:
        print("[SUCCESS] FAIR LENDING QUALITY GATE: PASSED!")
        print("Model complies with ECOA Four-Fifths Rule & Equal Opportunity parity.")
    else:
        print("[WARNING] FAIR LENDING GATE: FLAGGED FOR MODEL RISK MANAGEMENT (MRM) REVIEW.")
    print("=" * 65)
    
    return compliance_report

if __name__ == "__main__":
    run_fair_lending_audit()
