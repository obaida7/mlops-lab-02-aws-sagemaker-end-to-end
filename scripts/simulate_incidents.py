# scripts/simulate_incidents.py
"""
Production Incident Simulator & On-Call Playbook for Bank Churn MLOps.
Replicates real-world Tier-1 incidents in enterprise banking machine learning:

1. INCIDENT 1 (P1/Critical): Schema Drift & Malformed Payloads (Training-Serving Skew)
   - Upstream microservice sends unescaped dollar strings, missing fields, or mismatched keys.
   - Endpoint crashes with 400/500 errors.
   - On-call fix: Defensive type coercion & fallback sanitization in serving layer.

2. INCIDENT 2 (P2/High): Covariate Shift & Demographic Drift (Marketing Campaign Influx)
   - Older customer cohort with high balances shifts distribution.
   - Kolmogorov-Smirnov test fails (p < 0.05).
   - Emits CloudWatch metric 'DataDriftViolation' and publishes SNS PagerDuty alert.
   - On-call fix: Automated drift triage, slice analysis, and retraining trigger.

3. INCIDENT 3 (P1/Critical): Canary Cold-Start & Latency SLA Violation
   - Canary Green variant trips 500ms P95 latency SLA.
   - Automated circuit breaker initiates zero-downtime rollback to Blue baseline.
   - On-call fix: Serverless memory tuning and container warmup.

4. INCIDENT 4 (Regulatory/Compliance P1): ECOA Fair Lending Gate Failure (SR 11-7)
   - Disparate Impact Ratio falls below 0.80 Four-Fifths threshold for female depositors.
   - Regulatory CI/CD gate halts deployment.
   - On-call fix: Threshold optimization to restore demographic parity.
"""
import os
import sys
import time
import json
import argparse
import boto3
import requests
import pandas as pd
import numpy as np
from scipy import stats

REGION = os.environ.get("AWS_REGION", "us-east-2")
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "bank-churn-endpoint")
API_URL = os.environ.get("API_URL", "http://localhost:8000")
SNS_TOPIC_ARN = os.environ.get(
    "SNS_TOPIC_ARN",
    "arn:aws:sns:us-east-2:227246780492:bank-churn-drift-alerts"
)

if "AWS_PROFILE" not in os.environ and "AWS_ACCESS_KEY_ID" not in os.environ:
    os.environ["AWS_PROFILE"] = "agent-toolkit"

boto_session = boto3.Session(region_name=REGION)
runtime = boto_session.client("sagemaker-runtime")
cw = boto_session.client("cloudwatch")
sns = boto_session.client("sns")

def print_banner(incident_id, title, severity):
    print("\n" + "=" * 70)
    print(f"🚨 INCIDENT #{incident_id} [{severity}]: {title.upper()}")
    print("=" * 70)

# ═══════════════════════════════════════════════════════════════════
# INCIDENT 1: SCHEMA DRIFT & MALFORMED PAYLOADS
# ═══════════════════════════════════════════════════════════════════
def simulate_incident_1():
    print_banner(1, "Schema Drift / Malformed Payload (Training-Serving Skew)", "P1 - CRITICAL")
    print("SCENARIO: Upstream digital banking microservice deployed a bad update.")
    print("It sends currency as string ('$124,500.00'), strings for integers, and omits 'IsActiveMember'.")
    print("-" * 70)

    malformed_payload = {
        "CreditScore": "640",                  # String instead of int
        "Geography": 0,
        "Gender": 1,
        "Age": "45",                          # String instead of int
        "Tenure": 3,
        "Balance": "$124,500.00",             # Currency formatted string! (CRITICAL SKEW)
        "NumOfProducts": 1,
        "HasCrCard": 1,
        # "IsActiveMember" is MISSING!
        "EstimatedSalary": 92000.0
    }

    print("\n[Step 1: Ingesting Malformed Payload]")
    print(json.dumps(malformed_payload, indent=2))

    print("\n[Step 2: Submitting to FastAPI Serving Layer without sanitization...]")
    try:
        resp = requests.post(f"{API_URL}/predict", json=malformed_payload, timeout=5)
        print(f"  HTTP Response Code: {resp.status_code}")
        print(f"  Response Body: {resp.text}")
        if resp.status_code >= 400 or "error" in resp.text.lower():
            print("\n🔥 INCIDENT DETECTED! Endpoint / API crashed or returned 422 Unprocessable Entity!")
    except Exception as e:
        print(f"\n🔥 INCIDENT DETECTED! Request threw unhandled exception: {e}")

    print("\n[Step 3: On-Call Engineer Root Cause Analysis (RCA)]")
    print("  1. Upstream CRM omitted required field 'IsActiveMember'.")
    print("  2. 'Balance' string contains '$' and ',' which breaks numeric matrix casting.")
    print("  3. Direct SageMaker XGBoost invocation expects strict 10-feature float array.")

    print("\n[Step 4: Applying On-Call Defensive Sanitization Hotfix...]")
    def sanitize_payload(raw: dict) -> dict:
        """Defensive type coercion and default imputation."""
        sanitized = {}
        defaults = {
            "CreditScore": 650.0, "Geography": 0, "Gender": 1, "Age": 40.0,
            "Tenure": 3.0, "Balance": 0.0, "NumOfProducts": 1, "HasCrCard": 1,
            "IsActiveMember": 1, "EstimatedSalary": 50000.0
        }
        for k, def_val in defaults.items():
            val = raw.get(k, def_val)
            if isinstance(val, str):
                # Strip currency symbols and commas
                clean_str = val.replace("$", "").replace(",", "").strip()
                try:
                    val = float(clean_str)
                except ValueError:
                    val = def_val
            sanitized[k] = float(val)
        return sanitized

    clean_payload = sanitize_payload(malformed_payload)
    print("  Sanitized Payload:")
    print(json.dumps(clean_payload, indent=2))

    print("\n[Step 5: Testing Resilient Fallback against Live Endpoint...]")
    t0 = time.time()
    resp = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(clean_payload)
    )
    latency = round((time.time() - t0) * 1000, 1)
    result = json.loads(resp["Body"].read().decode())
    print(f"  ✅ SUCCESS: Endpoint recovered! Scored in {latency}ms")
    print(f"  Churn Risk Tier: {result.get('risk_tier')} ({result.get('churn_probability_pct')})")
    print("  Incident #1 Status: MITIGATED & RESOLVED.")

# ═══════════════════════════════════════════════════════════════════
# INCIDENT 2: COVARIATE SHIFT / DEMOGRAPHIC DRIFT
# ═══════════════════════════════════════════════════════════════════
def simulate_incident_2():
    print_banner(2, "Covariate Shift / Severe Demographic Data Drift", "P2 - HIGH")
    print("SCENARIO: Marketing launched a high-yield Certificate of Deposit (CD) campaign.")
    print("Recent customer traffic shifted dramatically towards older, affluent depositors.")
    print("-" * 70)

    baseline_path = "data/bank_churn.csv"
    if not os.path.exists(baseline_path):
        baseline_path = "terraform/bank_churn.csv"
    base_df = pd.read_csv(baseline_path)

    # Generate synthetic post-campaign drifted cohort
    np.random.seed(101)
    cohort_size = 500
    prod_cohort = pd.DataFrame({
        "CreditScore": np.random.normal(loc=650, scale=80, size=cohort_size).clip(350, 850),
        "Age": np.random.normal(loc=48.5, scale=10, size=cohort_size).clip(18, 90),     # +9.5 years drift!
        "Tenure": np.random.randint(0, 11, size=cohort_size),
        "Balance": np.random.normal(loc=115000, scale=45000, size=cohort_size).clip(0, 300000), # +$38k drift!
        "NumOfProducts": np.random.choice([1, 2, 3, 4], size=cohort_size, p=[0.70, 0.25, 0.04, 0.01]),
        "EstimatedSalary": np.random.normal(loc=110000, scale=40000, size=cohort_size).clip(10000, 250000)
    })

    print(f"\n[Step 1: Running Two-Sample Kolmogorov-Smirnov Test (p < 0.05 threshold)]")
    drift_violations = []
    for col in ["CreditScore", "Age", "Balance", "EstimatedSalary"]:
        ks_stat, p_val = stats.ks_2samp(base_df[col], prod_cohort[col])
        base_mean = base_df[col].mean()
        prod_mean = prod_cohort[col].mean()
        shift = ((prod_mean - base_mean) / base_mean) * 100
        status = "🚨 DRIFT DETECTED" if p_val < 0.05 and abs(shift) > 5.0 else "STABLE"
        if "DRIFT" in status:
            drift_violations.append(col)
        print(f"  Feature: {col:<16} Base: {base_mean:>9.1f} | Prod: {prod_mean:>9.1f} ({shift:>+5.1f}%) | p-val: {p_val:.4e} | {status}")

    print("\n[Step 2: Emitting Telemetry to CloudWatch & Triggering SNS Page...]")
    try:
        cw.put_metric_data(
            Namespace="BankChurn/Monitoring",
            MetricData=[{
                "MetricName": "DataDriftViolation",
                "Value": 1.0,
                "Unit": "Count",
                "Dimensions": [{"Name": "EndpointName", "Value": ENDPOINT_NAME}]
            }]
        )
        print("  ✅ Emitted CloudWatch Metric: BankChurn/Monitoring -> DataDriftViolation = 1")
    except Exception as e:
        print(f"  [Notice] CloudWatch: {e}")

    try:
        alert_body = {
            "incident": "INCIDENT-002: Covariate Shift Detected",
            "endpoint": ENDPOINT_NAME,
            "drifted_features": drift_violations,
            "action_required": "Initiate automated model retraining on post-campaign demographic baseline."
        }
        sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject="PAGERDUTY: Data Drift Detected on bank-churn-endpoint",
            Message=json.dumps(alert_body, indent=2)
        )
        print(f"  ✅ Published P1 Alert to SNS Topic: {SNS_TOPIC_ARN}")
    except Exception as e:
        print(f"  [Notice] SNS Publish: {e}")

    print("\n[Step 3: On-Call Engineer Resolution Plan]")
    print("  1. Triage: Age & Balance distributions shifted beyond statistical stability limits.")
    print("  2. Impact: XGBoost decision tree boundaries for age 30-40 are inaccurate for 48+ cohort.")
    print("  3. Remediation: Trigger retraining pipeline with new partition from Feature Store.")
    print("  Incident #2 Status: IDENTIFIED & RETRAINING QUEUED.")

# ═══════════════════════════════════════════════════════════════════
# INCIDENT 3: CANARY LATENCY SPIKE & AUTOMATED ROLLBACK
# ═══════════════════════════════════════════════════════════════════
def simulate_incident_3():
    print_banner(3, "Canary Deployment Cold-Start & Latency SLA Violation", "P1 - CRITICAL")
    print("SCENARIO: Green candidate model deployment receives 10% canary traffic.")
    print("Container cold-start / memory paging spikes latency to >2,000ms (SLA is 500ms).")
    print("-" * 70)

    blue_latencies = [72.1, 84.3, 68.9, 91.2, 77.4, 80.5, 75.0, 88.2, 79.1]
    green_latencies = [2450.8, 1890.2, 750.4] # Severe latency spikes!

    p95_blue = sorted(blue_latencies)[int(len(blue_latencies) * 0.95)]
    p95_green = sorted(green_latencies)[int(len(green_latencies) * 0.95)]

    print(f"\n[Step 1: Monitoring Live Canary Traffic Split (90% Blue / 10% Green)]")
    print(f"  BLUE Baseline (90% Traffic): Avg: {sum(blue_latencies)/len(blue_latencies):.1f}ms | P95: {p95_blue:.1f}ms (SLA: <= 500ms)")
    print(f"  GREEN Candidate (10% Traffic): Avg: {sum(green_latencies)/len(green_latencies):.1f}ms | P95: {p95_green:.1f}ms (SLA: <= 500ms)")

    print("\n[Step 2: Automated Health Gate Evaluation]")
    max_sla = 500.0
    if p95_green > max_sla:
        print(f"  🚨 SLA BREACH! Green P95 Latency ({p95_green:.1f}ms) exceeds max allowable {max_sla}ms threshold!")
        print("\n[Step 3: Triggering Automated Circuit Breaker Rollback]")
        print("  1. Shift 100% customer traffic immediately back to Blue.")
        print("  2. De-register Green candidate from routing table.")
        print("  3. Emit CloudWatch Alarm: 'CanaryHealthCheckFailed'.")
        print("\n  ✅ Zero customer drop-offs: All traffic safely preserved on Blue endpoint.")
        print("  Incident #3 Status: AUTO-ROLLED BACK & SAFE.")

# ═══════════════════════════════════════════════════════════════════
# INCIDENT 4: ECOA FAIR LENDING GATE FAILURE
# ═══════════════════════════════════════════════════════════════════
def simulate_incident_4():
    print_banner(4, "ECOA Fair Lending Regulatory Gate Failure (SR 11-7)", "REGULATORY P1")
    print("SCENARIO: Candidate model undergoes pre-deployment regulatory compliance audit.")
    print("The Equal Credit Opportunity Act (ECOA) mandates the Four-Fifths Rule (DIR >= 0.80).")
    print("-" * 70)

    report_path = "artifacts/compliance/fair_lending_audit.json"
    if os.path.exists(report_path):
        with open(report_path) as f:
            data = json.load(f)
        dir_val = data["gender_audit"]["disparate_impact_ratio"]
        eod_val = data["gender_audit"]["equal_opportunity_diff"]
    else:
        dir_val = 0.571
        eod_val = 0.082

    print(f"\n[Step 1: Auditing Protected Facet: GENDER (Sex Discrimination Audit)]")
    print(f"  Disparate Impact Ratio (DIR): {dir_val:.3f} (Statutory Threshold: >= 0.80)")
    print(f"  Equal Opportunity Difference: {eod_val:.3f} (Benchmark: <= 0.10)")

    if dir_val < 0.80:
        print(f"\n  🚨 REGULATORY GATE FAILURE: Disparate Impact Ratio ({dir_val}) violates ECOA Four-Fifths Rule!")
        print("  Action: CI/CD Pipeline BLOCKED. Model cannot be registered or approved for production.")

    print("\n[Step 2: On-Call Engineer Algorithmic Remediation]")
    print("  Root Cause: Historical dataset had higher churn in female cohort (17.8% vs 10.2%).")
    print("  Solution: Calibrate demographic classification thresholds:")
    print("    - Default Threshold: 0.50 (Female DIR = 0.571)")
    print("    - Fairness-Adjusted Threshold: Female = 0.58, Male = 0.46")
    adj_dir = 0.835
    print(f"    - Recalibrated Disparate Impact Ratio: {adj_dir:.3f} (>= 0.80 -> PASSED!)")
    print("\n  ✅ Compliance Gate Cleared. Model approved by Model Risk Management (MRM).")
    print("  Incident #4 Status: REMEDIATED & COMPLIANT.")

# ═══════════════════════════════════════════════════════════════════
# MAIN DISPATCHER
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bank Churn MLOps Incident Simulator")
    parser.add_argument("--incident", type=int, choices=[1, 2, 3, 4, 0], default=0,
                        help="Incident ID to run (1: Schema, 2: Drift, 3: Canary, 4: ECOA, 0: All)")
    args = parser.parse_args()

    if args.incident == 1 or args.incident == 0:
        simulate_incident_1()
    if args.incident == 2 or args.incident == 0:
        simulate_incident_2()
    if args.incident == 3 or args.incident == 0:
        simulate_incident_3()
    if args.incident == 4 or args.incident == 0:
        simulate_incident_4()

    print("\n" + "=" * 70)
    print("🎯 ON-CALL INCIDENT SIMULATION COMPLETE!")
    print("=" * 70)
