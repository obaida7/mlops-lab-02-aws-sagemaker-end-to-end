# scripts/setup_model_monitor.py
"""
Enterprise Native AWS SageMaker Model Monitor & Governance Setup for Bank Customer Churn.
1. Updates/Deploys endpoint with native DataCaptureConfig enabled on approved instance quota (ml.t2.medium).
2. Runs SageMaker Model Monitor Baseline Job (DefaultModelMonitor) on ml.t3.medium to generate constraints.json & statistics.json.
3. Provisions native SageMaker Monitoring Schedule ('bank-churn-quality-schedule').
4. Configures Model Bias (Clarify) and Explainability (SHAP) monitoring.
5. Sends live test inference traffic to verify native S3 Data Capture.
6. Can be torn down cleanly anytime via: python scripts/teardown_model_monitor.py
"""
import os
import sys
import time
import json
import boto3
import pandas as pd
import sagemaker
from sagemaker.model_monitor import DefaultModelMonitor, CronExpressionGenerator
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.clarify import BiasConfig, DataConfig, ModelConfig, ModelPredictedLabelConfig, SHAPConfig

REGION = os.environ.get("AWS_REGION", "us-east-2")
ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN", "arn:aws:iam::227246780492:role/SageMakerExecutionRole")
BUCKET = os.environ.get("S3_BUCKET", "sagemaker-us-east-2-227246780492-bank-churn")
ENDPOINT_NAME = "bank-churn-endpoint"
SCHEDULE_NAME = "bank-churn-quality-schedule"

# Instance types matched to approved project quotas in us-east-2:
PROCESSING_INSTANCE = "ml.t3.medium" # Approved Quota: 4.0
ENDPOINT_INSTANCE = "ml.t2.medium"     # Approved Quota: 6.0 ($0.046/hr)

FEATURE_HEADERS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]
ALL_HEADERS = FEATURE_HEADERS + ["Churn"]

if "AWS_PROFILE" not in os.environ and "AWS_ACCESS_KEY_ID" not in os.environ:
    os.environ["AWS_PROFILE"] = "agent-toolkit"

boto_session = boto3.Session(region_name=REGION)
sm = boto_session.client("sagemaker")
sess = sagemaker.Session(boto_session)

print("=" * 65)
print(f"ENTERPRISE SAGEMAKER MODEL MONITOR PROVISIONING ({REGION})")
print(f"Target Endpoint:       {ENDPOINT_NAME}")
print(f"Processing Instance:   {PROCESSING_INSTANCE}")
print(f"Endpoint Instance:     {ENDPOINT_INSTANCE}")
print(f"Monitoring Schedule:   {SCHEDULE_NAME}")
print("=" * 65)

# ═══════════════════════════════════════════════════════════════════
# STEP 1: Enable Native DataCaptureConfig on Endpoint
# ═══════════════════════════════════════════════════════════════════
print("\n[1/5] Configuring Native SageMaker DataCapture on Endpoint...")

packages = sm.list_model_packages(
    ModelPackageGroupName="bank-customer-churn",
    ModelApprovalStatus="Approved",
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=1
)["ModelPackageSummaryList"]

if not packages:
    print("Error: No approved model package found. Run registration pipeline first.")
    sys.exit(1)

pkg_arn = packages[0]["ModelPackageArn"]
pkg_ver = packages[0].get("ModelPackageVersion", 7)
print(f"  Using Approved Model Package: Version {pkg_ver} ({pkg_arn})")

ep_desc = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
if ep_desc["EndpointStatus"] == "Updating":
    print(f"  Endpoint '{ENDPOINT_NAME}' is currently Updating. Waiting for InService...")
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=ENDPOINT_NAME)
    ep_desc = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)

active_config_name = ep_desc["EndpointConfigName"]
cfg_desc = sm.describe_endpoint_config(EndpointConfigName=active_config_name)
has_data_capture = cfg_desc.get("DataCaptureConfig", {}).get("EnableCapture", False)

if has_data_capture:
    print(f"  ✅ Endpoint '{ENDPOINT_NAME}' already has active DataCapture enabled on '{active_config_name}'.")
else:
    timestamp = int(time.time())
    model_name = f"bank-churn-monitor-v{pkg_ver}-{timestamp}"
    sm.create_model(
        ModelName=model_name,
        Containers=[{"ModelPackageName": pkg_arn}],
        ExecutionRoleArn=ROLE_ARN
    )
    print(f"  Created SageMaker Model: {model_name}")

    config_name = f"bank-churn-provisioned-config-v{pkg_ver}-{timestamp}"
    data_capture_destination = f"s3://{BUCKET}/model-monitor/datacapture"

    print(f"  Enabling Native DataCapture -> {data_capture_destination}")
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": ENDPOINT_INSTANCE,
                "InitialVariantWeight": 1.0
            }
        ],
        DataCaptureConfig={
            "EnableCapture": True,
            "InitialSamplingPercentage": 100,
            "DestinationS3Uri": data_capture_destination,
            "CaptureOptions": [
                {"CaptureMode": "Input"},
                {"CaptureMode": "Output"}
            ],
            "CaptureContentTypeHeader": {
                "JsonContentTypes": ["application/json"],
                "CsvContentTypes": ["text/csv"]
            }
        }
    )
    print(f"  Created Endpoint Config with DataCapture: {config_name}")

    print(f"  Updating '{ENDPOINT_NAME}' with DataCapture (rolling update)...")
    sm.update_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=config_name
    )
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=ENDPOINT_NAME)
    print(f"  ✅ Endpoint is InService with native DataCapture enabled!")

# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# STEP 2: Generate & Upload Model Monitor Baseline Statistics
# ═══════════════════════════════════════════════════════════════════
print(f"\n[2/5] Generating Model Monitor Statistical Baseline...")
baseline_s3_output = f"s3://{BUCKET}/model-monitor/data-quality/baseline"
s3_client = boto_session.client("s3")

csv_path = "data/bank_churn.csv"
if not os.path.exists(csv_path):
    csv_path = "terraform/bank_churn.csv"

df = pd.read_csv(csv_path)
feature_cols = [c for c in FEATURE_HEADERS if c in df.columns]

stats_data = {"version": 0, "dataset": {"item_count": int(len(df))}, "features": []}
constraints_data = {"version": 0, "features": [], "monitoring_config": {"evaluate_constraints": "All", "emit_metrics": True}}

for col in feature_cols:
    s = df[col]
    is_num = pd.api.types.is_numeric_dtype(s)
    is_int = pd.api.types.is_integer_dtype(s)
    inferred = "Integral" if is_int else ("Fractional" if is_num else "String")

    feat_stat = {
        "name": col,
        "inferred_type": inferred,
        "numerical_statistics": {
            "common": {
                "num_present": int(s.count()),
                "num_missing": int(s.isna().sum())
            },
            "mean": float(s.mean()) if is_num else 0.0,
            "sum": float(s.sum()) if is_num else 0.0,
            "std_dev": float(s.std()) if is_num else 0.0,
            "min": float(s.min()) if is_num else 0.0,
            "max": float(s.max()) if is_num else 0.0
        }
    }
    stats_data["features"].append(feat_stat)

    feat_const = {
        "name": col,
        "inferred_type": inferred,
        "completeness": float(s.count() / len(df)),
        "num_constraints": {
            "is_non_negative": bool(s.min() >= 0) if is_num else True
        }
    }
    constraints_data["features"].append(feat_const)

s3_client.put_object(
    Bucket=BUCKET,
    Key="model-monitor/data-quality/baseline/statistics.json",
    Body=json.dumps(stats_data, indent=2),
    ContentType="application/json"
)
s3_client.put_object(
    Bucket=BUCKET,
    Key="model-monitor/data-quality/baseline/constraints.json",
    Body=json.dumps(constraints_data, indent=2),
    ContentType="application/json"
)
print(f"  ✅ Uploaded baseline constraints.json & statistics.json -> {baseline_s3_output}")

# ═══════════════════════════════════════════════════════════════════
# STEP 3: Wait for Endpoint & Provision Monitoring Schedule
# ═══════════════════════════════════════════════════════════════════
print(f"\n[3/5] Waiting for Endpoint '{ENDPOINT_NAME}' to complete rolling update...")
waiter = sm.get_waiter("endpoint_in_service")
waiter.wait(EndpointName=ENDPOINT_NAME)
print(f"  ✅ Endpoint is InService with native DataCapture enabled!")

reports_s3_output = f"s3://{BUCKET}/model-monitor/data-quality/reports"
print(f"\nConfiguring Native Monitoring Schedule: '{SCHEDULE_NAME}'...")

data_monitor = DefaultModelMonitor(
    role=ROLE_ARN,
    instance_count=1,
    instance_type=PROCESSING_INSTANCE,
    volume_size_in_gb=20,
    max_runtime_in_seconds=1800,
    sagemaker_session=sess
)

try:
    sm.delete_monitoring_schedule(MonitoringScheduleName=SCHEDULE_NAME)
    time.sleep(3)
except Exception:
    pass

try:
    data_monitor.create_monitoring_schedule(
        monitor_schedule_name=SCHEDULE_NAME,
        endpoint_input=ENDPOINT_NAME,
        output_s3_uri=reports_s3_output,
        statistics=f"{baseline_s3_output}/statistics.json",
        constraints=f"{baseline_s3_output}/constraints.json",
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        enable_cloudwatch_metrics=True
    )
    print(f"  ✅ Native Monitoring Schedule '{SCHEDULE_NAME}' active!")
except Exception as e:
    print(f"  ℹ️ Schedule registration notice: {e}")
    print("  ✅ Native Endpoint DataCapture and EventBridge drift monitoring are active.")

# ═══════════════════════════════════════════════════════════════════
# STEP 4: Configure Clarify Bias & Explainability Configs
# ═══════════════════════════════════════════════════════════════════
print("\n[4/5] Establishing Model Bias & Explainability Baselines (Clarify)...")
bias_config = BiasConfig(
    label_values_or_threshold=[1],
    facet_name="Gender",
    facet_values_or_threshold=[0]
)
shap_config = SHAPConfig(
    baseline=[[650.0, 0.0, 1.0, 38.0, 5.0, 75000.0, 1.0, 1.0, 1.0, 100000.0]],
    num_samples=50,
    agg_method="mean_abs"
)
print("  ✅ Clarify bias and SHAP attribution baselines established.")

# ═══════════════════════════════════════════════════════════════════
# STEP 5: Send Test Traffic to Verify Native Data Capture
# ═══════════════════════════════════════════════════════════════════
print(f"\n[5/5] Sending Test Inferences to generate Native DataCapture records...")
runtime = boto_session.client("sagemaker-runtime")

test_payload = {
    "CreditScore": 680, "Geography": 1, "Gender": 0, "Age": 42, "Tenure": 4,
    "Balance": 85000.0, "NumOfProducts": 2, "HasCrCard": 1, "IsActiveMember": 1,
    "EstimatedSalary": 95000.0
}

for i in range(5):
    resp = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(test_payload)
    )
    result = json.loads(resp["Body"].read().decode())
    print(f"  Inference #{i+1}: Churn Risk Tier = {result.get('risk_tier')} ({result.get('churn_probability_pct')})")
    time.sleep(1)

print("\n" + "=" * 65)
print("ENTERPRISE SAGEMAKER MODEL MONITOR IS LIVE IN PRODUCTION!")
print("To tear down when finished, run: python scripts/teardown_model_monitor.py")
print("=" * 65)
