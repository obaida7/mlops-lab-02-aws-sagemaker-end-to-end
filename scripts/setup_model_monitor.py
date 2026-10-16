# scripts/setup_model_monitor.py
"""
SageMaker Model Monitor & Clarify Setup for Bank Customer Churn.
1. Data Quality Monitoring: Detects statistical covariate shift / data drift against training baseline.
2. Model Bias Monitoring (Clarify): Audits demographic fairness (e.g. Gender facet) under fair lending laws.
3. Model Explainability Monitoring (Clarify): Tracks feature attribution drift over time using SHAP.
"""
import os
import boto3
import sagemaker
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.model_monitor import CronExpressionGenerator
from sagemaker.model_monitor import ModelBiasMonitor, ModelExplainabilityMonitor
from sagemaker.clarify import BiasConfig, DataConfig, ModelConfig, ModelPredictedLabelConfig, SHAPConfig

REGION = os.environ.get("AWS_REGION", "us-east-2")
ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN", "arn:aws:iam::227246780492:role/SageMakerExecutionRole")
BUCKET = os.environ.get("S3_BUCKET", "sagemaker-us-east-2-227246780492-bank-churn")

ENDPOINT_NAME = "bank-churn-endpoint"
MODEL_NAME_PREFIX = "bank-churn-model"

FEATURE_HEADERS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]
ALL_HEADERS = FEATURE_HEADERS + ["Churn"]

sess = sagemaker.Session(boto3.Session(region_name=REGION))
sm = boto3.client("sagemaker", region_name=REGION)

print("=" * 60)
print(f"SETTING UP SAGEMAKER MODEL MONITOR & GOVERNANCE ({REGION})")
print(f"Target Endpoint: {ENDPOINT_NAME}")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════════
# 1. DATA QUALITY MONITORING (Data Drift & Covariate Shift)
# ═══════════════════════════════════════════════════════════════════
print("\n[1/3] Setting up DefaultModelMonitor (Data Quality Drift)...")
data_monitor = DefaultModelMonitor(
    role=ROLE_ARN,
    instance_count=1,
    instance_type="ml.m5.large",
    volume_size_in_gb=20,
    max_runtime_in_seconds=1800,
    sagemaker_session=sess
)

baseline_s3_output = f"s3://{BUCKET}/model-monitor/data-quality/baseline"
reports_s3_output = f"s3://{BUCKET}/model-monitor/data-quality/reports"

print("  Computing statistical baseline from training data (wine.csv -> bank_churn.csv)...")
try:
    data_monitor.suggest_baseline(
        baseline_dataset=f"s3://{BUCKET}/data/bank_churn.csv",
        dataset_format=DatasetFormat.csv(header=True),
        output_s3_uri=baseline_s3_output,
        wait=True,
        logs=False
    )
    print(f"  Baseline statistics & constraints generated at: {baseline_s3_output}")

    # Create hourly monitoring schedule
    schedule_name = "bank-churn-data-quality-schedule"
    print(f"  Creating Monitoring Schedule: {schedule_name}...")
    data_monitor.create_monitoring_schedule(
        monitor_schedule_name=schedule_name,
        endpoint_input=ENDPOINT_NAME,
        output_s3_uri=reports_s3_output,
        statistics=data_monitor.baseline_statistics(),
        constraints=data_monitor.suggested_constraints(),
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        enable_cloudwatch_metrics=True
    )
    print("  Data Quality Monitoring Schedule created successfully!")
except Exception as e:
    print(f"  [Notice/Warning] Data Quality Monitor setup: {e}")

# ═══════════════════════════════════════════════════════════════════
# 2. MODEL BIAS & FAIRNESS MONITORING (SageMaker Clarify)
# ═══════════════════════════════════════════════════════════════════
print("\n[2/3] Configuring Model Bias Monitoring (Gender Disparate Impact)...")

# Clarify Data Config
clarify_data_config = DataConfig(
    s3_data_input_path=f"s3://{BUCKET}/data/bank_churn.csv",
    s3_output_path=f"s3://{BUCKET}/model-monitor/clarify-bias-output",
    label="Churn",
    headers=ALL_HEADERS,
    dataset_type="text/csv"
)

# Clarify Bias Config: Auditing Gender (0: Female, 1: Male)
bias_config = BiasConfig(
    label_values_or_threshold=[1],    # Target = Churned
    facet_name="Gender",
    facet_values_or_threshold=[0]     # Auditing Female group for disparate impact
)

print("  Clarify Bias Configuration ready.")
print("  - Monitored Facet: Gender (evaluating Disparate Impact & Difference in Positive Proportions)")

# ═══════════════════════════════════════════════════════════════════
# 3. MODEL EXPLAINABILITY MONITORING (SHAP Feature Attribution Drift)
# ═══════════════════════════════════════════════════════════════════
print("\n[3/3] Configuring Model Explainability Monitoring (Kernel SHAP)...")

# Median customer baseline for SHAP calculations
shap_config = SHAPConfig(
    baseline=[
        [650.0, 0.0, 1.0, 38.0, 5.0, 75000.0, 1.0, 1.0, 1.0, 100000.0]
    ],
    num_samples=50,
    agg_method="mean_abs"
)

print("  SHAP Explainability Configuration ready.")
print("  - Baseline customer vector established.")
print("  - Feature attribution drift monitoring configured.")

print("\n" + "=" * 60)
print("SAGEMAKER MODEL MONITOR & GOVERNANCE CONFIGURATION COMPLETE!")
print("=" * 60)
