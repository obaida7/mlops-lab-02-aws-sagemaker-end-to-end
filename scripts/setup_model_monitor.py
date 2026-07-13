import os
import boto3
import sagemaker
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.model_monitor import CronExpressionGenerator
from sagemaker.model_monitor import ModelBiasMonitor, ModelExplainabilityMonitor
from sagemaker.clarify import BiasConfig, DataConfig, ModelConfig, ModelPredictedLabelConfig, SHAPConfig

REGION = os.environ.get("AWS_REGION", "us-east-1")
ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN")
BUCKET = os.environ.get("S3_BUCKET")
ENDPOINT_NAME = "wine-quality-endpoint"
MODEL_NAME = "wine-quality-model"

if not ROLE_ARN or not BUCKET:
    raise ValueError("SAGEMAKER_ROLE_ARN and S3_BUCKET must be set!")

sess = sagemaker.Session(boto3.Session(region_name=REGION))

# ═══════════════════════════════════════════════════════════════════
# 1. DATA QUALITY MONITORING (Data Drift)
# ═══════════════════════════════════════════════════════════════════
print("\n[1/3] Setting up DefaultModelMonitor (Data Quality)...")
my_monitor = DefaultModelMonitor(
    role=ROLE_ARN,
    instance_count=1,
    instance_type='ml.m5.large',
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
    sagemaker_session=sess
)

print("  Running baseline job (this takes a few minutes)...")
try:
    my_monitor.suggest_baseline(
        baseline_dataset=f"s3://{BUCKET}/data/wine.csv",
        dataset_format=DatasetFormat.csv(header=True),
        output_s3_uri=f"s3://{BUCKET}/model-monitor/data-quality/baseline",
        wait=True,
        logs=False
    )
except Exception as e:
    print(f"  Warning: baseline failed: {e}")

try:
    my_monitor.create_monitoring_schedule(
        monitor_schedule_name='wine-data-quality-schedule',
        endpoint_input=ENDPOINT_NAME,
        output_s3_uri=f"s3://{BUCKET}/model-monitor/data-quality/reports",
        statistics=my_monitor.baseline_statistics(),
        constraints=my_monitor.suggested_constraints(),
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        enable_cloudwatch_metrics=True,
    )
    print("  Data Quality Schedule created successfully!")
except Exception as e:
    print(f"  Warning/Error creating DQ schedule: {e}")

# ═══════════════════════════════════════════════════════════════════
# Common Configs for Clarify (Bias & Explainability)
# ═══════════════════════════════════════════════════════════════════
data_config = DataConfig(
    s3_data_input_path=f"s3://{BUCKET}/data/wine.csv",
    s3_output_path=f"s3://{BUCKET}/model-monitor/clarify-output",
    label="quality",
    headers=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"],
    dataset_type="text/csv",
)

model_config = ModelConfig(
    model_name=MODEL_NAME,
    instance_count=1,
    instance_type='ml.m5.large',
)

model_predicted_label_config = ModelPredictedLabelConfig(probability_threshold=0.8)

# ═══════════════════════════════════════════════════════════════════
# 2. MODEL BIAS MONITORING (Bias Drift)
# ═══════════════════════════════════════════════════════════════════
print("\n[2/3] Setting up ModelBiasMonitor...")
bias_monitor = ModelBiasMonitor(
    role=ROLE_ARN,
    sagemaker_session=sess,
    max_runtime_in_seconds=3600,
)

bias_config = BiasConfig(
    label_values_or_threshold=[5.0],
    facet_name="alcohol",
    facet_values_or_threshold=[10.0]
)

try:
    print("  Running Bias baseline job...")
    bias_monitor.suggest_baseline(
        data_config=data_config,
        model_config=model_config,
        bias_config=bias_config,
        model_predicted_label_config=model_predicted_label_config,
        wait=True,
        logs=False
    )
    
    bias_monitor.create_monitoring_schedule(
        monitor_schedule_name="wine-bias-schedule",
        endpoint_input=ENDPOINT_NAME,
        output_s3_uri=f"s3://{BUCKET}/model-monitor/bias/reports",
        schedule_cron_expression=CronExpressionGenerator.hourly(),
    )
    print("  Bias Schedule created successfully!")
except Exception as e:
    print(f"  Warning/Error creating Bias schedule: {e}")

# ═══════════════════════════════════════════════════════════════════
# 3. MODEL EXPLAINABILITY MONITORING (Feature Attribution Drift)
# ═══════════════════════════════════════════════════════════════════
print("\n[3/3] Setting up ModelExplainabilityMonitor...")
explainability_monitor = ModelExplainabilityMonitor(
    role=ROLE_ARN,
    sagemaker_session=sess,
    max_runtime_in_seconds=3600,
)

shap_config = SHAPConfig(
    baseline=[
        [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
    ],
    num_samples=100,
    agg_method="mean_abs"
)

try:
    print("  Running Explainability baseline job...")
    explainability_monitor.suggest_baseline(
        data_config=data_config,
        model_config=model_config,
        explainability_config=shap_config,
        model_predicted_label_config=model_predicted_label_config,
        wait=True,
        logs=False
    )
    
    explainability_monitor.create_monitoring_schedule(
        monitor_schedule_name="wine-explainability-schedule",
        endpoint_input=ENDPOINT_NAME,
        output_s3_uri=f"s3://{BUCKET}/model-monitor/explainability/reports",
        schedule_cron_expression=CronExpressionGenerator.hourly(),
    )
    print("  Explainability Schedule created successfully!")
except Exception as e:
    print(f"  Warning/Error creating Explainability schedule: {e}")

print("\nAll monitoring setups completed!")
