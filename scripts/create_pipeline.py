# scripts/create_pipeline.py
"""
SageMaker Continuous Training Pipeline for Bank Customer Churn.
Orchestrates:
1. TrainingStep: Ephemeral ml.m5.large managed compute training XGBoost.
2. RegisterModelStep: Packages model weights & attaches evaluation metrics directly
   into SageMaker Model Registry under 'bank-customer-churn'.
"""
import os
import boto3
import sagemaker

from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.parameters import ParameterString
from sagemaker.model_metrics import ModelMetrics, MetricsSource

REGION = os.environ.get("AWS_REGION", "us-east-2")
ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN", "arn:aws:iam::227246780492:role/SageMakerExecutionRole")
BUCKET = os.environ.get("S3_BUCKET", "sagemaker-us-east-2-227246780492-bank-churn")

PIPELINE_NAME = "bank-churn-pipeline"
MODEL_PACKAGE_GROUP_NAME = "bank-customer-churn"

sess = sagemaker.Session(boto3.Session(region_name=REGION))

# Pipeline Parameters (Allows dynamic parameter overrides via AWS CLI / EventBridge)
input_data_param = ParameterString(
    name="InputDataUrl",
    default_value=f"s3://{BUCKET}/data/bank_churn.csv"
)

approval_status_param = ParameterString(
    name="ModelApprovalStatus",
    default_value="PendingManualApproval"
)

instance_type_param = ParameterString(
    name="TrainingInstanceType",
    default_value="ml.m5.large"
)

# 1. Estimator: Runs managed container training with custom code
estimator = SKLearn(
    entry_point="train_and_evaluate.py",
    source_dir="scripts",
    role=ROLE_ARN,
    instance_type=instance_type_param,
    instance_count=1,
    framework_version="1.2-1",
    py_version="py3",
    sagemaker_session=sess,
    output_path=f"s3://{BUCKET}/models/"
)

# 2. Step 1: Training Step
step_train = TrainingStep(
    name="TrainBankChurnModel",
    estimator=estimator,
    inputs={"train": TrainingInput(s3_data=input_data_param, content_type="text/csv")}
)

# 3. Model Metrics configuration for the Model Registry
model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=f"s3://{BUCKET}/models/evaluation.json",
        content_type="application/json"
    )
)

# 4. Step 2: Register Model Step into SageMaker Model Registry
step_register = RegisterModel(
    name="RegisterBankChurnModel",
    estimator=estimator,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv", "application/json"],
    response_types=["application/json", "text/csv"],
    inference_instances=["ml.m5.large", "ml.t2.medium"],
    transform_instances=["ml.m5.large"],
    model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
    approval_status=approval_status_param,
    model_metrics=model_metrics,
    description="Bank customer churn model evaluated with PR-AUC, ROC-AUC, and SHAP explainability"
)

# 5. Build Pipeline Directed Acyclic Graph (DAG)
pipeline = Pipeline(
    name=PIPELINE_NAME,
    parameters=[input_data_param, approval_status_param, instance_type_param],
    steps=[step_train, step_register],
    sagemaker_session=sess
)

def main():
    print(f"Creating/Updating SageMaker Pipeline: {PIPELINE_NAME} in {REGION}...")
    pipeline.upsert(role_arn=ROLE_ARN)
    print("Pipeline upserted successfully!")

    print("Triggering Pipeline Execution...")
    execution = pipeline.start()
    print(f"Pipeline started successfully! Execution ARN:\n{execution.arn}")
    return execution.arn

if __name__ == "__main__":
    main()
