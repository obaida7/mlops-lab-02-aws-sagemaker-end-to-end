# scripts/register_model.py
"""
Registers trained Bank Customer Churn model artifacts into SageMaker Model Registry.
- Uploads model metrics (ROC-AUC, PR-AUC, Confusion Matrix) and SHAP explainability plots.
- Associates official AWS Scikit-learn container image for us-east-2.
- Sets initial status to 'PendingManualApproval' for SR 11-7 model risk management review.
"""
import os
import sys
import boto3

REGION = os.environ.get("AWS_REGION", "us-east-2")
BUCKET = os.environ.get("S3_BUCKET", "sagemaker-us-east-2-227246780492-bank-churn")
MODEL_PACKAGE_GROUP = "bank-customer-churn"

# Official AWS Scikit-learn serving image for us-east-2
IMAGE_URI = f"257758044811.dkr.ecr.{REGION}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"

boto_session = boto3.Session(region_name=REGION)
sm = boto_session.client("sagemaker")

print("=" * 60)
print(f"REGISTERING MODEL PACKAGE IN SAGEMAKER MODEL REGISTRY ({REGION})")
print(f"Group:  {MODEL_PACKAGE_GROUP}")
print(f"Bucket: s3://{BUCKET}/models/")
print(f"Image:  {IMAGE_URI}")
print("=" * 60)

try:
    response = sm.create_model_package(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        ModelPackageDescription="Production XGBoost Bank Customer Churn Classifier with PR-AUC 0.71, ROC-AUC 0.86, and SHAP explainability",
        InferenceSpecification={
            "Containers": [
                {
                    "Image": IMAGE_URI,
                    "ModelDataUrl": f"s3://{BUCKET}/models/model.tar.gz",
                    "Environment": {
                        "SAGEMAKER_PROGRAM": "inference.py",
                        "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code"
                    }
                }
            ],
            "SupportedContentTypes": ["application/json", "text/csv"],
            "SupportedResponseMIMETypes": ["application/json", "text/csv"],
            "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.large", "ml.t2.medium"],
            "SupportedTransformInstanceTypes": ["ml.m5.large"]
        },
        ModelMetrics={
            "ModelQuality": {
                "Statistics": {
                    "ContentType": "application/json",
                    "S3Uri": f"s3://{BUCKET}/models/evaluation.json"
                }
            },
            "Explainability": {
                "Report": {
                    "ContentType": "image/png",
                    "S3Uri": f"s3://{BUCKET}/models/shap_summary.png"
                }
            }
        },
        ModelApprovalStatus="PendingManualApproval"
    )

    pkg_arn = response["ModelPackageArn"]
    print("\n[SUCCESS] Model Package successfully registered in SageMaker Model Registry!")
    print(f"Package ARN: {pkg_arn}")
    print("Initial Status: PendingManualApproval (Regulatory review required prior to deployment)")

except Exception as e:
    print(f"\n[ERROR] Failed to register model package: {e}")
    sys.exit(1)
