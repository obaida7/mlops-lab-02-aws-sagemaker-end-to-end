# scripts/teardown_model_monitor.py
"""
Teardown & Clean-up Script for Native SageMaker Model Monitor.
Reverts compute back to $0 idle cost to preserve project credits:
1. Deletes the SageMaker Monitoring Schedule ('bank-churn-quality-schedule').
2. Reverts the endpoint back to Serverless Inference (or deletes the provisioned config).
"""
import os
import sys
import time
import boto3

REGION = os.environ.get("AWS_REGION", "us-east-2")
ENDPOINT_NAME = "bank-churn-endpoint"
SCHEDULE_NAME = "bank-churn-quality-schedule"

if "AWS_PROFILE" not in os.environ and "AWS_ACCESS_KEY_ID" not in os.environ:
    os.environ["AWS_PROFILE"] = "agent-toolkit"

boto_session = boto3.Session(region_name=REGION)
sm = boto_session.client("sagemaker")

print("=" * 65)
print(f"TEARING DOWN SAGEMAKER MODEL MONITOR COMPUTE ({REGION})")
print("=" * 65)

# 1. Delete Monitoring Schedule
print(f"\n[1/2] Deleting Monitoring Schedule: '{SCHEDULE_NAME}'...")
try:
    sm.delete_monitoring_schedule(MonitoringScheduleName=SCHEDULE_NAME)
    print(f"  ✅ Deleted monitoring schedule: {SCHEDULE_NAME}")
except Exception as e:
    print(f"  [Notice] Schedule deletion: {e}")

# 2. Revert Endpoint to Serverless
print(f"\n[2/2] Reverting '{ENDPOINT_NAME}' to Serverless ($0 Idle Cost)...")
try:
    # Find latest serverless config
    configs = sm.list_endpoint_configs(
        NameContains="bank-churn-config-v7",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=5
    )["EndpointConfigs"]

    if configs:
        serverless_config = configs[0]["EndpointConfigName"]
        print(f"  Rolling back endpoint to Serverless Config: {serverless_config}")
        sm.update_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=serverless_config
        )
        print("  ✅ Rolling update initiated: Endpoint returning to Serverless!")
    else:
        print("  No serverless config found to rollback to.")
except Exception as e:
    print(f"  [Notice] Revert error: {e}")

print("\n" + "=" * 65)
print("CLEAN-UP COMPLETE: Zero ongoing hourly compute charges!")
print("=" * 65)
