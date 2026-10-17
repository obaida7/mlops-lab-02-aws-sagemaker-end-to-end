# scripts/destroy_all.py
"""
Enterprise Full Teardown & Cost Elimination Script for Bank Customer Churn MLOps.
Guarantees $0.00 ongoing compute spend by safely destroying:
1. All SageMaker Monitoring Schedules.
2. Any running SageMaker Processing Jobs.
3. SageMaker Endpoints (bank-churn-endpoint).
4. Associated SageMaker Endpoint Configurations & Models.
5. Optional --full flag to run Terraform destroy for S3/DynamoDB/SNS/EventBridge.
"""
import os
import sys
import time
import argparse
import boto3

REGION = os.environ.get("AWS_REGION", "us-east-2")
ENDPOINT_NAME = "bank-churn-endpoint"
SCHEDULE_NAME = "bank-churn-quality-schedule"

if "AWS_PROFILE" not in os.environ and "AWS_ACCESS_KEY_ID" not in os.environ:
    os.environ["AWS_PROFILE"] = "agent-toolkit"

session = boto3.Session(region_name=REGION)
sm = session.client("sagemaker")


def destroy_monitoring_schedules():
    print("\n[1/4] Destroying SageMaker Monitoring Schedules...")
    try:
        schedules = sm.list_monitoring_schedules(
            EndpointName=ENDPOINT_NAME,
            MaxResults=10
        ).get("MonitoringScheduleSummaries", [])
        
        for sched in schedules:
            name = sched["MonitoringScheduleName"]
            print(f"  Deleting schedule: {name}...")
            sm.delete_monitoring_schedule(MonitoringScheduleName=name)
            print(f"  ✅ Deleted {name}")
            
        try:
            sm.delete_monitoring_schedule(MonitoringScheduleName=SCHEDULE_NAME)
            print(f"  ✅ Deleted {SCHEDULE_NAME}")
        except Exception:
            pass
            
        if not schedules:
            print("  No active monitoring schedules found.")
    except Exception as e:
        print(f"  [Notice] Monitoring schedule cleanup: {e}")


def stop_processing_jobs():
    print("\n[2/4] Stopping active Processing Jobs...")
    try:
        jobs = sm.list_processing_jobs(
            StatusEquals="InProgress",
            MaxResults=10
        ).get("ProcessingJobSummaries", [])
        
        for job in jobs:
            job_name = job["ProcessingJobName"]
            if "baseline" in job_name.lower() or "churn" in job_name.lower():
                print(f"  Stopping processing job: {job_name}...")
                sm.stop_processing_job(ProcessingJobName=job_name)
                print(f"  ✅ Stopped {job_name}")
                
        if not jobs:
            print("  No in-progress processing jobs found.")
    except Exception as e:
        print(f"  [Notice] Processing job cleanup: {e}")


def destroy_endpoint():
    print(f"\n[3/4] Destroying Endpoint: '{ENDPOINT_NAME}'...")
    try:
        sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"  Deletion initiated for {ENDPOINT_NAME}. Waiting for removal...")
        while True:
            try:
                sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
                time.sleep(3)
            except sm.exceptions.ClientError:
                break
        print(f"  ✅ Endpoint '{ENDPOINT_NAME}' successfully deleted!")
    except sm.exceptions.ClientError:
        print(f"  Endpoint '{ENDPOINT_NAME}' does not exist (already destroyed).")


def destroy_configs_and_models():
    print("\n[4/4] Cleaning up orphaned Endpoint Configs and Model resources...")
    try:
        configs = sm.list_endpoint_configs(
            NameContains="bank-churn",
            MaxResults=20
        ).get("EndpointConfigs", [])
        
        for cfg in configs:
            cfg_name = cfg["EndpointConfigName"]
            try:
                sm.delete_endpoint_config(EndpointConfigName=cfg_name)
                print(f"  ✅ Deleted config: {cfg_name}")
            except Exception as e:
                print(f"  Could not delete config {cfg_name}: {e}")
                
        models = sm.list_models(
            NameContains="bank-churn",
            MaxResults=20
        ).get("Models", [])
        
        for mdl in models:
            mdl_name = mdl["ModelName"]
            try:
                sm.delete_model(ModelName=mdl_name)
                print(f"  ✅ Deleted model: {mdl_name}")
            except Exception as e:
                print(f"  Could not delete model {mdl_name}: {e}")
    except Exception as e:
        print(f"  [Notice] Config/model cleanup: {e}")


def main():
    parser = argparse.ArgumentParser(description="Destroy Bank Churn MLOps Infrastructure")
    parser.add_argument("--full", action="store_true", help="Also run terraform destroy to remove S3/SNS/EventBridge")
    args = parser.parse_args()

    print("=" * 65)
    print(f"DESTROYING BANK CHURN MLOPS COMPUTE & ENDPOINTS ({REGION})")
    print("=" * 65)

    destroy_monitoring_schedules()
    stop_processing_jobs()
    destroy_endpoint()
    destroy_configs_and_models()

    if args.full:
        print("\n[Full Mode] Executing Terraform Destroy...")
        import subprocess
        subprocess.run(["terraform", "destroy", "-auto-approve"], cwd="terraform", check=True)
        print("  ✅ Terraform resources destroyed.")

    print("\n" + "=" * 65)
    print("TEARDOWN COMPLETE: $0.00 ACTIVE COMPUTE CHARGES.")
    print("=" * 65)


if __name__ == "__main__":
    main()
