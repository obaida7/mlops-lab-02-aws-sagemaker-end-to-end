import boto3
import time
import os

region = os.environ.get("AWS_REGION", "us-east-2")
pipeline_name = "bank-churn-pipeline"

sm = boto3.client("sagemaker", region_name=region)

print(f"Waiting for latest execution of {pipeline_name} to finish in {region}...")
time.sleep(10)

executions = sm.list_pipeline_executions(
    PipelineName=pipeline_name,
    SortBy="CreationTime",
    SortOrder="Descending"
)["PipelineExecutionSummaries"]

if not executions:
    print("No pipeline executions found!")
    exit(1)

latest_arn = executions[0]["PipelineExecutionArn"]
print(f"Tracking execution: {latest_arn}")

while True:
    res = sm.describe_pipeline_execution(PipelineExecutionArn=latest_arn)
    status = res["PipelineExecutionStatus"]
    print(f"Current Pipeline Status: {status}...")

    if status == "Succeeded":
        print("Pipeline finished successfully! Model is now registered in SageMaker Model Registry under 'bank-customer-churn'.")
        break
    elif status in ["Failed", "Fault", "Stopped"]:
        failure_reason = res.get("FailureReason", "Unknown reason")
        print(f"Pipeline failed: {failure_reason}")
        exit(1)

    time.sleep(20)
