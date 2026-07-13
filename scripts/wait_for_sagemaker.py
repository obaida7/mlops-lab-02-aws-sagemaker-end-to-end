import boto3
import time
import os

sm = boto3.client("sagemaker", region_name=os.environ.get("AWS_REGION", "us-east-1"))
pipeline_name = "wine-mlflow-pipeline"

print(f"Waiting for latest execution of {pipeline_name} to finish...")
time.sleep(15) # Give create_pipeline.py a moment to actually start it

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
    print(f"Status: {status}...")
    
    if status == "Succeeded":
        print("Pipeline finished successfully! Model should now be in MLflow.")
        break
    elif status in ["Failed", "Fault", "Stopped"]:
        print("Pipeline failed or was stopped.")
        exit(1)
        
    time.sleep(30)
