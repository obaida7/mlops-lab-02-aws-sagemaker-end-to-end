"""
Deploy wine quality model using boto3 directly.
Bypasses SageMaker Python SDK serving bugs entirely.
"""
import boto3
import os
import time
import tarfile
import tempfile
import shutil

region = os.environ["AWS_REGION"]
role = os.environ["SAGEMAKER_ROLE_ARN"]
endpoint_name = "wine-quality-endpoint"
model_name = "wine-quality-model"

boto_session = boto3.Session(region_name=region)
sm = boto_session.client("sagemaker")
s3 = boto_session.client("s3")

IMAGE_URI = f"683313688378.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"

print(f"Region: {region}")
print(f"Role: {role}")

# ═══════════════════════════════════════════════════════════════════
# STEP 1: Find Production model from MLflow Registry
# ═══════════════════════════════════════════════════════════════════
print("\n[1/5] Finding Production model in MLflow Registry...")

import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI")
if not MLFLOW_URI:
    raise ValueError("MLFLOW_TRACKING_URI must be set in environment!")

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

model_name_mlflow = "wine-quality-model"
try:
    prod_version = client.get_model_version_by_alias(model_name_mlflow, "production")
except Exception as e:
    print("No model found with 'production' alias! Skipping deployment.")
    sys.exit(1)

version_num = prod_version.version
model_uri = f"models:/{model_name_mlflow}@production"
print(f"  Production Model Version: {version_num} (Run ID: {prod_version.run_id})")

# Idempotency Check: Are we already running this version?
try:
    ep = sm.describe_endpoint(EndpointName=endpoint_name)
    tags = sm.list_tags(ResourceArn=ep["EndpointArn"]).get("Tags", [])
    current_version = next((t["Value"] for t in tags if t["Key"] == "MLflowVersion"), None)
    if current_version == str(version_num):
        print(f"  Endpoint is already running version {version_num}. No deployment needed!")
        exit(0)
except sm.exceptions.ClientError:
    pass # No endpoint exists yet

# ═══════════════════════════════════════════════════════════════════
# STEP 2: Download from MLflow and repackage with inference code
# ═══════════════════════════════════════════════════════════════════
print("\n[2/5] Downloading model from MLflow and repackaging...")

tmpdir = tempfile.mkdtemp()
extract_dir = os.path.join(tmpdir, "contents")

# Download the model directory directly from MLflow
print(f"  Downloading from {model_uri}...")
local_model_dir = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=extract_dir)

# Remove any old code/ directory
code_dir = os.path.join(extract_dir, "code")
if os.path.exists(code_dir):
    shutil.rmtree(code_dir)
os.makedirs(code_dir)

# Write inference.py (Modified to load model.xgb from MLflow format)
with open(os.path.join(code_dir, "inference.py"), "w") as f:
    f.write("""import os
import numpy as np
import xgboost as xgb

FEATURE_NAMES = [
    "Wine", "Alcohol", "Malic.acid", "Ash", "Acl", "Mg",
    "Phenols", "Flavanoids", "Nonflavanoid.phenols", "Proanth",
    "Color.int", "Hue", "OD"
]

def model_fn(model_dir):
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(model_dir, "model.xgb"))
    return model

def input_fn(request_body, content_type):
    if content_type == "text/csv":
        lines = request_body.strip().split("\\n")
        parsed = []
        for line in lines:
            row = [float(x.strip()) for x in line.split(",")]
            parsed.append(row)
        return np.array(parsed)
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    dmatrix = xgb.DMatrix(input_data, feature_names=FEATURE_NAMES)
    prediction = model.get_booster().predict(dmatrix)
    return prediction

def output_fn(prediction, accept):
    return ",".join(str(round(float(p), 4)) for p in prediction)
""")

# Write MINIMAL requirements.txt
with open(os.path.join(code_dir, "requirements.txt"), "w") as f:
    f.write("xgboost==2.0.3\n")

print(f"  code/ contents: {os.listdir(code_dir)}")

# Repackage tar
new_tar = os.path.join(tmpdir, "model.tar.gz")
with tarfile.open(new_tar, "w:gz") as tar:
    for item in os.listdir(extract_dir):
        tar.add(os.path.join(extract_dir, item), arcname=item)

# Upload
bucket = os.environ["S3_BUCKET"]
new_key = f"models/deploy/production-v{version_num}-{int(time.time())}.tar.gz"
s3.upload_file(new_tar, bucket, new_key)
new_model_uri = f"s3://{bucket}/{new_key}"
print(f"  Uploaded: {new_model_uri}")

shutil.rmtree(tmpdir)

# ═══════════════════════════════════════════════════════════════════
# STEP 3: Cleanup ALL existing resources
# ═══════════════════════════════════════════════════════════════════
print("\n[3/5] Cleaning up existing resources...")

# Delete endpoint (wait if in-progress)
try:
    ep = sm.describe_endpoint(EndpointName=endpoint_name)
    status = ep["EndpointStatus"]
    print(f"  Endpoint status: {status}")

    # Wait for any in-progress state to finish
    while status in ("Creating", "Updating", "RollingBack", "Deleting"):
        print(f"  Endpoint is {status}, waiting 30s...")
        time.sleep(30)
        try:
            status = sm.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"]
        except sm.exceptions.ClientError:
            print("  Endpoint gone")
            status = "Gone"
            break

    if status != "Gone":
        # Now it's InService or Failed — safe to delete
        sm.delete_endpoint(EndpointName=endpoint_name)
        print("  Delete requested, waiting for removal...")
        while True:
            try:
                time.sleep(15)
                sm.describe_endpoint(EndpointName=endpoint_name)
            except sm.exceptions.ClientError:
                print("  Endpoint deleted")
                break

except sm.exceptions.ClientError:
    print("  No existing endpoint")

# Delete endpoint configs
try:
    configs = sm.list_endpoint_configs(NameContains="wine-quality")
    for cfg in configs.get("EndpointConfigs", []):
        sm.delete_endpoint_config(EndpointConfigName=cfg["EndpointConfigName"])
        print(f"  Deleted config: {cfg['EndpointConfigName']}")
except Exception:
    pass

# Delete old models
try:
    sm.delete_model(ModelName=model_name)
    print(f"  Deleted model: {model_name}")
except Exception:
    pass

try:
    models = sm.list_models(SortBy="CreationTime", SortOrder="Descending", MaxResults=10)
    for m in models["Models"]:
        mn = m["ModelName"]
        if "wine" in mn.lower() or "sagemaker-scikit" in mn.lower():
            sm.delete_model(ModelName=mn)
            print(f"  Deleted model: {mn}")
except Exception:
    pass

print("  Waiting 15s...")
time.sleep(15)

# ═══════════════════════════════════════════════════════════════════
# STEP 4: Create model with EXPLICIT env vars (no SDK magic)
# ═══════════════════════════════════════════════════════════════════
print("\n[4/5] Creating SageMaker model...")

sm.create_model(
    ModelName=model_name,
    PrimaryContainer={
        "Image": IMAGE_URI,
        "ModelDataUrl": new_model_uri,
        "Environment": {
            "SAGEMAKER_PROGRAM": "inference.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
            "SAGEMAKER_REGION": region,
        },
    },
    ExecutionRoleArn=role,
)
print(f"  Model created: {model_name}")

# ═══════════════════════════════════════════════════════════════════
# STEP 5: Create endpoint config + endpoint
# ═══════════════════════════════════════════════════════════════════
print("\n[5/5] Creating endpoint...")

config_name = f"{endpoint_name}-config-{int(time.time())}"

sm.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=[
        {
            "VariantName": "primary",
            "ModelName": model_name,
            "InstanceType": "ml.m5.large",
            "InitialInstanceCount": 1,
        }
    ],
    DataCaptureConfig={
        "EnableCapture": True,
        "InitialSamplingPercentage": 100,
        "DestinationS3Uri": f"s3://{bucket}/data-capture",
        "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
        "CaptureContentTypeHeader": {"CsvContentTypes": ["text/csv"]},
    }
)
print(f"  Endpoint config: {config_name}")

sm.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=config_name,
    Tags=[{"Key": "MLflowVersion", "Value": str(version_num)}]
)
print(f"  Endpoint creation started: {endpoint_name}")

# Wait for InService
print("  Waiting for endpoint to be InService...")
waiter = sm.get_waiter("endpoint_in_service")
waiter.wait(
    EndpointName=endpoint_name,
    WaiterConfig={"Delay": 30, "MaxAttempts": 40},
)

final_status = sm.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"]
print(f"\n{'='*60}")
print(f"  ENDPOINT STATUS: {final_status}")
print(f"  ENDPOINT NAME:   {endpoint_name}")
print(f"{'='*60}")

# ═══════════════════════════════════════════════════════════════════
# STEP 6: Configure Application Auto Scaling
# ═══════════════════════════════════════════════════════════════════
if final_status == "InService":
    print("\n[6/6] Configuring Application Auto Scaling...")
    autoscaling = boto_session.client("application-autoscaling")
    resource_id = f"endpoint/{endpoint_name}/variant/primary"
    
    # Register scalable target
    autoscaling.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=1,
        MaxCapacity=3
    )
    
    # Define scaling policy
    autoscaling.put_scaling_policy(
        PolicyName="wine-invocations-scaling-policy",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": 100.0,
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
            },
            "ScaleInCooldown": 300,
            "ScaleOutCooldown": 60
        }
    )
    print("  AutoScaling configured successfully! (Max 3 instances)")