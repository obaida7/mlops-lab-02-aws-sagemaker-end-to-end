# scripts/feature_store.py
"""
Bank Customer Feature Store (Online & Offline Store).
Replicates Fifth Third Bank / Enterprise Banking Feature Store Architecture:
- Warehouse Source: Ingests raw customer analytical records (Snowflake / Data Lake export).
- Online Store (DynamoDB: 'bank_customer_features'): Ultra-low latency (<10ms) feature retrieval
  for real-time inference (eliminating training-serving skew).
- Offline Store (Amazon S3 Parquet): Point-in-time correct historical feature repository
  used for training pipelines without data leakage.
"""
import os
import sys
import time
import json
import boto3
import pandas as pd
from decimal import Decimal

REGION = os.environ.get("AWS_REGION", "us-east-2")
BUCKET = os.environ.get("S3_BUCKET", "sagemaker-us-east-2-227246780492-bank-churn")
DYNAMODB_TABLE = "bank_customer_features"

if "AWS_PROFILE" not in os.environ and "AWS_ACCESS_KEY_ID" not in os.environ:
    os.environ["AWS_PROFILE"] = "agent-toolkit"

boto_session = boto3.Session(region_name=REGION)
dynamodb = boto_session.resource("dynamodb")
table = dynamodb.Table(DYNAMODB_TABLE)
s3 = boto_session.client("s3")

def ingest_from_warehouse(source_csv_path: str = "data/bank_churn.csv", sample_limit: int = 1000):
    """
    Ingests raw customer data into both the Online Store (DynamoDB) and Offline Store (S3).
    Simulates the ELT pipeline from Snowflake / Enterprise Data Warehouse.
    """
    print("=" * 65)
    print("INGESTING FEATURES INTO BANKING FEATURE STORE (ONLINE + OFFLINE)")
    print(f"Source:        {source_csv_path}")
    print(f"Online Store:  Amazon DynamoDB ({DYNAMODB_TABLE})")
    print(f"Offline Store: s3://{BUCKET}/feature-store/offline/")
    print("=" * 65)

    if not os.path.exists(source_csv_path):
        source_csv_path = "terraform/bank_churn.csv"

    df = pd.read_csv(source_csv_path)
    if "CustomerId" not in df.columns:
        # Generate synthetic unique customer IDs if missing
        df["CustomerId"] = range(15600000, 15600000 + len(df))

    # Add point-in-time event timestamp
    current_time = int(time.time())
    df["event_timestamp"] = current_time

    # 1. Offline Store: Save as point-in-time Parquet partition in S3
    print("\n[1/2] Writing Offline Feature Store Parquet partition to S3...")
    os.makedirs("artifacts/feature_store", exist_ok=True)
    parquet_path = f"artifacts/feature_store/features_{current_time}.parquet"
    df.to_parquet(parquet_path, index=False)

    s3_key = f"feature-store/offline/year={time.strftime('%Y')}/month={time.strftime('%m')}/features_{current_time}.parquet"
    s3.upload_file(parquet_path, BUCKET, s3_key)
    print(f"  Successfully uploaded offline partition: s3://{BUCKET}/{s3_key}")

    # 2. Online Store: Batch write customer records into DynamoDB
    print(f"\n[2/2] Ingesting {sample_limit} customer feature vectors into DynamoDB Online Store...")
    sample_df = df.head(sample_limit)
    
    with table.batch_writer() as batch:
        for idx, row in sample_df.iterrows():
            item = {
                "CustomerId": int(row["CustomerId"]),
                "CreditScore": int(row["CreditScore"]),
                "Geography": int(row["Geography"]),
                "Gender": int(row["Gender"]),
                "Age": int(row["Age"]),
                "Tenure": int(row["Tenure"]),
                "Balance": Decimal(str(round(row["Balance"], 2))),
                "NumOfProducts": int(row["NumOfProducts"]),
                "HasCrCard": int(row["HasCrCard"]),
                "IsActiveMember": int(row["IsActiveMember"]),
                "EstimatedSalary": Decimal(str(round(row["EstimatedSalary"], 2))),
                "last_updated": current_time
            }
            batch.put_item(Item=item)
            if (idx + 1) % 250 == 0:
                print(f"  Ingested {idx + 1}/{sample_limit} customer records...")

    print(f"\n[SUCCESS] Ingested {sample_limit} online customer records into DynamoDB!")


def get_online_features(customer_id: int) -> dict:
    """
    Sub-10ms point lookup of pre-computed customer features for real-time inference.
    Used by online banking microservices and the FastAPI serving layer.
    """
    t0 = time.time()
    response = table.get_item(Key={"CustomerId": int(customer_id)})
    lookup_latency_ms = (time.time() - t0) * 1000

    if "Item" in response:
        item = response["Item"]
        # Convert Decimals back to float
        features = {
            "CreditScore": int(item["CreditScore"]),
            "Geography": int(item["Geography"]),
            "Gender": int(item["Gender"]),
            "Age": int(item["Age"]),
            "Tenure": int(item["Tenure"]),
            "Balance": float(item["Balance"]),
            "NumOfProducts": int(item["NumOfProducts"]),
            "HasCrCard": int(item["HasCrCard"]),
            "IsActiveMember": int(item["IsActiveMember"]),
            "EstimatedSalary": float(item["EstimatedSalary"]),
            "_lookup_latency_ms": round(lookup_latency_ms, 2)
        }
        return features
    return None


if __name__ == "__main__":
    ingest_from_warehouse()
    
    # Smoke test single customer lookup
    test_id = 15600001
    print(f"\nTesting Online Feature Store Lookup for CustomerId: {test_id}...")
    feats = get_online_features(test_id)
    if feats:
        print(f"  Features Retrieved in {feats['_lookup_latency_ms']}ms:")
        print(json.dumps(feats, indent=2))
    else:
        print("  Customer not found.")
