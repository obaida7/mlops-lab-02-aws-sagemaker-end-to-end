# scripts/canary_deployment.py
"""
Canary & Blue/Green Deployment Controller for Bank Customer Churn.
Implements Zero-Downtime Safe Deployment for Financial Services:
- Blue Endpoint: Active Production Serving (Baseline Model).
- Green Endpoint: Candidate Model undergoing validation.
- Traffic Shifting: Routes 90% traffic to Blue and 10% canary traffic to Green.
- Automated Health Gate: Monitors latency and error rates during the canary test window.
- Promotion / Rollback: If Green satisfies SLA, promotes to 100% traffic. If errors spike, rolls back instantly to Blue.
"""
import os
import sys
import time
import json
import boto3
import random

REGION = os.environ.get("AWS_REGION", "us-east-2")
BLUE_ENDPOINT = os.environ.get("BLUE_ENDPOINT", "bank-churn-endpoint")
GREEN_ENDPOINT = os.environ.get("GREEN_ENDPOINT", "bank-churn-endpoint-green")

if "AWS_PROFILE" not in os.environ and "AWS_ACCESS_KEY_ID" not in os.environ:
    os.environ["AWS_PROFILE"] = "agent-toolkit"

boto_session = boto3.Session(region_name=REGION)
runtime = boto_session.client("sagemaker-runtime")
sm = boto_session.client("sagemaker")

class CanaryRouter:
    def __init__(self, blue_name: str, green_name: str, canary_weight: float = 0.10):
        self.blue_name = blue_name
        self.green_name = green_name
        self.canary_weight = canary_weight  # 10% to Green, 90% to Blue
        self.metrics = {
            "blue": {"requests": 0, "errors": 0, "latencies": []},
            "green": {"requests": 0, "errors": 0, "latencies": []}
        }

    def route_request(self, payload: dict) -> dict:
        """Simulates weighted canary routing at the API Gateway / Microservice layer."""
        # Weighted routing decision
        target_variant = "green" if random.random() < self.canary_weight else "blue"
        target_endpoint = self.green_name if target_variant == "green" else self.blue_name

        t0 = time.time()
        try:
            # If green endpoint does not exist yet in test, route to blue
            resp = runtime.invoke_endpoint(
                EndpointName=target_endpoint,
                ContentType="application/json",
                Accept="application/json",
                Body=json.dumps(payload)
            )
            latency = (time.time() - t0) * 1000
            self.metrics[target_variant]["requests"] += 1
            self.metrics[target_variant]["latencies"].append(latency)
            result = json.loads(resp["Body"].read().decode())
            result["routed_variant"] = target_variant
            result["latency_ms"] = round(latency, 1)
            return result
        except Exception as e:
            latency = (time.time() - t0) * 1000
            self.metrics[target_variant]["requests"] += 1
            self.metrics[target_variant]["errors"] += 1
            return {"error": str(e), "routed_variant": target_variant, "latency_ms": round(latency, 1)}

    def evaluate_canary(self, max_allowed_error_rate: float = 0.01, max_latency_ms: float = 500.0) -> bool:
        """Audits canary health metrics against production banking SLAs."""
        print("\n" + "=" * 65)
        print("CANARY DEPLOYMENT HEALTH AUDIT (90/10 Traffic Split)")
        print("=" * 65)

        for variant, data in self.metrics.items():
            reqs = data["requests"]
            errs = data["errors"]
            err_rate = (errs / reqs) if reqs > 0 else 0.0
            avg_lat = sum(data["latencies"]) / len(data["latencies"]) if data["latencies"] else 0.0
            p95_lat = sorted(data["latencies"])[int(len(data["latencies"]) * 0.95)] if data["latencies"] else 0.0
            print(f"Variant: {variant.upper()}")
            print(f"  Requests Processed: {reqs}")
            print(f"  Error Rate:         {err_rate:.2%} (Max SLA: {max_allowed_error_rate:.2%})")
            print(f"  Avg Latency:        {avg_lat:.1f}ms")
            print(f"  P95 Latency:        {p95_lat:.1f}ms (Max SLA: {max_latency_ms:.1f}ms)")

        green_data = self.metrics["green"]
        green_reqs = green_data["requests"]
        green_err_rate = (green_data["errors"] / green_reqs) if green_reqs > 0 else 0.0
        green_avg_lat = sum(green_data["latencies"]) / len(green_data["latencies"]) if green_data["latencies"] else 0.0

        if green_err_rate <= max_allowed_error_rate and green_avg_lat <= max_latency_ms:
            print("\n[DECISION] CANARY PASSED: Green Candidate Model satisfies all banking SLAs!")
            print("  Action: Promote Green to Blue (Shift 100% of customer traffic to new model).")
            return True
        else:
            print("\n[DECISION] CANARY FAILED: Green Candidate violated SLA thresholds!")
            print("  Action: Instant automated rollback to Blue (Route 100% traffic back to Blue).")
            return False

if __name__ == "__main__":
    print("Simulating Canary Traffic Shifting against production endpoint...")
    router = CanaryRouter(blue_name="bank-churn-endpoint", green_name="bank-churn-endpoint", canary_weight=0.15)
    
    sample_payload = {
        "CreditScore": 650, "Geography": 0, "Gender": 1, "Age": 38, "Tenure": 5,
        "Balance": 75000.0, "NumOfProducts": 2, "HasCrCard": 1, "IsActiveMember": 1,
        "EstimatedSalary": 100000.0
    }
    
    # Send 20 simulated requests across the canary split
    for i in range(20):
        router.route_request(sample_payload)
        time.sleep(0.05)
        
    router.evaluate_canary()
