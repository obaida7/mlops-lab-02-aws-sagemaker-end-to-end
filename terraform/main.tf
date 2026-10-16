terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = "us-east-2"
}

# ════════════════════════════════════════════════════════════════════════
# 1. S3 BUCKET FOR DATA, MODELS, METRICS & CAPTURED INFERENCE TRAFFIC
# ════════════════════════════════════════════════════════════════════════
resource "aws_s3_bucket" "sagemaker_bucket" {
  bucket        = "sagemaker-us-east-2-227246780492-bank-churn"
  force_destroy = true

  tags = {
    Project     = "BankCustomerChurn"
    Environment = "MLOps"
  }
}

resource "aws_s3_bucket_public_access_block" "sagemaker_bucket_pab" {
  bucket = aws_s3_bucket.sagemaker_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_object" "bank_churn_dataset" {
  bucket       = aws_s3_bucket.sagemaker_bucket.id
  key          = "data/bank_churn.csv"
  source       = "${path.module}/bank_churn.csv"
  content_type = "text/csv"
  depends_on   = [aws_s3_bucket.sagemaker_bucket]
}

# ════════════════════════════════════════════════════════════════════════
# 2. SAGEMAKER MODEL REGISTRY (MODEL PACKAGE GROUP)
# ════════════════════════════════════════════════════════════════════════
resource "aws_sagemaker_model_package_group" "churn_group" {
  model_package_group_name        = "bank-customer-churn"
  model_package_group_description = "Banking customer churn prediction models adhering to SR 11-7 model risk management"

  tags = {
    Project     = "BankCustomerChurn"
    Compliance  = "SR-11-7"
  }
}

# ════════════════════════════════════════════════════════════════════════
# 3. EVENTBRIDGE RULE FOR MODEL MONITOR DRIFT ALERTS
# ════════════════════════════════════════════════════════════════════════
resource "aws_cloudwatch_event_rule" "model_monitor_drift" {
  name        = "bank-churn-model-monitor-drift-rule"
  description = "Triggers alerts and automated workflows on SageMaker Model Monitor violations"

  event_pattern = jsonencode({
    source        = ["aws.sagemaker"]
    "detail-type" = ["SageMaker Model Monitor Schedule Status"]
    detail = {
      MonitorScheduleStatus = ["CompletedWithViolations"]
    }
  })

  tags = {
    Project = "BankCustomerChurn"
  }
}

# SNS Topic for drift alerting
resource "aws_sns_topic" "drift_alerts" {
  name = "bank-churn-drift-alerts"

  tags = {
    Project = "BankCustomerChurn"
  }
}

resource "aws_sns_topic_policy" "drift_alerts_policy" {
  arn = aws_sns_topic.drift_alerts.arn

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "AllowEventBridgePublish"
        Effect    = "Allow"
        Principal = {
          Service = "events.amazonaws.com"
        }
        Action   = "sns:Publish"
        Resource = aws_sns_topic.drift_alerts.arn
      }
    ]
  })
}

resource "aws_cloudwatch_event_target" "sns_target" {
  rule      = aws_cloudwatch_event_rule.model_monitor_drift.name
  target_id = "SendDriftAlertToSNS"
  arn       = aws_sns_topic.drift_alerts.arn
}

# ════════════════════════════════════════════════════════════════════════
# 4. OUTPUTS
# ════════════════════════════════════════════════════════════════════════
output "s3_bucket" {
  description = "The S3 bucket for datasets, model artifacts, and monitoring outputs"
  value       = aws_s3_bucket.sagemaker_bucket.bucket
}

output "model_package_group" {
  description = "The SageMaker Model Package Group name for versioned models"
  value       = aws_sagemaker_model_package_group.churn_group.model_package_group_name
}

output "drift_alerts_sns_arn" {
  description = "SNS topic ARN for Model Monitor drift violations"
  value       = aws_sns_topic.drift_alerts.arn
}
