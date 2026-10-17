# ════════════════════════════════════════════════════════════════════════
# ROOT MODULE OUTPUTS
# ════════════════════════════════════════════════════════════════════════

output "s3_bucket" {
  description = "The S3 bucket for datasets, model artifacts, and monitoring outputs"
  value       = module.storage.bucket_id
}

output "model_package_group" {
  description = "The SageMaker Model Package Group name for versioned models"
  value       = module.registry.model_package_group_name
}

output "dynamodb_feature_store" {
  description = "DynamoDB Table Name for customer feature store"
  value       = module.feature_store.table_name
}

output "drift_alerts_sns_arn" {
  description = "SNS topic ARN for Model Monitor and SLA alerts"
  value       = module.monitoring.sns_topic_arn
}

output "cloudwatch_dashboard" {
  description = "CloudWatch Executive MLOps Dashboard Name"
  value       = module.monitoring.dashboard_name
}

output "cloudwatch_alarms" {
  description = "List of CloudWatch metric alarms provisioned"
  value       = module.monitoring.alarm_names
}
