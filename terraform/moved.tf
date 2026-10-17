# ════════════════════════════════════════════════════════════════════════
# TERRAFORM STATE REFACTORING (Zero Downtime / Zero Re-creation)
# Maps legacy root resources to modular paths
# ════════════════════════════════════════════════════════════════════════

# Storage Module
moved {
  from = aws_s3_bucket.sagemaker_bucket
  to   = module.storage.aws_s3_bucket.sagemaker_bucket
}

moved {
  from = aws_s3_bucket_public_access_block.sagemaker_bucket_pab
  to   = module.storage.aws_s3_bucket_public_access_block.sagemaker_bucket_pab
}

moved {
  from = aws_s3_object.bank_churn_dataset
  to   = module.storage.aws_s3_object.bank_churn_dataset
}

# Registry Module
moved {
  from = aws_sagemaker_model_package_group.churn_group
  to   = module.registry.aws_sagemaker_model_package_group.churn_group
}

# Feature Store Module
moved {
  from = aws_dynamodb_table.bank_customer_features
  to   = module.feature_store.aws_dynamodb_table.bank_customer_features
}

# Monitoring Module
moved {
  from = aws_sns_topic.drift_alerts
  to   = module.monitoring.aws_sns_topic.drift_alerts
}

moved {
  from = aws_sns_topic_policy.drift_alerts_policy
  to   = module.monitoring.aws_sns_topic_policy.drift_alerts_policy
}

moved {
  from = aws_cloudwatch_event_rule.model_monitor_drift
  to   = module.monitoring.aws_cloudwatch_event_rule.model_monitor_drift
}

moved {
  from = aws_cloudwatch_event_target.sns_target
  to   = module.monitoring.aws_cloudwatch_event_target.sns_target
}
