import {
  to = module.storage.aws_s3_bucket.sagemaker_bucket
  id = "sagemaker-us-east-2-227246780492-bank-churn"
}

import {
  to = module.feature_store.aws_dynamodb_table.bank_customer_features
  id = "bank_customer_features"
}

import {
  to = module.registry.aws_sagemaker_model_package_group.churn_group
  id = "bank-customer-churn"
}

import {
  to = module.monitoring.aws_sns_topic.drift_alerts
  id = "arn:aws:sns:us-east-2:227246780492:bank-churn-drift-alerts"
}

import {
  to = module.monitoring.aws_cloudwatch_event_rule.model_monitor_drift
  id = "bank-churn-model-monitor-drift-rule"
}
