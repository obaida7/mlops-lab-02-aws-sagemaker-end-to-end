aws_region               = "us-east-2"
bucket_name              = "sagemaker-us-east-2-227246780492-bank-churn"
model_package_group_name = "bank-customer-churn"
feature_store_table_name = "bank_customer_features"
sns_topic_name           = "bank-churn-drift-alerts"
endpoint_name            = "bank-churn-endpoint"
dashboard_name           = "BankChurn-MLOps-Dashboard"

tags = {
  Project     = "BankCustomerChurn"
  Environment = "MLOps"
  ManagedBy   = "Terraform"
  Compliance  = "SR-11-7"
}
