variable "aws_region" {
  description = "Target AWS Region for MLOps infrastructure"
  type        = string
  default     = "us-east-2"
}

variable "bucket_name" {
  description = "S3 bucket for training data, model artifacts, and monitoring outputs"
  type        = string
  default     = "sagemaker-us-east-2-227246780492-bank-churn"
}

variable "model_package_group_name" {
  description = "SageMaker Model Package Group name for versioned models"
  type        = string
  default     = "bank-customer-churn"
}

variable "feature_store_table_name" {
  description = "DynamoDB table name for customer feature store"
  type        = string
  default     = "bank_customer_features"
}

variable "sns_topic_name" {
  description = "SNS topic name for Model Monitor and SLA alerts"
  type        = string
  default     = "bank-churn-drift-alerts"
}

variable "endpoint_name" {
  description = "SageMaker endpoint name monitored by CloudWatch alarms and dashboards"
  type        = string
  default     = "bank-churn-endpoint"
}

variable "dashboard_name" {
  description = "Name of the CloudWatch Executive MLOps Dashboard"
  type        = string
  default     = "BankChurn-MLOps-Dashboard"
}

variable "tags" {
  description = "Default resource tags applied across all MLOps modules"
  type        = map(string)
  default = {
    Project     = "BankCustomerChurn"
    Environment = "MLOps"
    ManagedBy   = "Terraform"
    Compliance  = "SR-11-7"
  }
}
