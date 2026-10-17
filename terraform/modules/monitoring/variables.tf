variable "sns_topic_name" {
  description = "SNS topic name for Model Monitor drift alerts"
  type        = string
  default     = "bank-churn-drift-alerts"
}

variable "endpoint_name" {
  description = "Target SageMaker endpoint name to monitor"
  type        = string
  default     = "bank-churn-endpoint"
}

variable "dashboard_name" {
  description = "CloudWatch Executive MLOps Dashboard name"
  type        = string
  default     = "BankChurn-MLOps-Dashboard"
}

variable "tags" {
  description = "Resource tags"
  type        = map(string)
  default     = {}
}
