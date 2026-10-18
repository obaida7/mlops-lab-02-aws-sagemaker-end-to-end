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

variable "github_token" {
  description = "GitHub Personal Access Token for triggering retraining workflow"
  type        = string
  sensitive   = true
  default     = "ghp_placeholder_token_for_dispatch"
}

variable "github_repo" {
  description = "GitHub repository (owner/repo)"
  type        = string
  default     = "obaida7/mlops-lab-02-aws-sagemaker-end-to-end"
}

variable "tags" {
  description = "Resource tags"
  type        = map(string)
  default     = {}
}
