variable "group_name" {
  description = "SageMaker Model Package Group name"
  type        = string
  default     = "bank-customer-churn"
}

variable "description" {
  description = "Description of the Model Package Group"
  type        = string
  default     = "Banking customer churn prediction models adhering to SR 11-7 model risk management"
}

variable "tags" {
  description = "Resource tags"
  type        = map(string)
  default     = {}
}
