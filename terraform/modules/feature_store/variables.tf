variable "table_name" {
  description = "Name of the DynamoDB feature store table"
  type        = string
  default     = "bank_customer_features"
}

variable "hash_key" {
  description = "Partition key attribute name"
  type        = string
  default     = "CustomerId"
}

variable "billing_mode" {
  description = "DynamoDB billing mode (PAY_PER_REQUEST or PROVISIONED)"
  type        = string
  default     = "PAY_PER_REQUEST"
}

variable "tags" {
  description = "Resource tags"
  type        = map(string)
  default     = {}
}
