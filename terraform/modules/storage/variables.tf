variable "bucket_name" {
  description = "Name of the S3 bucket for SageMaker data and artifacts"
  type        = string
}

variable "dataset_source_path" {
  description = "Path to the local baseline CSV dataset"
  type        = string
}

variable "tags" {
  description = "Resource tags"
  type        = map(string)
  default     = {}
}
