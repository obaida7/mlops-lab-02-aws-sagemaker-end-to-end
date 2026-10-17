output "bucket_id" {
  description = "The S3 bucket name"
  value       = aws_s3_bucket.sagemaker_bucket.id
}

output "bucket_arn" {
  description = "The S3 bucket ARN"
  value       = aws_s3_bucket.sagemaker_bucket.arn
}
