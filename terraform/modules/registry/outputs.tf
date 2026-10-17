output "model_package_group_name" {
  description = "The SageMaker Model Package Group name"
  value       = aws_sagemaker_model_package_group.churn_group.model_package_group_name
}

output "model_package_group_arn" {
  description = "The SageMaker Model Package Group ARN"
  value       = aws_sagemaker_model_package_group.churn_group.arn
}
