resource "aws_sagemaker_model_package_group" "churn_group" {
  model_package_group_name        = var.group_name
  model_package_group_description = var.description

  tags = var.tags
}
