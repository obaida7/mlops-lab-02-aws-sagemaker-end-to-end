resource "aws_dynamodb_table" "bank_customer_features" {
  name         = var.table_name
  billing_mode = var.billing_mode
  hash_key     = var.hash_key

  attribute {
    name = var.hash_key
    type = "N"
  }

  tags = var.tags
}
