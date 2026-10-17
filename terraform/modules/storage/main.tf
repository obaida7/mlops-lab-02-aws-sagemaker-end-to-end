resource "aws_s3_bucket" "sagemaker_bucket" {
  bucket        = var.bucket_name
  force_destroy = true

  tags = var.tags
}

resource "aws_s3_bucket_public_access_block" "sagemaker_bucket_pab" {
  bucket = aws_s3_bucket.sagemaker_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_object" "bank_churn_dataset" {
  bucket       = aws_s3_bucket.sagemaker_bucket.id
  key          = "data/bank_churn.csv"
  source       = var.dataset_source_path
  content_type = "text/csv"
  depends_on   = [aws_s3_bucket.sagemaker_bucket]
}
