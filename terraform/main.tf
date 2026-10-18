terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }

  backend "s3" {
    bucket = "sagemaker-us-east-2-227246780492"
    key    = "terraform/state/bank-churn.tfstate"
    region = "us-east-2"
  }
}

provider "aws" {
  region = var.aws_region
}

# ════════════════════════════════════════════════════════════════════════
# 1. STORAGE MODULE (S3 Bucket, Security, Baseline CSV)
# ════════════════════════════════════════════════════════════════════════
module "storage" {
  source              = "./modules/storage"
  bucket_name         = var.bucket_name
  dataset_source_path = "${path.module}/bank_churn.csv"
  tags                = var.tags
}

# ════════════════════════════════════════════════════════════════════════
# 2. MODEL REGISTRY MODULE (SageMaker Model Package Group)
# ════════════════════════════════════════════════════════════════════════
module "registry" {
  source      = "./modules/registry"
  group_name  = var.model_package_group_name
  description = "Banking customer churn prediction models adhering to SR 11-7 model risk management"
  tags        = var.tags
}

# ════════════════════════════════════════════════════════════════════════
# 3. ONLINE FEATURE STORE MODULE (DynamoDB Real-Time Features)
# ════════════════════════════════════════════════════════════════════════
module "feature_store" {
  source       = "./modules/feature_store"
  table_name   = var.feature_store_table_name
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "CustomerId"
  tags         = var.tags
}

# ════════════════════════════════════════════════════════════════════════
# 4. MONITORING & OBSERVABILITY MODULE (Alarms, Dashboard, SNS, EventBridge)
# ════════════════════════════════════════════════════════════════════════
module "monitoring" {
  source         = "./modules/monitoring"
  sns_topic_name = var.sns_topic_name
  endpoint_name  = var.endpoint_name
  dashboard_name = var.dashboard_name
  github_token   = var.github_token
  tags           = var.tags
}
