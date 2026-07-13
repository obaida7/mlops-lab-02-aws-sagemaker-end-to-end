terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

# Fetch the existing sandbox VPC dynamically
data "aws_vpcs" "all" {}

data "aws_vpc" "main" {
  id = tolist(data.aws_vpcs.all.ids)[0]
}

data "aws_route_table" "main_rt" {
  vpc_id = data.aws_vpc.main.id
}

resource "aws_internet_gateway" "igw" {
  vpc_id = data.aws_vpc.main.id
  tags = {
    Name = "mlflow-igw"
  }
}

resource "aws_route" "internet_access" {
  route_table_id         = data.aws_route_table.main_rt.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.igw.id
}

# Fetch availability zones
data "aws_availability_zones" "available" {
  state = "available"
}

# Fetch the existing subnet in AZ a
data "aws_subnets" "public" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.main.id]
  }
  filter {
    name   = "availability-zone"
    values = [data.aws_availability_zones.available.names[0]]
  }
}

data "aws_subnet" "public" {
  id = tolist(data.aws_subnets.public.ids)[0]
}

# Create a second subnet for RDS (requires 2 subnets in different AZs)
resource "aws_subnet" "rds_subnet_2" {
  vpc_id            = data.aws_vpc.main.id
  cidr_block        = "10.192.40.0/24"
  availability_zone = data.aws_availability_zones.available.names[1]

  tags = {
    Name = "rds-subnet-v3"
  }
}

resource "aws_db_subnet_group" "mlflow_db_subnet" {
  name       = "mlflow-db-subnet-group-v3"
  subnet_ids = [data.aws_subnet.public.id, aws_subnet.rds_subnet_2.id]

  tags = {
    Name = "MLflow DB Subnet Group V3"
  }
}

# Fetch latest Amazon Linux 2023 AMI
data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-x86_64"]
  }
}

# ════════════════════════════════════════════════════════════════════════
# 1. S3 BUCKET & INITIAL DATA
# ════════════════════════════════════════════════════════════════════════
resource "aws_s3_bucket" "sagemaker_bucket" {
  bucket        = "sagemaker-us-east-1-339712725413-v3"
  force_destroy = true
}

resource "aws_s3_object" "wine_dataset" {
  bucket       = aws_s3_bucket.sagemaker_bucket.id
  key          = "data/wine.csv"
  source       = "${path.module}/wine.csv"
  content_type = "text/csv"
  depends_on   = [aws_s3_bucket.sagemaker_bucket]
}

# ════════════════════════════════════════════════════════════════════════
# 2. SECURITY GROUPS
# ════════════════════════════════════════════════════════════════════════
resource "aws_security_group" "mlflow_sg" {
  name        = "mlflow-ec2-sg-v3"
  description = "Allow inbound SSH and MLflow UI"
  vpc_id      = data.aws_vpc.main.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "MLflow UI"
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "rds_sg" {
  name        = "mlflow-rds-sg-v3"
  description = "Allow inbound traffic from MLflow EC2"
  vpc_id      = data.aws_vpc.main.id

  ingress {
    description     = "PostgreSQL from EC2"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.mlflow_sg.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ════════════════════════════════════════════════════════════════════════
# 3. RDS DATABASE
# ════════════════════════════════════════════════════════════════════════
resource "aws_db_instance" "mlflow_db" {
  identifier             = "mlflow-db-v3"
  allocated_storage      = 20
  engine                 = "postgres"
  engine_version         = "15"
  instance_class         = "db.t3.micro"
  db_name                = "mlflow"
  username               = "mlflow_user"
  password               = "mlflow_password123"
  parameter_group_name   = "default.postgres15"
  skip_final_snapshot    = true
  publicly_accessible    = false
  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.mlflow_db_subnet.name
}

# ════════════════════════════════════════════════════════════════════════
# 4. EC2 INSTANCE (MLFLOW TRACKING SERVER)
# ════════════════════════════════════════════════════════════════════════
resource "aws_iam_role" "mlflow_ec2_role" {
  name = "MLflowEC2RoleV3"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "mlflow_s3_access" {
  role       = aws_iam_role.mlflow_ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_role_policy_attachment" "mlflow_ssm_access" {
  role       = aws_iam_role.mlflow_ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "mlflow_profile" {
  name = "MLflowEC2InstanceProfileV3"
  role = aws_iam_role.mlflow_ec2_role.name
}

resource "aws_instance" "mlflow_server" {
  ami                         = data.aws_ami.amazon_linux.id
  instance_type               = "t3.small"
  subnet_id                   = data.aws_subnet.public.id
  vpc_security_group_ids      = [aws_security_group.mlflow_sg.id]
  iam_instance_profile        = aws_iam_instance_profile.mlflow_profile.name
  associate_public_ip_address = true
  user_data_replace_on_change = true

  user_data = <<-EOF
              #!/bin/bash
              # Force rebuild for IGW fix 2
              sudo yum update -y
              sudo yum install -y python3 python3-pip
              
              # Create a virtual environment to avoid PEP 668 externally-managed-environment error
              python3 -m venv /opt/mlflow_env
              /opt/mlflow_env/bin/pip install mlflow boto3 psycopg2-binary
              
              cat <<'EOT' > /etc/systemd/system/mlflow.service
              [Unit]
              Description=MLflow Tracking Server
              After=network.target

              [Service]
              User=ec2-user
              ExecStart=/opt/mlflow_env/bin/mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri postgresql://${aws_db_instance.mlflow_db.username}:${aws_db_instance.mlflow_db.password}@${aws_db_instance.mlflow_db.endpoint}/${aws_db_instance.mlflow_db.db_name} --default-artifact-root s3://${aws_s3_bucket.sagemaker_bucket.bucket}/artifacts
              Restart=always

              [Install]
              WantedBy=multi-user.target
              EOT
              
              sudo systemctl daemon-reload
              sudo systemctl enable mlflow
              sudo systemctl start mlflow
              EOF

  tags = {
    Name = "MLflow-Tracking-Server"
  }
}

output "mlflow_ui_url" {
  value = "http://${aws_instance.mlflow_server.public_dns}:5000"
}

# ════════════════════════════════════════════════════════════════════════
# 5. EVENTBRIDGE RULE FOR MODEL MONITOR DRIFT
# ════════════════════════════════════════════════════════════════════════
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

resource "aws_cloudwatch_event_rule" "model_monitor_drift" {
  name        = "wine-model-monitor-drift-rule"
  description = "Triggers retraining on model monitor violations"

  event_pattern = jsonencode({
    source = ["aws.sagemaker"]
    "detail-type" = ["SageMaker Model Monitor Schedule Status"]
    detail = {
      MonitorScheduleStatus = ["CompletedWithViolations"]
    }
  })
}

resource "aws_iam_role" "eventbridge_sagemaker_role" {
  name = "EventBridgeSageMakerPipelineRoleV3"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "events.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_cloudwatch_event_connection" "github" {
  name               = "github-connection"
  description        = "Connection to GitHub API"
  authorization_type = "API_KEY"

  auth_parameters {
    api_key {
      key   = "Authorization"
      value = "Bearer PLACEHOLDER_PAT_REMOVED_FOR_SECURITY"
    }
  }
}

resource "aws_cloudwatch_event_api_destination" "github_actions" {
  name                             = "github-actions-trigger"
  description                      = "Trigger GitHub Actions Pipeline"
  invocation_endpoint              = "https://api.github.com/repos/obaida7/mlops-lab-02-aws-sagemaker-end-to-end/actions/workflows/end-to-end.yml/dispatches"
  http_method                      = "POST"
  connection_arn                   = aws_cloudwatch_event_connection.github.arn
}

resource "aws_iam_role_policy" "eventbridge_sagemaker_policy" {
  name = "InvokeApiDestinationPolicy"
  role = aws_iam_role.eventbridge_sagemaker_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "events:InvokeApiDestination"
        ]
        Effect   = "Allow"
        Resource = aws_cloudwatch_event_api_destination.github_actions.arn
      }
    ]
  })
}

resource "aws_cloudwatch_event_target" "trigger_pipeline" {
  rule      = aws_cloudwatch_event_rule.model_monitor_drift.name
  target_id = "TriggerGitHubActionsPipeline"
  arn       = aws_cloudwatch_event_api_destination.github_actions.arn
  role_arn  = aws_iam_role.eventbridge_sagemaker_role.arn

  http_target {
    header_parameters = {
      "Accept" = "application/vnd.github.v3+json"
    }
  }

  input = jsonencode({
    ref = "main"
  })
}
