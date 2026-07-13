
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ECR Repository for training image
resource "aws_ecr_repository" "training" {
  name = "chest-ct-training"
  force_delete = true
}

# ============================================
# IAM Role for Batch Service
# ============================================
resource "aws_iam_role" "batch_service_role" {
  name = "chest-ct-batch-service-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "batch.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "batch_service_role_policy" {
  role       = aws_iam_role.batch_service_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

resource "aws_iam_role_policy_attachment" "batch_service_ecs_full" {
  role       = aws_iam_role.batch_service_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonECS_FullAccess"
}

resource "aws_iam_role_policy_attachment" "batch_service_cloudwatch" {
  role       = aws_iam_role.batch_service_role.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
}

# ============================================
# IAM Role for FARGATE tasks
# ============================================
resource "aws_iam_role" "batch_role" {
  name = "chest-ct-batch-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "batch_role_ecs_full" {
  role       = aws_iam_role.batch_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonECS_FullAccess"
}

resource "aws_iam_role_policy_attachment" "batch_s3" {
  role       = aws_iam_role.batch_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_role_policy_attachment" "batch_ecr" {
  role       = aws_iam_role.batch_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_role_policy_attachment" "batch_logs" {
  role       = aws_iam_role.batch_role.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
}

# Attach Secrets Manager policy
resource "aws_iam_role_policy_attachment" "batch_secrets_manager" {
  role       = aws_iam_role.batch_role.name
  policy_arn = aws_iam_policy.batch_secrets_access.arn
}

# ============================================
# Security Group
# ============================================
resource "aws_security_group" "batch" {
  name        = "chest-ct-batch-sg"
  description = "Security group for Batch compute"
  vpc_id      = var.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ============================================
# Batch Compute Environment (FARGATE)
# ============================================
resource "aws_batch_compute_environment" "training" {
  compute_environment_name = "chest-ct-training-fargate"
  type                     = "MANAGED"
  service_role             = aws_iam_role.batch_service_role.arn

  compute_resources {
    type                = "FARGATE"
    max_vcpus           = 256
    subnets             = var.subnet_ids
    security_group_ids  = [aws_security_group.batch.id]
  }
}

# ============================================
# Batch Job Queue
# ============================================
resource "aws_batch_job_queue" "training" {
  name     = "chest-ct-training-queue"
  state    = "ENABLED"
  priority = 1
  
  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.training.arn
  }
}

# ============================================
# Batch Job Definition
# ============================================
resource "aws_batch_job_definition" "retraining" {
  name = "chest-ct-retraining"
  type = "container"
  
  platform_capabilities = ["FARGATE"]

  container_properties = jsonencode({
    image = "${aws_ecr_repository.training.repository_url}:latest"
    
    command = ["python", "retrain.py"]
    
    resourceRequirements = [
      { type = "VCPU", value = tostring(var.job_vcpus) },
      { type = "MEMORY", value = tostring(var.job_memory) }
    ]
    
    environment = [
      { name = "AWS_REGION", value = var.aws_region },
      { name = "JOB_NAME", value = "ecs-cicd-d" }
    ]
    
    secrets = [
      {
        name      = "MLFLOW_TRACKING_URI"
        valueFrom = "${data.aws_secretsmanager_secret.mlflow_credentials.arn}:tracking_uri::"
      },
      {
        name      = "MLFLOW_TRACKING_USERNAME"
        valueFrom = "${data.aws_secretsmanager_secret.mlflow_credentials.arn}:username::"
      },
      {
        name      = "MLFLOW_TRACKING_PASSWORD"
        valueFrom = "${data.aws_secretsmanager_secret.mlflow_credentials.arn}:password::"
      },
      {
        name      = "MODEL_BUCKET"
        valueFrom = "${data.aws_secretsmanager_secret.s3_credentials.arn}:models_bucket::"
      },
      {
        name      = "DATA_BUCKET"
        valueFrom = "${data.aws_secretsmanager_secret.s3_credentials.arn}:data_bucket::"
      },
      {
        name      = "JENKINS_URL"
        valueFrom = "${data.aws_secretsmanager_secret.jenkins_credentials.arn}:url::"
      },
      {
        name      = "JENKINS_TOKEN"
        valueFrom = "${data.aws_secretsmanager_secret.jenkins_credentials.arn}:token::"
      },
      {
        name      = "JENKINS_USERNAME"
        valueFrom = "${data.aws_secretsmanager_secret.jenkins_credentials.arn}:username::"
      },
      {
        name      = "JENKINS_API_TOKEN"
        valueFrom = "${data.aws_secretsmanager_secret.jenkins_credentials.arn}:api_token::"
      }
    ]
    
    executionRoleArn = aws_iam_role.batch_role.arn
    jobRoleArn = aws_iam_role.batch_role.arn
    
    networkConfiguration = {
      assignPublicIp = "ENABLED"
    }
    
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group" = aws_cloudwatch_log_group.batch.name
        "awslogs-region" = var.aws_region
      }
    }
  })
}

# ============================================
# CloudWatch Log Group
# ============================================
resource "aws_cloudwatch_log_group" "batch" {
  name = "/aws/batch/chest-ct-retraining"
  retention_in_days = 30
}

# ============================================
# Daily Schedule at 2 PM IST (8:30 AM UTC)
# ============================================
resource "aws_cloudwatch_event_rule" "daily_retraining" {
  name                = "chest-ct-daily-retraining"
  description         = "Trigger retraining daily at 2 PM IST (8:30 AM UTC)"
  schedule_expression = "cron(30 8 * * ? *)"
}

resource "aws_iam_role" "eventbridge_role" {
  name = "chest-ct-eventbridge-role"
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

resource "aws_iam_role_policy" "eventbridge_policy" {
  name = "chest-ct-eventbridge-policy"
  role = aws_iam_role.eventbridge_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = "batch:SubmitJob"
        Resource = "*"
      }
    ]
  })
}

resource "aws_cloudwatch_event_target" "batch_job" {
  rule      = aws_cloudwatch_event_rule.daily_retraining.name
  target_id = "SubmitBatchJob"
  arn       = aws_batch_job_queue.training.arn

  batch_target {
    job_name       = "chest-ct-retraining-${formatdate("YYYYMMDD-HHmm", timestamp())}"
    job_definition = aws_batch_job_definition.retraining.arn
  }

  role_arn = aws_iam_role.eventbridge_role.arn
}