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

# ============================================================
# ECR Repository for training image
# ============================================================
resource "aws_ecr_repository" "training" {
  name = "chest-ct-training"
  force_delete = true
  
  image_scanning_configuration {
    scan_on_push = true
  }
}

# ============================================================
# IAM Role for Batch Service
# ============================================================
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

# ============================================================
# IAM Role for FARGATE tasks
# ============================================================
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

resource "aws_iam_policy" "ecs_list_policy" {
  name        = "BatchECSListPolicy"
  description = "Allow Batch to list ECS clusters and resources"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecs:ListClusters",
          "ecs:DescribeClusters",
          "ecs:ListContainerInstances",
          "ecs:DescribeContainerInstances",
          "ecs:ListTasks",
          "ecs:DescribeTasks",
          "ecs:RegisterContainerInstance",
          "ecs:DeregisterContainerInstance"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "batch_service_ecs_list" {
  role       = aws_iam_role.batch_service_role.name
  policy_arn = aws_iam_policy.ecs_list_policy.arn
}

resource "aws_iam_role_policy_attachment" "batch_role_ecs_list" {
  role       = aws_iam_role.batch_role.name
  policy_arn = aws_iam_policy.ecs_list_policy.arn
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

# ============================================================
# Security Group
# ============================================================
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

# ============================================================
# Batch Compute Environment (FARGATE)
# ============================================================
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

# ============================================================
# Batch Job Queue
# ============================================================
resource "aws_batch_job_queue" "training" {
  name     = "chest-ct-training-queue"
  state    = "ENABLED"
  priority = 1
  
  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.training.arn
  }
}

# ============================================================
# Batch Job Definition
# ============================================================
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

# ============================================================
# CloudWatch Log Group
# ============================================================
resource "aws_cloudwatch_log_group" "batch" {
  name = "/aws/batch/chest-ct-retraining"
  retention_in_days = 30
}

# ============================================================
# SNS TOPIC FOR ALERTS
# ============================================================
resource "aws_sns_topic" "alerts" {
  name = "chest-ct-alerts"
  
  tags = {
    Environment = "production"
    Project     = "chest-ct-mlops"
  }
}

resource "aws_sns_topic_subscription" "email_alert" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# ============================================================
# CLOUDWATCH ALARMS
# ============================================================

# Alarm for model performance drop
resource "aws_cloudwatch_metric_alarm" "model_performance_drop" {
  alarm_name          = "chest-ct-model-performance-drop"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "3"
  metric_name         = "ModelAccuracy"
  namespace           = "ChestCT/MLOps"
  period              = "300"
  statistic           = "Average"
  threshold           = "0.85"
  alarm_description   = "Alert when model accuracy drops below 85%"
  
  alarm_actions = [aws_sns_topic.alerts.arn]
  
  tags = {
    Environment = "production"
    Project     = "chest-ct-mlops"
  }
}

# Alarm for high drift
resource "aws_cloudwatch_metric_alarm" "high_drift" {
  alarm_name          = "chest-ct-high-drift"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "DriftPercentage"
  namespace           = "ChestCT/MLOps"
  period              = "300"
  statistic           = "Average"
  threshold           = "30"
  alarm_description   = "Alert when data drift exceeds 30%"
  
  alarm_actions = [aws_sns_topic.alerts.arn]
  
  tags = {
    Environment = "production"
    Project     = "chest-ct-mlops"
  }
}

# Alarm for Batch job failures
resource "aws_cloudwatch_metric_alarm" "batch_job_failure" {
  alarm_name          = "chest-ct-batch-job-failure"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "BatchJobFailed"
  namespace           = "AWS/Batch"
  period              = "300"
  statistic           = "Sum"
  threshold           = "0"
  alarm_description   = "Alert when Batch job fails"
  
  alarm_actions = [aws_sns_topic.alerts.arn]
  
  tags = {
    Environment = "production"
    Project     = "chest-ct-mlops"
  }
}

# ============================================================
# EVENTBRIDGE RULES FOR EVENT-DRIVEN RETRAINING
# ============================================================

# 1. DRIFT DETECTION EVENT (S3 → EventBridge → Batch)
resource "aws_cloudwatch_event_rule" "drift_trigger" {
  name        = "chest-ct-drift-trigger"
  description = "Trigger retraining when drift is detected"

  event_pattern = jsonencode({
    source = ["aws.s3"]
    detail-type = ["Object Created"]
    detail = {
      bucket = {
        name = ["chest-ct-models-155407238004"]
      }
      object = {
        key = [{
          prefix = "drift_reports/"
        }]
      }
    }
  })
}

resource "aws_cloudwatch_event_target" "drift_batch" {
  rule      = aws_cloudwatch_event_rule.drift_trigger.name
  target_id = "TriggerRetrainingOnDrift"
  arn       = aws_batch_job_queue.training.arn

  batch_target {
    job_name       = "chest-ct-retraining-drift-${formatdate("YYYYMMDD-HHmm", timestamp())}"
    job_definition = aws_batch_job_definition.retraining.arn
  }

  role_arn = aws_iam_role.eventbridge_role.arn
}

# 2. NEW DATA EVENT (S3 Upload → EventBridge → Batch)
resource "aws_cloudwatch_event_rule" "new_data_trigger" {
  name        = "chest-ct-new-data-trigger"
  description = "Trigger retraining when new data is uploaded"

  event_pattern = jsonencode({
    source = ["aws.s3"]
    detail-type = ["Object Created"]
    detail = {
      bucket = {
        name = ["chest-models-gajju"]
      }
      object = {
        key = [{
          prefix = "chest-data.zip"
        }]
      }
    }
  })
}

resource "aws_cloudwatch_event_target" "new_data_batch" {
  rule      = aws_cloudwatch_event_rule.new_data_trigger.name
  target_id = "TriggerRetrainingOnNewData"
  arn       = aws_batch_job_queue.training.arn

  batch_target {
    job_name       = "chest-ct-retraining-newdata-${formatdate("YYYYMMDD-HHmm", timestamp())}"
    job_definition = aws_batch_job_definition.retraining.arn
  }

  role_arn = aws_iam_role.eventbridge_role.arn
}

# 3. WEEKLY SCHEDULE (Fallback - every Sunday at 2 AM)
resource "aws_cloudwatch_event_rule" "weekly_retraining" {
  name                = "chest-ct-weekly-retraining"
  description         = "Weekly retraining as a fallback (every Sunday at 2 AM UTC)"
  schedule_expression = "cron(0 2 ? * SUN *)"
}

resource "aws_cloudwatch_event_target" "weekly_batch" {
  rule      = aws_cloudwatch_event_rule.weekly_retraining.name
  target_id = "TriggerWeeklyRetraining"
  arn       = aws_batch_job_queue.training.arn

  batch_target {
    job_name       = "chest-ct-retraining-weekly-${formatdate("YYYYMMDD-HHmm", timestamp())}"
    job_definition = aws_batch_job_definition.retraining.arn
  }

  role_arn = aws_iam_role.eventbridge_role.arn
}

# 4. PERFORMANCE TRIGGER (CloudWatch Alarm → EventBridge → Batch)
resource "aws_cloudwatch_event_rule" "performance_trigger" {
  name        = "chest-ct-performance-trigger"
  description = "Trigger retraining when performance drops"

  event_pattern = jsonencode({
    source = ["aws.cloudwatch"]
    detail-type = ["CloudWatch Alarm State Change"]
    detail = {
      alarmName = [aws_cloudwatch_metric_alarm.model_performance_drop.alarm_name]
      state = {
        value = ["ALARM"]
      }
    }
  })
}

resource "aws_cloudwatch_event_target" "performance_batch" {
  rule      = aws_cloudwatch_event_rule.performance_trigger.name
  target_id = "TriggerRetrainingOnPerformance"
  arn       = aws_batch_job_queue.training.arn

  batch_target {
    job_name       = "chest-ct-retraining-perf-${formatdate("YYYYMMDD-HHmm", timestamp())}"
    job_definition = aws_batch_job_definition.retraining.arn
  }

  role_arn = aws_iam_role.eventbridge_role.arn
}

# ============================================================
# IAM Role for EventBridge
# ============================================================
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

# ============================================================
# Data Sources for Secrets Manager
# ============================================================
data "aws_secretsmanager_secret" "mlflow_credentials" {
  name = "chest-ct/mlflow/credentials"
}

data "aws_secretsmanager_secret" "s3_credentials" {
  name = "chest-ct/s3/credentials"
}

data "aws_secretsmanager_secret" "jenkins_credentials" {
  name = "chest-ct/jenkins/credentials"
}

data "aws_secretsmanager_secret" "deployment_credentials" {
  name = "chest-ct/deployment/credentials"
}

# ============================================================
# IAM Policy for Batch to access Secrets Manager
# ============================================================
resource "aws_iam_policy" "batch_secrets_access" {
  name        = "chest-ct-batch-secrets-access"
  description = "Allow Batch jobs to access Secrets Manager"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = [
          data.aws_secretsmanager_secret.mlflow_credentials.arn,
          data.aws_secretsmanager_secret.s3_credentials.arn,
          data.aws_secretsmanager_secret.jenkins_credentials.arn,
          data.aws_secretsmanager_secret.deployment_credentials.arn
        ]
      }
    ]
  })
}