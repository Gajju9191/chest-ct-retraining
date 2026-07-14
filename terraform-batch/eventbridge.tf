
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

# 4. PERFORMANCE MONITORING (CloudWatch Alarm → EventBridge → Batch)
resource "aws_cloudwatch_metric_alarm" "model_performance_alarm" {
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
}

resource "aws_cloudwatch_event_rule" "performance_trigger" {
  name        = "chest-ct-performance-trigger"
  description = "Trigger retraining when performance drops"

  event_pattern = jsonencode({
    source = ["aws.sns"]
    detail-type = ["SNS Notification"]
    detail = {
      Subject = [{
        prefix = "ALARM"
      }]
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