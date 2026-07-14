# ============================================================
# AWS BATCH OUTPUTS
# ============================================================

output "ecr_repository_url" {
  description = "ECR repository URL for the training image"
  value = aws_ecr_repository.training.repository_url
}

output "batch_job_queue" {
  description = "AWS Batch job queue name"
  value = aws_batch_job_queue.training.name
}

output "batch_job_definition" {
  description = "AWS Batch job definition name"
  value = aws_batch_job_definition.retraining.name
}

output "batch_job_definition_revision" {
  description = "AWS Batch job definition revision number"
  value = aws_batch_job_definition.retraining.revision
}

# ============================================================
# EVENTBRIDGE OUTPUTS
# ============================================================

output "eventbridge_rules" {
  description = "EventBridge rule names for retraining triggers"
  value = {
    drift_trigger       = aws_cloudwatch_event_rule.drift_trigger.name
    new_data_trigger    = aws_cloudwatch_event_rule.new_data_trigger.name
    weekly_trigger      = aws_cloudwatch_event_rule.weekly_retraining.name
    performance_trigger = aws_cloudwatch_event_rule.performance_trigger.name
  }
}

output "eventbridge_rule_arns" {
  description = "EventBridge rule ARNs for retraining triggers"
  value = {
    drift_trigger       = aws_cloudwatch_event_rule.drift_trigger.arn
    new_data_trigger    = aws_cloudwatch_event_rule.new_data_trigger.arn
    weekly_trigger      = aws_cloudwatch_event_rule.weekly_retraining.arn
    performance_trigger = aws_cloudwatch_event_rule.performance_trigger.arn
  }
}

# ============================================================
# SNS OUTPUTS
# ============================================================

output "sns_topic_arn" {
  description = "ARN of the SNS topic for alerts"
  value       = aws_sns_topic.alerts.arn
}

output "sns_topic_name" {
  description = "Name of the SNS topic for alerts"
  value       = aws_sns_topic.alerts.name
}

# ============================================================
# CLOUDWATCH OUTPUTS
# ============================================================

output "cloudwatch_log_group" {
  description = "CloudWatch log group name for Batch jobs"
  value = aws_cloudwatch_log_group.batch.name
}

output "cloudwatch_alarms" {
  description = "CloudWatch alarm names for monitoring"
  value = {
    performance_drop = aws_cloudwatch_metric_alarm.model_performance_drop.name
    drift_high       = aws_cloudwatch_metric_alarm.high_drift.name
    batch_failure    = aws_cloudwatch_metric_alarm.batch_job_failure.name
  }
}

# ============================================================
# COMPLETE DEPLOYMENT INFORMATION
# ============================================================

output "retraining_info" {
  description = "Complete retraining pipeline information"
  value = {
    ecr_repository      = aws_ecr_repository.training.repository_url
    job_queue           = aws_batch_job_queue.training.name
    job_definition      = aws_batch_job_definition.retraining.name
    job_definition_revision = aws_batch_job_definition.retraining.revision
    triggers = {
      drift       = aws_cloudwatch_event_rule.drift_trigger.name
      new_data    = aws_cloudwatch_event_rule.new_data_trigger.name
      weekly      = aws_cloudwatch_event_rule.weekly_retraining.name
      performance = aws_cloudwatch_event_rule.performance_trigger.name
    }
    alerts = {
      sns_topic = aws_sns_topic.alerts.arn
      alarms    = {
        performance = aws_cloudwatch_metric_alarm.model_performance_drop.name
        drift       = aws_cloudwatch_metric_alarm.high_drift.name
        batch       = aws_cloudwatch_metric_alarm.batch_job_failure.name
      }
    }
    logs = aws_cloudwatch_log_group.batch.name
  }
}