
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

# Email subscription
resource "aws_sns_topic_subscription" "email_alert" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# SMS subscription (optional)
# resource "aws_sns_topic_subscription" "sms_alert" {
#   topic_arn = aws_sns_topic.alerts.arn
#   protocol  = "sms"
#   endpoint  = var.alert_phone
# }