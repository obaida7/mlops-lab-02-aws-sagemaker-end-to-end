output "sns_topic_arn" {
  description = "SNS topic ARN for drift and operational alerts"
  value       = aws_sns_topic.drift_alerts.arn
}

output "dashboard_name" {
  description = "CloudWatch Dashboard name"
  value       = aws_cloudwatch_dashboard.mlops_dashboard.dashboard_name
}

output "eventbridge_rule_arn" {
  description = "EventBridge Rule ARN for SageMaker Model Monitor"
  value       = aws_cloudwatch_event_rule.model_monitor_drift.arn
}

output "alarm_names" {
  description = "List of CloudWatch metric alarms provisioned"
  value = [
    aws_cloudwatch_metric_alarm.error_5xx.alarm_name,
    aws_cloudwatch_metric_alarm.p95_latency.alarm_name,
    aws_cloudwatch_metric_alarm.data_drift.alarm_name,
    aws_cloudwatch_metric_alarm.throttling.alarm_name
  ]
}
