# ════════════════════════════════════════════════════════════════════════
# 1. SNS TOPIC & POLICY FOR DRIFT ALERTS
# ════════════════════════════════════════════════════════════════════════
resource "aws_sns_topic" "drift_alerts" {
  name = var.sns_topic_name
  tags = var.tags
}

resource "aws_sns_topic_policy" "drift_alerts_policy" {
  arn = aws_sns_topic.drift_alerts.arn

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowEventBridgePublish"
        Effect = "Allow"
        Principal = {
          Service = "events.amazonaws.com"
        }
        Action   = "sns:Publish"
        Resource = aws_sns_topic.drift_alerts.arn
      }
    ]
  })
}

# ════════════════════════════════════════════════════════════════════════
# 2. EVENTBRIDGE RULE FOR MODEL MONITOR DRIFT
# ════════════════════════════════════════════════════════════════════════
resource "aws_cloudwatch_event_rule" "model_monitor_drift" {
  name        = "bank-churn-model-monitor-drift-rule"
  description = "Triggers alerts and automated workflows on SageMaker Model Monitor violations"

  event_pattern = jsonencode({
    source        = ["aws.sagemaker"]
    "detail-type" = ["SageMaker Model Monitor Schedule Status"]
    detail = {
      MonitorScheduleStatus = ["CompletedWithViolations"]
    }
  })

  tags = var.tags
}

resource "aws_cloudwatch_event_target" "sns_target" {
  rule      = aws_cloudwatch_event_rule.model_monitor_drift.name
  target_id = "SendDriftAlertToSNS"
  arn       = aws_sns_topic.drift_alerts.arn
}

# ════════════════════════════════════════════════════════════════════════
# 2b. EVENTBRIDGE TARGET 2: API DESTINATION FOR GITHUB ACTIONS RETRAINING
# ════════════════════════════════════════════════════════════════════════
resource "aws_cloudwatch_event_connection" "github_connection" {
  name               = "bank-churn-github-connection"
  description        = "Authorization connection for GitHub Actions API"
  authorization_type = "API_KEY"

  auth_parameters {
    api_key {
      key   = "Authorization"
      value = "Bearer ${var.github_token}"
    }
  }
}

resource "aws_cloudwatch_event_api_destination" "github_api_destination" {
  name                             = "bank-churn-github-retrain-destination"
  description                      = "POSTs to GitHub Actions to trigger end-to-end.yml on data drift"
  invocation_endpoint              = "https://api.github.com/repos/${var.github_repo}/actions/workflows/end-to-end.yml/dispatches"
  http_method                      = "POST"
  invocation_rate_limit_per_second = 1
  connection_arn                   = aws_cloudwatch_event_connection.github_connection.arn
}

resource "aws_iam_role" "eventbridge_api_destination_role" {
  name = "bank-churn-eventbridge-api-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "events.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_policy" "eventbridge_api_destination_policy" {
  name        = "bank-churn-eventbridge-api-policy"
  description = "Allows EventBridge to invoke the GitHub Actions API Destination"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "events:InvokeApiDestination"
        Resource = aws_cloudwatch_event_api_destination.github_api_destination.arn
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "eventbridge_api_attach" {
  role       = aws_iam_role.eventbridge_api_destination_role.name
  policy_arn = aws_iam_policy.eventbridge_api_destination_policy.arn
}

resource "aws_cloudwatch_event_target" "github_target" {
  rule      = aws_cloudwatch_event_rule.model_monitor_drift.name
  target_id = "TriggerGitHubRetrainingWorkflow"
  arn       = aws_cloudwatch_event_api_destination.github_api_destination.arn
  role_arn  = aws_iam_role.eventbridge_api_destination_role.arn

  input = jsonencode({
    ref = "main"
    inputs = {
      data_mode = "retrain_with_production_capture"
    }
  })
}

# ════════════════════════════════════════════════════════════════════════
# 3. CLOUDWATCH PRODUCTION METRIC ALARMS
# ════════════════════════════════════════════════════════════════════════
resource "aws_cloudwatch_metric_alarm" "error_5xx" {
  alarm_name          = "BankChurn-5xxErrorAlarm"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "Invocation5XXErrors"
  namespace           = "AWS/SageMaker"
  period              = 60
  statistic           = "Sum"
  threshold           = 1
  alarm_description   = "Triggers when SageMaker endpoint throws any 5xx container or serving errors."
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.drift_alerts.arn]

  dimensions = {
    EndpointName = var.endpoint_name
    VariantName  = "AllTraffic"
  }

  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "p95_latency" {
  alarm_name          = "BankChurn-P95LatencySLAAlarm"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "ModelLatency"
  namespace           = "AWS/SageMaker"
  period              = 60
  extended_statistic  = "p95"
  threshold           = 500000 # 500ms SLA in microseconds
  alarm_description   = "Triggers when P95 latency exceeds the 500ms banking SLA."
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.drift_alerts.arn]

  dimensions = {
    EndpointName = var.endpoint_name
    VariantName  = "AllTraffic"
  }

  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "data_drift" {
  alarm_name          = "BankChurn-DataDriftViolationAlarm"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "DataDriftViolation"
  namespace           = "BankChurn/Monitoring"
  period              = 300
  statistic           = "Maximum"
  threshold           = 1
  alarm_description   = "Triggers when Two-Sample Kolmogorov-Smirnov test detects significant demographic drift."
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.drift_alerts.arn]

  dimensions = {
    EndpointName = var.endpoint_name
  }

  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "throttling" {
  alarm_name          = "BankChurn-ThrottlingAlarm"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "ThrottledRequests"
  namespace           = "AWS/SageMaker"
  period              = 60
  statistic           = "Sum"
  threshold           = 1
  alarm_description   = "Triggers when incoming request rate exceeds serverless max concurrency limits."
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.drift_alerts.arn]

  dimensions = {
    EndpointName = var.endpoint_name
  }

  tags = var.tags
}

# ════════════════════════════════════════════════════════════════════════
# 4. CLOUDWATCH EXECUTIVE MLOPS DASHBOARD
# ════════════════════════════════════════════════════════════════════════
resource "aws_cloudwatch_dashboard" "mlops_dashboard" {
  dashboard_name = var.dashboard_name
  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "text"
        x      = 0
        y      = 0
        width  = 24
        height = 2
        properties = {
          markdown = "# 🏦 Apex Bank | Customer Churn Model Monitoring\n**Endpoint:** `${var.endpoint_name}` | **SLA:** 500ms P95 | **Model Registry:** `bank-customer-churn`"
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 2
        width  = 8
        height = 6
        properties = {
          metrics = [
            ["AWS/SageMaker", "Invocations", "EndpointName", var.endpoint_name, "VariantName", "AllTraffic", { stat = "Sum", color = "#2ca02c" }]
          ]
          view    = "timeSeries"
          stacked = false
          region  = "us-east-2"
          title   = "Real-Time Invocations (Volume)"
          period  = 300
        }
      },
      {
        type   = "metric"
        x      = 8
        y      = 2
        width  = 8
        height = 6
        properties = {
          metrics = [
            ["AWS/SageMaker", "Invocation5XXErrors", "EndpointName", var.endpoint_name, "VariantName", "AllTraffic", { stat = "Sum", color = "#d62728" }],
            ["AWS/SageMaker", "Invocation4XXErrors", "EndpointName", var.endpoint_name, "VariantName", "AllTraffic", { stat = "Sum", color = "#ff7f0e" }]
          ]
          view    = "timeSeries"
          stacked = false
          region  = "us-east-2"
          title   = "Error Rates (4xx & 5xx)"
          period  = 300
        }
      },
      {
        type   = "metric"
        x      = 16
        y      = 2
        width  = 8
        height = 6
        properties = {
          metrics = [
            ["AWS/SageMaker", "ModelLatency", "EndpointName", var.endpoint_name, "VariantName", "AllTraffic", { stat = "p95", color = "#e377c2", label = "P95 Latency (us)" }],
            ["AWS/SageMaker", "ModelLatency", "EndpointName", var.endpoint_name, "VariantName", "AllTraffic", { stat = "Average", color = "#1f77b4", label = "Avg Latency (us)" }]
          ]
          view    = "timeSeries"
          stacked = false
          region  = "us-east-2"
          title   = "Model Latency vs 500ms SLA (Microseconds)"
          period  = 300
          annotations = {
            horizontal = [
              { value = 500000, label = "500ms P95 SLA Threshold", color = "#d62728" }
            ]
          }
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 8
        width  = 12
        height = 6
        properties = {
          metrics = [
            ["BankChurn/Monitoring", "DataDriftViolation", "EndpointName", var.endpoint_name, { stat = "Maximum", color = "#9467bd" }]
          ]
          view    = "timeSeries"
          stacked = false
          region  = "us-east-2"
          title   = "Statistical Covariate Drift Violations (K-S Test)"
          period  = 300
          yAxis = {
            left = { max = 1, min = 0 }
          }
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 8
        width  = 12
        height = 6
        properties = {
          metrics = [
            ["AWS/SageMaker", "ThrottledRequests", "EndpointName", var.endpoint_name, { stat = "Sum", color = "#8c564b" }]
          ]
          view    = "timeSeries"
          stacked = false
          region  = "us-east-2"
          title   = "Serverless Concurrency Throttling Events"
          period  = 300
        }
      }
    ]
  })
}
