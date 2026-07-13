
# ============================================================
# AWS SECRETS MANAGER FOR RETRAINING
# ============================================================

# Fetch existing secrets (data sources)
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

# IAM Policy for Batch to access Secrets Manager
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

# Attach policy to Batch role (adds to main.tf)
# This will be attached via aws_iam_role_policy_attachment in main.tf