
#!/usr/bin/env python3
"""
AWS Secrets Manager integration for secure credential management.
"""
import os
import json
import boto3
import logging
from functools import lru_cache
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class SecretsManager:
    """Secure secrets manager with caching"""
    
    def __init__(self, region: str = "us-east-1"):
        self.client = boto3.client('secretsmanager', region_name=region)
        self._cache = {}
        self._load_secrets()
    
    def _load_secrets(self):
        """Load common secrets on initialization"""
        secret_names = [
            'chest-ct/mlflow/credentials',
            'chest-ct/s3/credentials',
            'chest-ct/jenkins/credentials',
            'chest-ct/deployment/credentials'
        ]
        
        for secret_name in secret_names:
            try:
                self._cache[secret_name] = self._get_secret(secret_name)
                logger.info(f"✅ Loaded secret: {secret_name}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load secret {secret_name}: {e}")
    
    @lru_cache(maxsize=128)
    def _get_secret(self, secret_name: str) -> dict:
        """Retrieve a secret from AWS Secrets Manager"""
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
            
            if 'SecretString' in response:
                return json.loads(response['SecretString'])
            else:
                raise ValueError("Binary secrets not supported")
                
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceNotFoundException':
                logger.error(f"❌ Secret {secret_name} not found")
            elif error_code == 'InvalidRequestException':
                logger.error(f"❌ Invalid request for secret {secret_name}")
            else:
                logger.error(f"❌ Error retrieving secret {secret_name}: {e}")
            raise
    
    def get_secret(self, secret_name: str) -> dict:
        """Get a specific secret by name"""
        return self._cache.get(secret_name, {})
    
    def get_mlflow_credentials(self) -> dict:
        """Get MLflow credentials from secrets"""
        return self.get_secret('chest-ct/mlflow/credentials')
    
    def get_s3_credentials(self) -> dict:
        """Get S3 bucket names from secrets"""
        return self.get_secret('chest-ct/s3/credentials')
    
    def get_jenkins_credentials(self) -> dict:
        """Get Jenkins credentials from secrets"""
        return self.get_secret('chest-ct/jenkins/credentials')
    
    def get_deployment_credentials(self) -> dict:
        """Get deployment credentials from secrets"""
        return self.get_secret('chest-ct/deployment/credentials')
    
    def setup_mlflow(self, config_uri=None):
        """Setup MLflow environment variables from secrets"""
        creds = self.get_mlflow_credentials()
        
        if creds:
            tracking_uri = creds.get('tracking_uri', config_uri)
            os.environ['MLFLOW_TRACKING_URI'] = tracking_uri
            os.environ['MLFLOW_TRACKING_USERNAME'] = creds.get('username', '')
            os.environ['MLFLOW_TRACKING_PASSWORD'] = creds.get('password', '')
            logger.info(f"✅ MLflow configured from Secrets Manager: {tracking_uri}")
            return tracking_uri
        elif config_uri:
            os.environ['MLFLOW_TRACKING_URI'] = config_uri
            logger.info(f"✅ MLflow configured from config: {config_uri}")
            return config_uri
        return None