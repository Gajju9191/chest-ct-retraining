
#!/usr/bin/env python3
"""
Model comparison logic for determining if new model should be deployed.
"""
import numpy as np
import tensorflow as tf
import boto3
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelComparator:
    """Compare new model with production model"""
    
    def __init__(self, model_bucket: str, improvement_threshold: float = 1.0):
        self.model_bucket = model_bucket
        self.improvement_threshold = improvement_threshold
        self.s3 = boto3.client('s3')
    
    def compare_models(self, new_model, val_features, val_labels):
        """
        Compare new model with production model
        
        Args:
            new_model: Newly trained model
            val_features: Validation features
            val_labels: Validation labels
            
        Returns:
            Dict with comparison results
        """
        # Evaluate new model
        new_loss, new_accuracy = new_model.evaluate(val_features, val_labels, verbose=0)
        logger.info(f"📊 New Model Accuracy: {new_accuracy:.4f}")
        
        # Download and evaluate production model
        prod_path = '/tmp/production_model.h5'
        try:
            self.s3.download_file(self.model_bucket, 'production/model.h5', prod_path)
            prod_model = tf.keras.models.load_model(prod_path)
            prod_loss, prod_accuracy = prod_model.evaluate(val_features, val_labels, verbose=0)
            
            if prod_accuracy > 0:
                improvement = ((new_accuracy - prod_accuracy) / prod_accuracy) * 100
            else:
                improvement = 100 if new_accuracy > 0 else 0
            
            should_deploy = improvement > self.improvement_threshold
            logger.info(f"📊 Old Model Accuracy: {prod_accuracy:.4f}")
            logger.info(f"📈 Improvement: {improvement:+.2f}%")
            
        except Exception as e:
            logger.warning(f"⚠️ No existing model found: {e}")
            prod_accuracy = 0
            improvement = 100
            should_deploy = True
        
        return {
            'new_accuracy': new_accuracy,
            'prod_accuracy': prod_accuracy,
            'improvement': improvement,
            'should_deploy': should_deploy,
            'timestamp': datetime.now().isoformat()
        }
    
    def compare_models_from_paths(self, new_model_path, val_features, val_labels):
        """Compare models from file paths"""
        new_model = tf.keras.models.load_model(new_model_path)
        return self.compare_models(new_model, val_features, val_labels)