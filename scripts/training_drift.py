#!/usr/bin/env python3
"""
Pre-training drift detection for validating new data batches.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import boto3
import mlflow
import logging

logger = logging.getLogger(__name__)


class TrainingDriftDetector:
    """
    Detect data drift in new training batches.
    Used as a quality gate before retraining.
    """
    
    def __init__(
        self,
        threshold: float = 0.05,
        reference_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ):
        self.threshold = threshold
        self.reference_data = reference_data
        self.feature_names = feature_names or []
        self.reference_stats = None
        self.drift_history = []
        
        if reference_data is not None:
            self.compute_reference_stats(reference_data)
    
    def compute_reference_stats(self, data: np.ndarray):
        """Compute reference statistics from baseline data"""
        self.reference_stats = {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0),
            'q1': np.percentile(data, 25, axis=0),
            'q3': np.percentile(data, 75, axis=0),
            'skew': pd.DataFrame(data).skew().values,
            'kurtosis': pd.DataFrame(data).kurtosis().values
        }
        self.reference_data = data
        logger.info(f"✅ Computed reference stats for {data.shape[1]} features")
    
    def detect_drift(
        self,
        data: np.ndarray,
        metadata: Optional[pd.DataFrame] = None,
        batch_id: Optional[str] = None
    ) -> Dict:
        """
        Detect drift in new data batch
        """
        if batch_id is None:
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # If no reference, use first batch as reference
        if self.reference_stats is None:
            self.compute_reference_stats(data)
            return {
                'batch_id': batch_id,
                'drift_detected': False,
                'message': 'Reference data established',
                'drift_percentage': 0,
                'drifted_features': 0,
                'total_features': data.shape[1],
                'recommendation': 'No drift detected - reference data set'
            }
        
        # Compute statistics for new data
        new_stats = {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0)
        }
        
        # Detect drift for each feature
        results = []
        drift_features = []
        
        for i in range(data.shape[1]):
            # KS test for distribution drift
            stat, p_value = ks_2samp(self.reference_data[:, i], data[:, i])
            
            # Mean shift detection (z-score)
            mean_shift = abs(new_stats['mean'][i] - self.reference_stats['mean'][i])
            if self.reference_stats['std'][i] > 0:
                mean_shift_z = mean_shift / self.reference_stats['std'][i]
            else:
                mean_shift_z = 0
            
            drift_detected = (p_value < self.threshold) or (mean_shift_z > 2.0)
            
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}'
            
            result = {
                'feature_index': i,
                'feature_name': feature_name,
                'p_value': float(p_value),
                'statistic': float(stat),
                'mean_shift': float(mean_shift_z),
                'drift_detected': bool(drift_detected)
            }
            results.append(result)
            
            if drift_detected:
                drift_features.append(feature_name)
        
        # Calculate overall drift
        total_features = len(results)
        drifted_features = len(drift_features)
        drift_percentage = (drifted_features / total_features) * 100 if total_features > 0 else 0
        
        # Generate recommendations
        recommendations = []
        if drift_percentage > 30:
            recommendations.append({
                'type': 'critical',
                'message': '🚨 High drift detected. Investigate data pipeline before retraining.'
            })
        elif drift_percentage > 15:
            recommendations.append({
                'type': 'warning',
                'message': f'⚠️ Significant drift in {drifted_features} features. Retrain soon.'
            })
        elif drift_percentage > 5:
            recommendations.append({
                'type': 'info',
                'message': 'ℹ️ Minor drift detected. Monitor closely.'
            })
        else:
            recommendations.append({
                'type': 'success',
                'message': '✅ No significant drift detected. Proceed with retraining.'
            })
        
        report = {
            'batch_id': batch_id,
            'timestamp': datetime.now().isoformat(),
            'total_features': total_features,
            'drifted_features': drifted_features,
            'drift_percentage': drift_percentage,
            'threshold': self.threshold,
            'results': results,
            'recommendations': recommendations,
            'drift_detected': drift_percentage > 5
        }
        
        self.drift_history.append(report)
        self._log_to_mlflow(report)
        
        logger.info(f"📊 Drift detection: {drift_percentage:.1f}% drift detected")
        return report
    
    def _log_to_mlflow(self, report: Dict):
        """Log drift results to MLflow"""
        try:
            with mlflow.start_run(run_name=f"drift_detection_{report['batch_id']}"):
                mlflow.log_metrics({
                    'drift_percentage': report['drift_percentage'],
                    'drifted_features': report['drifted_features'],
                    'total_features': report['total_features']
                })
                for result in report['results']:
                    if result['drift_detected']:
                        mlflow.log_metric(f"{result['feature_name']}_drift", 1)
                    mlflow.log_metric(f"{result['feature_name']}_p_value", result['p_value'])
        except Exception as e:
            logger.warning(f"Could not log to MLflow: {e}")
    
    def save_report(self, report: Dict, bucket: str, key: str = 'drift_reports/latest_training_drift.json'):
        """Save drift report to S3"""
        try:
            s3 = boto3.client('s3')
            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(report, indent=2),
                ContentType='application/json'
            )
            logger.info(f"✅ Saved drift report to s3://{bucket}/{key}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save drift report: {e}")
            return False