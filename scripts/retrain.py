#!/usr/bin/env python3
"""
Event-Driven Retraining Pipeline with VGG16
- Triggers on: Data Drift, Performance Drop, New Data, Schedule
- Consistent with deployment model
- Feature pipeline used ONLY for drift detection
"""
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import boto3
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import requests
import zipfile
import shutil

# Import modules
from feature_pipeline import FeaturePipeline
from training_drift import TrainingDriftDetector
from secrets_manager import SecretsManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# SECRETS MANAGER
# ============================================================
secrets = SecretsManager()

# MLflow credentials
mlflow_creds = secrets.get_mlflow_credentials()
if mlflow_creds:
    os.environ['MLFLOW_TRACKING_URI'] = mlflow_creds.get('tracking_uri', '')
    os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_creds.get('username', '')
    os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_creds.get('password', '')
    mlflow.set_tracking_uri(mlflow_creds.get('tracking_uri', ''))
    logger.info(f"✅ MLflow configured from Secrets Manager")

# S3 credentials
s3_creds = secrets.get_s3_credentials()
MODEL_BUCKET = s3_creds.get('models_bucket', 'chest-ct-models-155407238004')
DATA_BUCKET = s3_creds.get('data_bucket', 'chest-models-gajju')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')

# Jenkins credentials
jenkins_creds = secrets.get_jenkins_credentials()
JENKINS_URL = jenkins_creds.get('url', 'http://54.81.172.67:8080')
JENKINS_TOKEN = jenkins_creds.get('token', 'ct-trigger-token')
JENKINS_USERNAME = jenkins_creds.get('username', 'Gajju9191')
JENKINS_API_TOKEN = jenkins_creds.get('api_token', '')
JOB_NAME = "ecs-cicd-d"

# ============================================================
# EVENT-DRIVEN CONFIGURATION
# ============================================================
IMPROVEMENT_THRESHOLD = 1.0  # Minimum improvement to deploy (%)
DRIFT_THRESHOLD = 10.0  # Drift percentage to trigger retraining
PERFORMANCE_THRESHOLD = 0.85  # Minimum accuracy before retraining (85%)
MIN_IMPROVEMENT = 0.02  # At least 2% improvement to deploy

# VGG16 Training parameters
BATCH_SIZE = 12
EPOCHS = 15
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
IMAGE_SIZE = (224, 224)

# ============================================================
# ALERT SYSTEM
# ============================================================

def send_email_alert(subject, message, severity="INFO"):
    """Send email alert via AWS SNS"""
    try:
        sns = boto3.client('sns', region_name=AWS_REGION)
        topic_arn = "arn:aws:sns:us-east-1:437619427369:chest-ct-alerts"
        
        full_message = f"""
        ═══════════════════════════════════════════════════════
        CHEST CT MLOPS ALERT
        ═══════════════════════════════════════════════════════
        
        Severity: {severity}
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
        
        {message}
        
        ═══════════════════════════════════════════════════════
        """
        
        response = sns.publish(
            TopicArn=topic_arn,
            Subject=f"[{severity}] {subject}",
            Message=full_message
        )
        logger.info(f"✅ Alert sent: {subject}")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Could not send email alert: {e}")
        return False

# ============================================================
# TRIGGER DETECTION FUNCTIONS
# ============================================================

def should_retrain_due_to_drift(reference_features_path='/tmp/reference_features.npy'):
    """Check if drift threshold is exceeded"""
    try:
        s3 = boto3.client('s3')
        s3.download_file(MODEL_BUCKET, 'reference_features.npy', reference_features_path)
        reference_data = np.load(reference_features_path)
        logger.info("✅ Loaded reference data for drift check")
    except:
        logger.info("ℹ️ No reference data found. First retraining will establish baseline.")
        return True  # First run should always retrain
    
    try:
        # Check latest drift report
        response = s3.get_object(
            Bucket=MODEL_BUCKET,
            Key='drift_reports/latest_training_drift.json'
        )
        drift_report = json.loads(response['Body'].read())
        
        drift_pct = drift_report.get('drift_percentage', 0)
        logger.info(f"📊 Last drift percentage: {drift_pct:.1f}%")
        
        if drift_pct > DRIFT_THRESHOLD:
            logger.info(f"🚨 Drift ({drift_pct:.1f}%) exceeds threshold ({DRIFT_THRESHOLD}%)")
            return True
        else:
            logger.info(f"✅ Drift ({drift_pct:.1f}%) below threshold ({DRIFT_THRESHOLD}%)")
            return False
            
    except Exception as e:
        logger.warning(f"⚠️ Could not check drift: {e}")
        return True  # Retrain if can't check

def should_retrain_due_to_performance():
    """Check if production model performance has dropped below threshold"""
    try:
        s3 = boto3.client('s3')
        
        # Get production model accuracy from MLflow or S3
        # Check if we have a recent evaluation
        try:
            response = s3.get_object(
                Bucket=MODEL_BUCKET,
                Key='evaluation_metrics.json'
            )
            metrics = json.loads(response['Body'].read())
            current_accuracy = metrics.get('accuracy', 1.0)
        except:
            # If no metrics, assume model is good
            current_accuracy = 1.0
        
        logger.info(f"📊 Current production accuracy: {current_accuracy:.2%}")
        
        if current_accuracy < PERFORMANCE_THRESHOLD:
            logger.info(f"🚨 Performance ({current_accuracy:.2%}) below threshold ({PERFORMANCE_THRESHOLD:.2%})")
            return True
        else:
            logger.info(f"✅ Performance ({current_accuracy:.2%}) above threshold ({PERFORMANCE_THRESHOLD:.2%})")
            return False
            
    except Exception as e:
        logger.warning(f"⚠️ Could not check performance: {e}")
        return True  # Retrain if can't check

def should_retrain_due_to_new_data():
    """Check if new data has arrived since last retraining"""
    try:
        s3 = boto3.client('s3')
        
        # Check when data was last modified
        response = s3.head_object(
            Bucket=DATA_BUCKET,
            Key='chest-data.zip'
        )
        data_last_modified = response['LastModified']
        
        # Check when last retraining happened
        try:
            response = s3.head_object(
                Bucket=MODEL_BUCKET,
                Key='retraining_history/last_run.txt'
            )
            last_retraining = response['LastModified']
        except:
            last_retraining = datetime(2024, 1, 1)  # Long time ago
        
        logger.info(f"📅 Data last modified: {data_last_modified}")
        logger.info(f"📅 Last retraining: {last_retraining}")
        
        # If data is newer than last retraining
        if data_last_modified > last_retraining:
            logger.info("🆕 New data detected since last retraining")
            return True
        else:
            logger.info("✅ No new data since last retraining")
            return False
            
    except Exception as e:
        logger.warning(f"⚠️ Could not check new data: {e}")
        return False  # Don't retrain if can't check

def check_retraining_triggers():
    """
    Check all triggers and determine if retraining is needed
    Returns: (should_retrain, trigger_reason)
    """
    reasons = []
    
    # Check drift
    if should_retrain_due_to_drift():
        reasons.append("Data drift detected")
    
    # Check performance
    if should_retrain_due_to_performance():
        reasons.append("Model performance dropped below threshold")
    
    # Check new data
    if should_retrain_due_to_new_data():
        reasons.append("New data available")
    
    # Default: First run
    if not reasons:
        # Check if this is first run (no model exists)
        s3 = boto3.client('s3')
        try:
            s3.head_object(Bucket=MODEL_BUCKET, Key='production/model.h5')
            # Model exists, no triggers
            logger.info("✅ No retraining triggers detected")
            return False, "No triggers detected"
        except:
            # No model exists - first run
            reasons.append("First run - no production model")
    
    if reasons:
        return True, ", ".join(reasons)
    else:
        return False, "No triggers detected"

# ============================================================
# DATA LOADING
# ============================================================

def fix_nested_directory(data_path):
    """Fix nested directory structure if needed"""
    items = list(data_path.iterdir())
    
    if len(items) == 1 and items[0].is_dir():
        inner_dir = items[0]
        inner_items = list(inner_dir.iterdir())
        if inner_items and all(item.is_dir() for item in inner_items):
            logger.info(f"📁 Found nested directory structure: {inner_dir.name}")
            for item in inner_items:
                target_path = data_path / item.name
                if target_path.exists():
                    shutil.rmtree(target_path)
                shutil.move(str(item), str(target_path))
                logger.info(f"   ✅ Moved: {item.name}")
            inner_dir.rmdir()
            logger.info("✅ Fixed nested directory structure!")
            return True
    return False


def download_training_data():
    """Download and extract training data from S3"""
    s3 = boto3.client('s3')
    data_path = Path('/tmp/data')
    data_path.mkdir(parents=True, exist_ok=True)
    
    try:
        zip_path = '/tmp/data.zip'
        s3.download_file(DATA_BUCKET, 'chest-data.zip', zip_path)
        logger.info("✅ Downloaded data from S3")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        logger.info(f"✅ Extracted data to {data_path}")
        
        fix_nested_directory(data_path)
        
        # Verify data structure
        class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        logger.info(f"📊 Found {len(class_dirs)} class directories:")
        for class_dir in class_dirs:
            image_count = len(list(class_dir.glob('*.[jp][pn][g]')))
            logger.info(f"   - {class_dir.name}: {image_count} images")
        
        return data_path
        
    except Exception as e:
        error_msg = f"Data download failed: {e}"
        logger.error(f"❌ {error_msg}")
        send_email_alert(
            "⚠️ RETRAINING DATA DOWNLOAD FAILED",
            f"Failed to download training data from S3.\nError: {e}\nBucket: {DATA_BUCKET}",
            severity="ERROR"
        )
        raise

# ============================================================
# VGG16 MODEL
# ============================================================

def create_vgg16_model():
    """Create VGG16 model - IDENTICAL to deployment model"""
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    
    logger.info("🔄 Creating VGG16 model (identical to deployment)...")
    
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info("✅ VGG16 model created (identical to deployment)")
    return model


def train_vgg16_model(model, data_path):
    """Train VGG16 on raw images with 15 epochs"""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    # Check data structure
    logger.info(f"📁 Checking data path: {data_path}")
    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    if len(class_dirs) == 0:
        logger.error(f"❌ No class directories found in {data_path}")
        return model, None, None, None
    
    logger.info(f"Found {len(class_dirs)} class directories: {[d.name for d in class_dirs]}")
    for class_dir in class_dirs:
        image_count = len(list(class_dir.glob('*.[jp][pn][g]')))
        logger.info(f"  - {class_dir.name}: {image_count} images")
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=VALIDATION_SPLIT
    )
    
    train_generator = datagen.flow_from_directory(
        data_path,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_generator = datagen.flow_from_directory(
        data_path,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    logger.info(f"✅ Classes: {train_generator.class_indices}")
    logger.info(f"✅ Training samples: {train_generator.samples}")
    logger.info(f"✅ Validation samples: {val_generator.samples}")
    
    if train_generator.samples < 10 or val_generator.samples < 10:
        logger.warning("⚠️ Not enough samples for training!")
        return model, train_generator, val_generator, None
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7)
    ]
    
    logger.info(f"🚀 Training VGG16 on {train_generator.samples} images for {EPOCHS} epochs...")
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    final_accuracy = history.history['accuracy'][-1]
    val_accuracy = history.history['val_accuracy'][-1]
    logger.info(f"✅ Training complete. Accuracy: {final_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    return model, train_generator, val_generator, history

# ============================================================
# DRIFT DETECTION
# ============================================================

def check_data_drift(data_path):
    """Run drift detection on new training data"""
    logger.info("📊 Running pre-training drift detection...")
    
    feature_pipeline = FeaturePipeline({
        'use_pca': True,
        'pca_components': 50
    })
    
    X, y, metadata = feature_pipeline.extract_features(data_path)
    
    if len(X) == 0:
        logger.warning("⚠️ No features extracted for drift detection. Skipping...")
        return {
            'drift_percentage': 0,
            'drifted_features': 0,
            'recommendation': 'No drift detection - insufficient data'
        }
    
    X_transformed = feature_pipeline.fit_transform(X)
    feature_pipeline.save(Path('/tmp/feature_pipeline'))
    
    drift_detector = TrainingDriftDetector(
        threshold=0.05,
        feature_names=feature_pipeline.feature_names
    )
    
    try:
        s3 = boto3.client('s3')
        s3.download_file(MODEL_BUCKET, 'reference_features.npy', '/tmp/reference_features.npy')
        reference_data = np.load('/tmp/reference_features.npy')
        drift_detector.compute_reference_stats(reference_data)
        logger.info("✅ Loaded reference data from S3")
    except:
        logger.info("ℹ️ No reference data found. Using current batch as reference.")
    
    drift_report = drift_detector.detect_drift(
        X_transformed,
        metadata,
        batch_id=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    drift_detector.save_report(drift_report, MODEL_BUCKET)
    
    try:
        mlflow.log_metrics({
            "drift_percentage": drift_report['drift_percentage'],
            "drifted_features": drift_report['drifted_features']
        })
    except Exception as e:
        logger.warning(f"Could not log drift metrics to MLflow: {e}")
    
    np.save('/tmp/reference_features.npy', X_transformed)
    s3 = boto3.client('s3')
    s3.upload_file('/tmp/reference_features.npy', MODEL_BUCKET, 'reference_features.npy')
    
    return drift_report

# ============================================================
# MODEL COMPARISON
# ============================================================

def compare_models(new_model, val_generator):
    """Compare new model with production model on same validation set"""
    from tensorflow.keras.models import load_model
    
    if val_generator is None:
        logger.warning("⚠️ No validation generator available. Skipping model comparison.")
        return {
            'new_accuracy': 0,
            'prod_accuracy': 0,
            'improvement': 0,
            'should_deploy': False
        }
    
    # Evaluate new model
    new_loss, new_accuracy = new_model.evaluate(val_generator, verbose=0)
    logger.info(f"📊 New Model Accuracy: {new_accuracy:.4f}")
    
    # Download and evaluate production model
    s3 = boto3.client('s3')
    prod_path = '/tmp/production_model.h5'
    try:
        s3.download_file(MODEL_BUCKET, 'production/model.h5', prod_path)
        prod_model = load_model(prod_path)
        prod_loss, prod_accuracy = prod_model.evaluate(val_generator, verbose=0)
        
        if prod_accuracy > 0:
            improvement = ((new_accuracy - prod_accuracy) / prod_accuracy) * 100
        else:
            improvement = 100 if new_accuracy > 0 else 0
        
        should_deploy = improvement > IMPROVEMENT_THRESHOLD
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
        'should_deploy': should_deploy
    }


def upload_model_to_s3(model_path):
    """Upload model to S3 with versioning"""
    s3 = boto3.client('s3')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    version_key = f"models/model_{timestamp}.h5"
    s3.upload_file(model_path, MODEL_BUCKET, version_key)
    logger.info(f"✅ Uploaded versioned model: {version_key}")
    
    copy_source = {'Bucket': MODEL_BUCKET, 'Key': version_key}
    s3.copy_object(CopySource=copy_source, Bucket=MODEL_BUCKET, Key='production/model.h5')
    logger.info("✅ Updated production model")
    
    s3.copy_object(CopySource=copy_source, Bucket=MODEL_BUCKET, Key='model.h5')
    logger.info("✅ Copied model to root path")
    
    # Update retraining history
    s3.put_object(
        Bucket=MODEL_BUCKET,
        Key='retraining_history/last_run.txt',
        Body=datetime.now().isoformat()
    )
    
    return version_key


def trigger_jenkins():
    """Trigger Jenkins deployment with CSRF crumb"""
    try:
        crumb_url = f"{JENKINS_URL}/crumbIssuer/api/json"
        logger.info(f"🔑 Getting CSRF crumb...")
        
        crumb_resp = requests.get(
            crumb_url,
            auth=(JENKINS_USERNAME, JENKINS_API_TOKEN),
            timeout=30
        )
        
        if crumb_resp.status_code == 200:
            crumb_data = crumb_resp.json()
            crumb = crumb_data['crumb']
            crumb_field = crumb_data['crumbRequestField']
            logger.info(f"✅ CSRF crumb obtained")
            
            url = f"{JENKINS_URL}/job/{JOB_NAME}/build"
            headers = {crumb_field: crumb}
            
            response = requests.post(
                url,
                headers=headers,
                auth=(JENKINS_USERNAME, JENKINS_API_TOKEN),
                timeout=30
            )
            
            if response.status_code == 201 or response.status_code == 200:
                logger.info("✅ Jenkins build triggered successfully!")
                return True
            else:
                logger.error(f"❌ Jenkins trigger failed: {response.status_code}")
                return False
        else:
            logger.error(f"❌ Failed to get CSRF crumb: {crumb_resp.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error triggering Jenkins: {e}")
        return False

# ============================================================
# MAIN RETRAINING PIPELINE
# ============================================================

def main():
    """Event-driven retraining pipeline"""
    logger.info("=" * 60)
    logger.info("🔄 Starting Event-Driven VGG16 Retraining Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Check if retraining is needed
    should_retrain, trigger_reason = check_retraining_triggers()
    
    if not should_retrain:
        logger.info(f"⏸️ Skipping retraining: {trigger_reason}")
        return
    
    logger.info(f"📋 Retraining triggered by: {trigger_reason}")
    send_email_alert(
        "🔄 Retraining Triggered",
        f"Retraining started due to: {trigger_reason}",
        severity="INFO"
    )
    
    try:
        with mlflow.start_run(run_name=f"retraining_vgg16_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log trigger reason
            mlflow.log_param("trigger_reason", trigger_reason)
            
            # Log parameters
            mlflow.log_params({
                "model_architecture": "VGG16",
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "image_size": IMAGE_SIZE,
                "validation_split": VALIDATION_SPLIT,
                "drift_threshold": DRIFT_THRESHOLD,
                "improvement_threshold": IMPROVEMENT_THRESHOLD,
                "performance_threshold": PERFORMANCE_THRESHOLD
            })
            
            # Step 2: Download data
            logger.info("📥 Downloading training data...")
            data_path = download_training_data()
            
            # Step 3: Check data drift
            logger.info("📊 Running pre-training drift detection...")
            drift_report = check_data_drift(data_path)
            
            rec_message = drift_report.get('recommendation', 'No recommendation')
            if isinstance(rec_message, list):
                rec_message = rec_message[0].get('message', 'No recommendation') if rec_message else 'No recommendation'
            logger.info(f"📊 Drift: {drift_report['drift_percentage']:.1f}% - {rec_message}")
            
            # Step 4: Create VGG16 model
            model = create_vgg16_model()
            
            # Step 5: Train on raw images
            model, train_generator, val_generator, history = train_vgg16_model(model, data_path)
            
            # Step 6: Save model
            model_path = '/tmp/model.h5'
            model.save(model_path)
            logger.info(f"✅ Model saved to {model_path}")
            
            # Step 7: Compare with production model
            logger.info("⚖️ Comparing models...")
            comparison = compare_models(model, val_generator)
            
            mlflow.log_metrics({
                "new_accuracy": comparison['new_accuracy'],
                "old_accuracy": comparison['prod_accuracy'],
                "improvement": comparison['improvement']
            })
            
            # Step 8: Deploy if improved
            if comparison['should_deploy']:
                logger.info(f"✅ Model improved by {comparison['improvement']:.2f}%")
                version = upload_model_to_s3(model_path)
                jenkins_triggered = trigger_jenkins()
                
                mlflow.log_param("deployed_version", version)
                mlflow.log_param("deployed", True)
                
                send_email_alert(
                    "✅ NEW MODEL DEPLOYED!",
                    f"""
                    Retraining triggered by: {trigger_reason}
                    
                    New Model Accuracy: {comparison['new_accuracy']:.2%}
                    Old Model Accuracy: {comparison['prod_accuracy']:.2%}
                    Improvement: +{comparison['improvement']:.2f}%
                    Model Version: {version}
                    Jenkins Triggered: {jenkins_triggered}
                    """,
                    severity="SUCCESS"
                )
            else:
                logger.info(f"⏸️ No significant improvement: {comparison['improvement']:.2f}%")
                mlflow.log_param("deployed", False)
                
                send_email_alert(
                    "ℹ️ No Model Improvement",
                    f"""
                    Retraining triggered by: {trigger_reason}
                    
                    New Model Accuracy: {comparison['new_accuracy']:.2%}
                    Old Model Accuracy: {comparison['prod_accuracy']:.2%}
                    Improvement: {comparison['improvement']:.2f}%
                    Threshold: {IMPROVEMENT_THRESHOLD}%
                    """,
                    severity="INFO"
                )
            
            logger.info("=" * 60)
            logger.info("✅ Retraining pipeline completed successfully!")
            logger.info("=" * 60)
            
    except Exception as e:
        error_msg = f"Retraining failed: {e}"
        logger.error(f"❌ {error_msg}")
        
        send_email_alert(
            "❌ RETRAINING FAILED",
            f"""
            Trigger Reason: {trigger_reason}
            Error: {e}
            Check CloudWatch logs for details.
            """,
            severity="ERROR"
        )
        raise


if __name__ == "__main__":
    main()