
#!/usr/bin/env python3
"""
Enhanced AWS Batch Retraining Script with:
- VGG16 Model (consistent with deployment)
- Feature Engineering Pipeline
- Pre-training Drift Detection
- AWS Secrets Manager Integration
- MLflow Tracking
- Model Comparison & Deployment
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

# Import modules
from feature_pipeline import FeaturePipeline
from training_drift import TrainingDriftDetector
from secrets_manager import SecretsManager
from model_comparator import ModelComparator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# SECRETS MANAGER - Fetch all credentials
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
JENKINS_USERNAME = jenkins_creds.get('username', 'Gajanan Wagalgave')
JENKINS_API_TOKEN = jenkins_creds.get('api_token', '')
JOB_NAME = "ecs-cicd-d"

# Deployment credentials
deploy_creds = secrets.get_deployment_credentials()
if deploy_creds:
    logger.info("✅ Deployment credentials loaded")

# ============================================================
# CONFIGURATION
# ============================================================
IMPROVEMENT_THRESHOLD = 1.0  # Minimum improvement to deploy (%)
DRIFT_THRESHOLD = 15.0  # Drift percentage to trigger retraining

# VGG16 Training parameters
BATCH_SIZE = 12
EPOCHS = 10
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
IMAGE_SIZE = (224, 224, 3)

# ============================================================
# VGG16 MODEL FUNCTIONS
# ============================================================

def download_training_data():
    """Download and extract training data from S3"""
    s3 = boto3.client('s3')
    data_path = Path('/tmp/data')
    data_path.mkdir(parents=True, exist_ok=True)
    
    try:
        zip_path = '/tmp/data.zip'
        s3.download_file(DATA_BUCKET, 'chest-data.zip', zip_path)
        logger.info("✅ Downloaded data from S3")
        
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        logger.info(f"✅ Extracted data to {data_path}")
        return data_path
    except Exception as e:
        logger.error(f"❌ Data download failed: {e}")
        raise


def get_vgg16_base_model():
    """Load VGG16 base model with ImageNet weights"""
    logger.info("🔄 Loading VGG16 base model...")
    
    base_model = tf.keras.applications.VGG16(
        input_shape=IMAGE_SIZE,
        weights='imagenet',
        include_top=False
    )
    base_model.trainable = False
    
    # Add custom head
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info("✅ VGG16 model created with custom head")
    return model


def train_vgg16_model(model, train_data_path, val_data_path):
    """Train VGG16 model on images"""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_data_path,
        target_size=IMAGE_SIZE[:-1],
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_data_path,
        target_size=IMAGE_SIZE[:-1],
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    ]
    
    logger.info(f"🚀 Training VGG16 on {train_generator.samples} images...")
    
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
    
    return model, history


def compare_models(new_model, val_generator):
    """Compare new VGG16 model with production model"""
    # Evaluate new model
    new_loss, new_accuracy = new_model.evaluate(val_generator, verbose=0)
    logger.info(f"📊 New Model Accuracy: {new_accuracy:.4f}")
    
    # Download production model
    s3 = boto3.client('s3')
    prod_path = '/tmp/production_model.h5'
    try:
        s3.download_file(MODEL_BUCKET, 'production/model.h5', prod_path)
        prod_model = tf.keras.models.load_model(prod_path)
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
    
    # Versioned
    version_key = f"models/model_{timestamp}.h5"
    s3.upload_file(model_path, MODEL_BUCKET, version_key)
    logger.info(f"✅ Uploaded versioned model: {version_key}")
    
    # Production
    copy_source = {'Bucket': MODEL_BUCKET, 'Key': version_key}
    s3.copy_object(CopySource=copy_source, Bucket=MODEL_BUCKET, Key='production/model.h5')
    logger.info("✅ Updated production model")
    
    # Root (for Jenkins)
    s3.copy_object(CopySource=copy_source, Bucket=MODEL_BUCKET, Key='model.h5')
    logger.info("✅ Copied model to root path")
    
    return version_key


def trigger_jenkins():
    """Trigger Jenkins deployment"""
    try:
        # Get CSRF crumb
        crumb_url = f"{JENKINS_URL}/crumbIssuer/api/json"
        crumb_resp = requests.get(crumb_url, timeout=30)
        
        headers = {}
        if crumb_resp.status_code == 200:
            crumb_data = crumb_resp.json()
            crumb = crumb_data['crumb']
            crumb_field = crumb_data['crumbRequestField']
            headers = {crumb_field: crumb}
        
        url = f"{JENKINS_URL}/job/{JOB_NAME}/build?token={JENKINS_TOKEN}"
        response = requests.post(url, headers=headers, timeout=30)
        
        if response.status_code == 201:
            logger.info("✅ Jenkins build triggered successfully!")
            return True
    except Exception as e:
        logger.error(f"❌ Jenkins trigger failed: {e}")
    return False

# ============================================================
# MAIN RETRAINING PIPELINE
# ============================================================

def main():
    """Main retraining pipeline with VGG16, feature engineering, and drift detection"""
    logger.info("=" * 60)
    logger.info("🔄 Starting VGG16 Retraining Pipeline with Drift Detection")
    logger.info("=" * 60)
    
    try:
        with mlflow.start_run(run_name=f"retraining_vgg16_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
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
                "model_bucket": MODEL_BUCKET,
                "data_bucket": DATA_BUCKET
            })
            
            # Step 1: Download data
            logger.info("📥 Downloading training data...")
            data_path = download_training_data()
            
            # Step 2: Split into train/val
            # (Using the existing data structure with class folders)
            train_data_path = data_path
            val_data_path = data_path  # Same path, using validation_split in flow_from_directory
            
            # Step 3: Create VGG16 model
            model = get_vgg16_base_model()
            
            # Step 4: Load validation generator for drift detection
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            
            val_datagen = ImageDataGenerator(rescale=1./255, validation_split=VALIDATION_SPLIT)
            val_generator = val_datagen.flow_from_directory(
                data_path,
                target_size=IMAGE_SIZE[:-1],
                batch_size=BATCH_SIZE,
                class_mode='categorical',
                subset='validation',
                shuffle=False
            )
            
            # Step 5: Pre-training drift detection using feature extraction
            logger.info("📊 Detecting data drift...")
            
            # Extract features from validation data for drift detection
            feature_pipeline = FeaturePipeline({
                'use_pca': True,
                'pca_components': 50
            })
            
            # Use feature pipeline to extract features
            X, y, metadata = feature_pipeline.extract_features(data_path)
            X_transformed = feature_pipeline.fit_transform(X)
            
            # Save feature pipeline
            feature_pipeline.save(Path('/tmp/feature_pipeline'))
            
            # Initialize drift detector
            drift_detector = TrainingDriftDetector(
                threshold=0.05,
                feature_names=feature_pipeline.feature_names
            )
            
            # Load reference data from S3 if exists
            try:
                s3 = boto3.client('s3')
                s3.download_file(MODEL_BUCKET, 'reference_features.npy', '/tmp/reference_features.npy')
                reference_data = np.load('/tmp/reference_features.npy')
                drift_detector.compute_reference_stats(reference_data)
            except:
                logger.info("No reference data found. Using current batch as reference.")
            
            # Detect drift
            drift_report = drift_detector.detect_drift(
                X_transformed,
                metadata,
                batch_id=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Save drift report
            drift_detector.save_report(drift_report, MODEL_BUCKET)
            
            # Log drift metrics
            mlflow.log_metrics({
                "drift_percentage": drift_report['drift_percentage'],
                "drifted_features": drift_report['drifted_features']
            })
            
            # Step 6: Train VGG16 model
            logger.info("🏋️ Training VGG16 model...")
            
            train_generator = val_datagen.flow_from_directory(
                data_path,
                target_size=IMAGE_SIZE[:-1],
                batch_size=BATCH_SIZE,
                class_mode='categorical',
                subset='training',
                shuffle=True
            )
            
            model, history = train_vgg16_model(model, data_path, data_path)
            
            # Step 7: Save model
            model_path = '/tmp/model.h5'
            model.save(model_path)
            logger.info(f"✅ Model saved to {model_path}")
            
            # Step 8: Compare models
            logger.info("⚖️ Comparing models...")
            comparison = compare_models(model, val_generator)
            
            mlflow.log_metrics({
                "new_accuracy": comparison['new_accuracy'],
                "old_accuracy": comparison['prod_accuracy'],
                "improvement": comparison['improvement']
            })
            
            # Step 9: Deploy if improved
            if comparison['should_deploy']:
                logger.info(f"✅ Model improved by {comparison['improvement']:.2f}%")
                version = upload_model_to_s3(model_path)
                trigger_jenkins()
                mlflow.log_param("deployed_version", version)
                mlflow.log_param("deployed", True)
            else:
                logger.info(f"⏸️ No significant improvement: {comparison['improvement']:.2f}%")
                mlflow.log_param("deployed", False)
            
            # Step 10: Save reference data for future
            np.save('/tmp/reference_features.npy', X_transformed)
            s3 = boto3.client('s3')
            s3.upload_file('/tmp/reference_features.npy', MODEL_BUCKET, 'reference_features.npy')
            
            logger.info("=" * 60)
            logger.info("✅ Retraining pipeline completed successfully!")
            logger.info("=" * 60)
            
    except Exception as e:
        logger.error(f"❌ Retraining failed: {e}")
        raise


if __name__ == "__main__":
    main()