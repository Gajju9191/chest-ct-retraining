#!/usr/bin/env python3
"""
Feature engineering pipeline for chest CT scans.
Extracts radiomic and deep features for model training.
Uses VGG16 for deep feature extraction (consistent with deployment).
"""
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from skimage.feature import graycomatrix, graycoprops
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Feature extraction pipeline for chest CT images.
    
    Extracts:
    1. Deep features (VGG16 - consistent with deployment)
    2. Radiomic features (shape, intensity, texture)
    3. Combines and reduces dimensions with PCA
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.base_model = None
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_names = []
        self.is_fitted = False
        
    def _get_base_model(self):
        """Load pre-trained VGG16 for feature extraction"""
        if self.base_model is None:
            self.base_model = VGG16(
                weights='imagenet',
                include_top=False,
                pooling='avg'
            )
            logger.info("✅ Loaded VGG16 for feature extraction")
        return self.base_model
    
    def extract_deep_features(self, image_path: Path) -> Optional[np.ndarray]:
        """Extract deep features using VGG16"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            img = cv2.resize(img, (224, 224))
            img = preprocess_input(img)
            img = np.expand_dims(img, axis=0)
            
            model = self._get_base_model()
            features = model.predict(img, verbose=0).flatten()
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting deep features from {image_path}: {e}")
            return None
    
    def extract_radiomic_features(self, image_path: Path) -> Dict[str, float]:
        """Extract radiomic features from image"""
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return {}
            
            features = {}
            
            # Shape features
            binary = (img > 0).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest = max(contours, key=cv2.contourArea)
                features['area'] = cv2.contourArea(largest)
                features['perimeter'] = cv2.arcLength(largest, True)
                features['compactness'] = (4 * np.pi * features['area']) / (features['perimeter']**2 + 1e-10)
            else:
                features['area'] = 0
                features['perimeter'] = 0
                features['compactness'] = 0
            
            # Intensity features
            non_zero = img[img > 0]
            if len(non_zero) > 0:
                features['mean_intensity'] = np.mean(non_zero)
                features['std_intensity'] = np.std(non_zero)
                features['skew'] = pd.Series(non_zero).skew()
                features['kurtosis'] = pd.Series(non_zero).kurtosis()
            else:
                features['mean_intensity'] = 0
                features['std_intensity'] = 0
                features['skew'] = 0
                features['kurtosis'] = 0
            
            # Texture features (GLCM)
            try:
                glcm = graycomatrix(img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                  levels=256, symmetric=True)
                features['contrast'] = np.mean(graycoprops(glcm, 'contrast'))
                features['energy'] = np.mean(graycoprops(glcm, 'energy'))
                features['homogeneity'] = np.mean(graycoprops(glcm, 'homogeneity'))
                features['correlation'] = np.mean(graycoprops(glcm, 'correlation'))
            except:
                features['contrast'] = 0
                features['energy'] = 0
                features['homogeneity'] = 0
                features['correlation'] = 0
            
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting radiomic features from {image_path}: {e}")
            return {}
    
    def extract_features(self, data_path: Path) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Extract all features from dataset
        
        Returns:
            Tuple of (features, labels, metadata)
        """
        image_paths = []
        labels = []
        
        # ✅ FIX: Recursively find all images in subdirectories
        logger.info(f"Searching for images in: {data_path}")
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = Path(root) / file
                    image_paths.append(full_path)
                    # Use parent directory name as label
                    labels.append(Path(root).name)
                    logger.debug(f"Found image: {full_path} with label: {Path(root).name}")
        
        # ✅ FIX: If no images found, try different directory depth
        if len(image_paths) == 0:
            logger.warning(f"No images found in {data_path}, trying alternative search...")
            # Try one level deeper
            for subdir in data_path.iterdir():
                if subdir.is_dir():
                    for img_path in subdir.glob('*.[jp][pn][g]'):
                        image_paths.append(img_path)
                        labels.append(subdir.name)
        
        logger.info(f"Found {len(image_paths)} images")
        
        # ✅ FIX: Handle case with no images
        if len(image_paths) == 0:
            logger.error(f"No images found in {data_path}")
            return np.array([]), np.array([]), pd.DataFrame()
        
        # Extract features
        all_features = []
        metadata = []
        
        for img_path in image_paths:
            # Deep features
            deep = self.extract_deep_features(img_path)
            if deep is None:
                continue
            
            # Radiomic features
            radio = self.extract_radiomic_features(img_path)
            
            # Combine
            combined = np.concatenate([deep, list(radio.values())])
            all_features.append(combined)
            
            metadata.append({
                'image_id': img_path.stem,
                'path': str(img_path),
                'label': img_path.parent.name,
                'size': img_path.stat().st_size
            })
        
        # ✅ FIX: Check if any features were extracted
        if len(all_features) == 0:
            logger.error("No features could be extracted from images")
            return np.array([]), np.array([]), pd.DataFrame()
        
        # Convert to arrays
        X = np.array(all_features)
        y = np.array([m['label'] for m in metadata])
        metadata_df = pd.DataFrame(metadata)
        
        # Store feature names
        deep_length = len(deep) if 'deep' in locals() else all_features[0].shape[0] - len(radio)
        self.feature_names = [f'deep_{i}' for i in range(deep_length)]
        self.feature_names.extend(list(radio.keys()))
        
        logger.info(f"Extracted {X.shape[1]} features from {X.shape[0]} images")
        
        return X, y, metadata_df
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit scaler and PCA, then transform features"""
        if X is None or len(X) == 0:
            logger.error("No data to fit")
            return np.array([])
            
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA if configured
        if self.config.get('use_pca', True) and X_scaled.shape[1] > 0:
            n_components = min(self.config.get('pca_components', 50), X_scaled.shape[1])
            self.pca = PCA(n_components=n_components)
            X_transformed = self.pca.fit_transform(X_scaled)
            logger.info(f"PCA reduced dimensions from {X_scaled.shape[1]} to {n_components}")
            logger.info(f"Explained variance: {self.pca.explained_variance_ratio_.sum():.2%}")
        else:
            X_transformed = X_scaled
        
        self.is_fitted = True
        return X_transformed
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler and PCA"""
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit_transform first.")
        
        if X is None or len(X) == 0:
            return np.array([])
            
        X_scaled = self.scaler.transform(X)
        if self.pca:
            X_transformed = self.pca.transform(X_scaled)
        else:
            X_transformed = X_scaled
        
        return X_transformed
    
    def save(self, output_dir: Path):
        """Save fitted transformers"""
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, output_dir / 'scaler.pkl')
        if self.pca:
            joblib.dump(self.pca, output_dir / 'pca.pkl')
        with open(output_dir / 'feature_names.txt', 'w') as f:
            f.write('\n'.join(self.feature_names))
        logger.info(f"✅ Saved feature pipeline to {output_dir}")
    
    def load(self, output_dir: Path):
        """Load fitted transformers"""
        self.scaler = joblib.load(output_dir / 'scaler.pkl')
        pca_path = output_dir / 'pca.pkl'
        if pca_path.exists():
            self.pca = joblib.load(pca_path)
        with open(output_dir / 'feature_names.txt', 'r') as f:
            self.feature_names = f.read().split('\n')
        self.is_fitted = True
        logger.info(f"✅ Loaded feature pipeline from {output_dir}")