#!/usr/bin/env python3
"""
K-Nearest Neighbors algorithm implementation for Image2Shape

This module implements KNN for pixel-based classification in drone imagery,
providing an alternative to Random Forest with different strengths and
characteristics.

Key Features:
- Instance-based learning (no training phase)
- Excellent for similar pattern detection
- Non-linear decision boundaries
- Memory-based predictions
- Configurable distance metrics

Author: Image2Shape Development Team
Version: 2.1 - Multi-Algorithm Step 2
"""

import numpy as np
import os
from typing import Dict, Any, List, Tuple
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from .base_algorithm import MLAlgorithmBase
import joblib
from datetime import datetime

class KNNAlgorithm(MLAlgorithmBase):
    """
    K-Nearest Neighbors implementation for drone image feature detection
    
    KNN is an instance-based learning algorithm that makes predictions based on
    the k nearest neighbors in the feature space. It's particularly effective
    for datasets where similar patterns should be classified similarly.
    
    Advantages:
    - No training phase (instant "training")
    - Excellent for non-linear patterns
    - Adapts well to local patterns
    - Simple and interpretable
    
    Best Use Cases:
    - Similar pattern detection
    - Non-linear feature boundaries
    - Small to medium datasets
    - When training time is critical
    """
    
    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform', 
                 metric: str = 'euclidean', **kwargs):
        """
        Initialize KNN algorithm
        
        Args:
            n_neighbors: Number of neighbors to consider
            weights: Weight function ('uniform' or 'distance')
            metric: Distance metric ('euclidean', 'manhattan', etc.)
            **kwargs: Additional parameters
        """
        super().__init__("K-Nearest Neighbors", 
                        n_neighbors=n_neighbors, 
                        weights=weights, 
                        metric=metric, 
                        **kwargs)
        
        # Initialize the KNN classifier
        self.knn_classifier = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            n_jobs=-1  # Use all available cores
        )
        
        # Store training data for memory-based predictions
        self.X_train = None
        self.y_train = None
        
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> Dict[str, Any]:
        """
        "Train" the KNN classifier (actually just stores the data)
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,) - 0 for background, 1 for feature
            feature_names: List of feature names for analysis
            
        Returns:
            Dictionary with training results and metrics
        """
        # Validate training data
        is_valid, error_msg = self.validate_training_data(X, y)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Store feature names
        if feature_names:
            self.feature_names = feature_names
        
        
        start_time = datetime.now()
        
        # "Train" KNN (just fits the data)
        self.knn_classifier.fit(X, y)
        self.is_trained = True
        
        # Store training data for analysis
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        # Calculate training metrics (using cross-validation for KNN)
        from sklearn.model_selection import cross_val_score
        
        # Use smaller CV for KNN as it can be computationally expensive
        cv_folds = min(3, len(X) // 10) if len(X) >= 30 else 2
        cv_scores = cross_val_score(self.knn_classifier, X, y, cv=cv_folds, n_jobs=-1)
        
        # Training accuracy (perfect for KNN on training data)
        y_pred = self.knn_classifier.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # Update model info
        self.model_info.update({
            'last_trained': end_time.isoformat(),
            'n_samples': len(X),
            'n_features': X.shape[1],
            'accuracy': accuracy,
            'cv_accuracy': np.mean(cv_scores),
            'training_time': training_time
        })
        
        if self.model_info['created'] is None:
            self.model_info['created'] = self.model_info['last_trained']
        
        # Store training session
        training_session = {
            'timestamp': self.model_info['last_trained'],
            'n_samples': len(X),
            'accuracy': accuracy,
            'cv_accuracy': np.mean(cv_scores),
            'class_distribution': np.bincount(y.astype(int)),
            'training_time': training_time
        }
        self.training_history.append(training_session)
        
        # Generate detailed results
        results = {
            'accuracy': accuracy,
            'cv_accuracy': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'classification_report': classification_report(y, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'class_distribution': np.bincount(y.astype(int)),
            'n_samples': len(X),
            'n_features': X.shape[1],
            'training_time': training_time,
            'algorithm': self.algorithm_name
        }
        
        print(f"KNN training complete - Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f} ({training_time:.1f}s)")
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes for new samples
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predicted classes (0=background, 1=feature)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.knn_classifier.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Class probabilities (n_samples, 2) - [prob_background, prob_feature]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.knn_classifier.predict_proba(X)
    
    def save_model(self, filepath: str):
        """
        Save the trained model and metadata
        
        Args:
            filepath: Path to save the model (without extension)
        """
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save KNN model
        model_file = f"{filepath}.joblib"
        joblib.dump(self.knn_classifier, model_file)
        
        # Save training data (required for KNN)
        training_data_file = f"{filepath}_training_data.joblib"
        training_data = {
            'X_train': self.X_train,
            'y_train': self.y_train
        }
        joblib.dump(training_data, training_data_file)
        
        # Save metadata
        metadata = {
            'algorithm_name': self.algorithm_name,
            'model_info': self.model_info,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'model_params': self.knn_classifier.get_params()
        }
        
        metadata_file = f"{filepath}_metadata.joblib"
        joblib.dump(metadata, metadata_file)
        
        print(f"KNN model saved to {model_file}")
    
    def load_model(self, filepath: str):
        """
        Load a previously trained model
        
        Args:
            filepath: Path to the saved model (without extension)
        """
        model_file = f"{filepath}.joblib"
        training_data_file = f"{filepath}_training_data.joblib"
        metadata_file = f"{filepath}_metadata.joblib"
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # Load KNN model
        self.knn_classifier = joblib.load(model_file)
        self.is_trained = True
        
        # Load training data
        if os.path.exists(training_data_file):
            training_data = joblib.load(training_data_file)
            self.X_train = training_data['X_train']
            self.y_train = training_data['y_train']
        
        # Load metadata if available
        if os.path.exists(metadata_file):
            try:
                metadata = joblib.load(metadata_file)
                self.model_info = metadata.get('model_info', {})
                self.feature_names = metadata.get('feature_names', [])
                self.training_history = metadata.get('training_history', [])
            except Exception as e:
                print(f"Warning: Could not load metadata: {e}")
        
        print(f"KNN model loaded from {model_file}")
    
    def get_neighbor_analysis(self, X_query: np.ndarray, n_examples: int = 5) -> Dict[str, Any]:
        """
        Analyze nearest neighbors for given query points
        
        Args:
            X_query: Query points to analyze
            n_examples: Number of examples to return
            
        Returns:
            Dictionary with neighbor analysis
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Get distances and indices of nearest neighbors
        distances, indices = self.knn_classifier.kneighbors(X_query, n_neighbors=n_examples)
        
        analysis = {
            'query_points': len(X_query),
            'neighbors_per_query': n_examples,
            'average_distances': np.mean(distances, axis=1),
            'neighbor_classes': self.y_train[indices],
            'neighbor_distances': distances
        }
        
        return analysis
    
    def _get_training_time_estimate(self) -> str:
        """Get estimated training time for KNN"""
        return "Instant (no training phase)"
    
    def _get_memory_requirements(self) -> str:
        """Get memory requirements for KNN"""
        return "Medium (stores all training data)"
    
    def _get_best_use_cases(self) -> List[str]:
        """Get list of best use cases for KNN"""
        return [
            "Similar pattern detection",
            "Non-linear boundaries",
            "Small to medium datasets",
            "Instant training required",
            "Local pattern adaptation"
        ]
    
    def get_algorithm_specific_info(self) -> Dict[str, Any]:
        """Get KNN-specific information"""
        info = {
            'n_neighbors': self.knn_classifier.n_neighbors,
            'weights': self.knn_classifier.weights,
            'metric': self.knn_classifier.metric,
            'memory_based': True,
            'training_samples_stored': len(self.X_train) if self.X_train is not None else 0
        }
        
        if self.is_trained and self.X_train is not None:
            info['memory_usage_mb'] = (self.X_train.nbytes + self.y_train.nbytes) / (1024 * 1024)
        
        return info