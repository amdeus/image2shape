#!/usr/bin/env python3
"""
Random Forest algorithm implementation for Image2Shape

This module wraps the existing DroneImageClassifier in the new ML architecture
framework, providing backward compatibility while enabling future algorithm
extensions.

Key Features:
- Wraps existing Random Forest implementation
- Maintains all current functionality
- Provides standardized interface
- Zero breaking changes to existing code
- Enhanced metadata and performance tracking

Author: Image2Shape Development Team
Version: 2.1 - Multi-Algorithm Foundation
"""

import numpy as np
import os
from typing import Dict, Any, List, Tuple
from .base_algorithm import MLAlgorithmBase
from .random_forest_classifier import DroneImageClassifier

class RandomForestAlgorithm(MLAlgorithmBase):
    """
    Random Forest implementation wrapper for the existing DroneImageClassifier
    
    This class provides a clean interface to the existing Random Forest
    implementation while conforming to the new ML algorithm architecture.
    It maintains 100% backward compatibility with existing functionality.
    """
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42, **kwargs):
        """
        Initialize Random Forest algorithm
        
        Args:
            n_estimators: Number of trees in the forest
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters for the Random Forest
        """
        super().__init__("Random Forest", 
                        n_estimators=n_estimators, 
                        random_state=random_state, 
                        **kwargs)
        
        # Initialize the underlying Random Forest classifier
        self.rf_classifier = DroneImageClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
        
        # Sync training state
        self.is_trained = self.rf_classifier.is_trained
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Train the Random Forest classifier
        
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
        
        # Train using existing implementation
        training_results = self.rf_classifier.train(X, y, feature_names)
        
        # Update our training state
        self.is_trained = self.rf_classifier.is_trained
        
        # Update training history
        self.update_training_history(training_results)
        
        # Add algorithm-specific information
        training_results['algorithm'] = self.algorithm_name
        training_results['algorithm_info'] = self.get_algorithm_info()
        
        return training_results
    
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
        
        return self.rf_classifier.predict(X)
    
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
        
        return self.rf_classifier.predict_proba(X)
    
    def save_model(self, filepath: str):
        """
        Save the trained model and metadata
        
        Args:
            filepath: Path to save the model (without extension)
        """
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        # Save using existing implementation
        self.rf_classifier.save_model(filepath)
        
        # Save additional algorithm metadata
        import joblib
        algorithm_metadata = {
            'algorithm_name': self.algorithm_name,
            'algorithm_info': self.get_algorithm_info(),
            'training_history': self.training_history,
            'model_info': self.model_info
        }
        
        metadata_file = f"{filepath}_algorithm_metadata.joblib"
        joblib.dump(algorithm_metadata, metadata_file)
        
        print(f"{self.algorithm_name} model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a previously trained model
        
        Args:
            filepath: Path to the saved model (without extension)
        """
        # Load using existing implementation
        self.rf_classifier.load_model(filepath)
        
        # Update our training state
        self.is_trained = self.rf_classifier.is_trained
        
        # Load additional algorithm metadata if available
        metadata_file = f"{filepath}_algorithm_metadata.joblib"
        if os.path.exists(metadata_file):
            import joblib
            try:
                algorithm_metadata = joblib.load(metadata_file)
                self.training_history = algorithm_metadata.get('training_history', [])
                self.model_info.update(algorithm_metadata.get('model_info', {}))
            except Exception as e:
                print(f"Warning: Could not load algorithm metadata: {e}")
        
        print(f"{self.algorithm_name} model loaded from {filepath}")
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        Get the most important features for classification
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        return self.rf_classifier.get_feature_importance(top_n)
    
    def _get_training_time_estimate(self) -> str:
        """Get estimated training time for Random Forest"""
        return "Fast (30s for 30k samples)"
    
    def _get_memory_requirements(self) -> str:
        """Get memory requirements for Random Forest"""
        return "Low (< 2GB)"
    
    def _get_best_use_cases(self) -> List[str]:
        """Get list of best use cases for Random Forest"""
        return [
            "White/black marker detection",
            "High-contrast features",
            "Small training datasets",
            "Fast training required",
            "Feature importance analysis"
        ]
    
    def retrain(self, X_new: np.ndarray, y_new: np.ndarray, X_old: np.ndarray = None, y_old: np.ndarray = None) -> Dict[str, Any]:
        """
        Retrain the model with new data, optionally combining with old data
        
        Args:
            X_new: New feature matrix
            y_new: New labels
            X_old: Previous feature matrix (optional)
            y_old: Previous labels (optional)
            
        Returns:
            Training results dictionary
        """
        # Use existing retrain functionality
        training_results = self.rf_classifier.retrain(X_new, y_new, X_old, y_old)
        
        # Update our training state
        self.is_trained = self.rf_classifier.is_trained
        
        # Update training history
        self.update_training_history(training_results)
        
        return training_results