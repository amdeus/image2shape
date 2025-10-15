#!/usr/bin/env python3
"""
Abstract base class for ML algorithms in Image2Shape

This module provides the foundation for extensible ML algorithm support,
enabling future integration of KNN, CNN, YOLO, and other algorithms while
maintaining backward compatibility with the existing Random Forest implementation.

Key Features:
- Abstract interface for all ML algorithms
- Standardized training and prediction methods
- Model persistence and metadata management
- Performance metrics and validation
- Clean integration with existing codebase

Author: Image2Shape Development Team
Version: 2.1 - Multi-Algorithm Foundation
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from datetime import datetime

class MLAlgorithmBase(ABC):
    """
    Abstract base class for all ML algorithms in Image2Shape
    
    This class defines the standard interface that all ML algorithms must implement,
    ensuring consistent behavior across different algorithm types while allowing
    for algorithm-specific optimizations and features.
    
    Key Design Principles:
    - Consistent API across all algorithms
    - Standardized training and prediction workflow
    - Built-in performance tracking and validation
    - Model persistence with metadata
    - Integration with existing feature extraction pipeline
    """
    
    def __init__(self, algorithm_name: str, **kwargs):
        """
        Initialize the ML algorithm
        
        Args:
            algorithm_name: Human-readable name for the algorithm
            **kwargs: Algorithm-specific parameters
        """
        self.algorithm_name = algorithm_name
        self.is_trained = False
        self.feature_names = []
        self.training_history = []
        self.model_info = {
            'algorithm_name': algorithm_name,
            'created': datetime.now().isoformat(),
            'last_trained': None,
            'n_samples': 0,
            'n_features': 0,
            'accuracy': 0.0,
            'parameters': kwargs
        }
        
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Train the ML algorithm
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,) - 0 for background, 1 for feature
            feature_names: List of feature names for analysis
            
        Returns:
            Dictionary with training results and metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes for new samples
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predicted classes (0=background, 1=feature)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Class probabilities (n_samples, 2) - [prob_background, prob_feature]
        """
        pass
    
    @abstractmethod
    def save_model(self, filepath: str):
        """
        Save the trained model and metadata
        
        Args:
            filepath: Path to save the model (without extension)
        """
        pass
    
    @abstractmethod
    def load_model(self, filepath: str):
        """
        Load a previously trained model
        
        Args:
            filepath: Path to the saved model (without extension)
        """
        pass
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get algorithm information and capabilities"""
        return {
            'name': self.algorithm_name,
            'is_trained': self.is_trained,
            'supports_probabilities': True,
            'supports_feature_importance': hasattr(self, 'get_feature_importance'),
            'training_time_estimate': self._get_training_time_estimate(),
            'memory_requirements': self._get_memory_requirements(),
            'best_use_cases': self._get_best_use_cases()
        }
    
    def _get_training_time_estimate(self) -> str:
        """Get estimated training time for this algorithm"""
        return "Variable"
    
    def _get_memory_requirements(self) -> str:
        """Get memory requirements for this algorithm"""
        return "Standard"
    
    def _get_best_use_cases(self) -> List[str]:
        """Get list of best use cases for this algorithm"""
        return ["General purpose"]
    
    def get_model_summary(self) -> str:
        """Generate a summary of the current model"""
        if not self.is_trained:
            return f"{self.algorithm_name} - Not trained yet"
        
        summary = f"""{self.algorithm_name} Model Summary:

Training Information:
- Algorithm: {self.algorithm_name}
- Created: {self.model_info.get('created', 'Unknown')}
- Last Trained: {self.model_info.get('last_trained', 'Unknown')}
- Training Samples: {self.model_info.get('n_samples', 0):,}
- Features: {self.model_info.get('n_features', 0)}

Performance:
- Training Accuracy: {self.model_info.get('accuracy', 0):.3f}
- Training Sessions: {len(self.training_history)}

Status: {'Trained' if self.is_trained else 'Not Trained'}"""
        
        return summary
    
    def validate_training_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[bool, str]:
        """
        Validate training data before training
        
        Args:
            X: Feature matrix
            y: Labels
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(X) == 0:
            return False, "No training data provided"
        
        if len(np.unique(y)) < 2:
            return False, "Need at least 2 classes for training"
        
        if len(X) != len(y):
            return False, "Feature matrix and labels must have same length"
        
        # Algorithm-specific validation can be added in subclasses
        return True, ""
    
    def update_training_history(self, training_results: Dict[str, Any]):
        """Update training history with new session"""
        session = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': training_results.get('n_samples', 0),
            'accuracy': training_results.get('accuracy', 0),
            'parameters': training_results.get('parameters', {})
        }
        self.training_history.append(session)
        
        # Update model info
        self.model_info.update({
            'last_trained': session['timestamp'],
            'n_samples': session['n_samples'],
            'accuracy': session['accuracy']
        })