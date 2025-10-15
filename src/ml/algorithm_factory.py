#!/usr/bin/env python3
"""
ML Algorithm Factory for Image2Shape

This module provides a factory pattern for creating ML algorithm instances,
enabling easy extension with new algorithms while maintaining clean code
organization and consistent interfaces.

Key Features:
- Factory pattern for algorithm instantiation
- Algorithm registration and discovery
- Parameter validation and defaults
- Algorithm capability information
- Future-ready for new algorithm types

Author: Image2Shape Development Team
Version: 2.1 - Multi-Algorithm Foundation
"""

from typing import Dict, Any, List, Type, Optional
from .base_algorithm import MLAlgorithmBase
from .random_forest_algorithm import RandomForestAlgorithm
from .knn_algorithm import KNNAlgorithm

class MLAlgorithmFactory:
    """
    Factory class for creating ML algorithm instances
    
    This factory provides a centralized way to create and manage different
    ML algorithms, making it easy to add new algorithms in the future while
    maintaining consistent interfaces and parameter handling.
    """
    
    # Registry of available algorithms
    _algorithms = {
        'random_forest': {
            'class': RandomForestAlgorithm,
            'name': 'Random Forest',
            'description': 'Ensemble method with decision trees - Fast, reliable, interpretable',
            'training_time': 'Fast (30s)',
            'memory_usage': 'Low (< 2GB)',
            'best_for': ['White/black markers', 'High contrast features', 'Small datasets'],
            'parameters': {
                'n_estimators': {'type': int, 'default': 100, 'range': [10, 500], 'description': 'Number of trees'},
                'random_state': {'type': int, 'default': 42, 'range': [1, 1000], 'description': 'Random seed'}
            }
        },
        'knn': {
            'class': KNNAlgorithm,
            'name': 'K-Nearest Neighbors',
            'description': 'Instance-based learning - Excellent for similar patterns and non-linear boundaries',
            'training_time': 'Instant',
            'memory_usage': 'Medium (stores training data)',
            'best_for': ['Similar pattern detection', 'Non-linear boundaries', 'Local patterns'],
            'parameters': {
                'n_neighbors': {'type': int, 'default': 5, 'range': [1, 50], 'description': 'Number of neighbors'},
                'weights': {'type': str, 'default': 'uniform', 'options': ['uniform', 'distance'], 'description': 'Weight function'},
                'metric': {'type': str, 'default': 'euclidean', 'options': ['euclidean', 'manhattan', 'minkowski'], 'description': 'Distance metric'}
            }
        }
        # Future algorithms:
        # 'cnn': {...},
        # 'yolo': {...}
    }
    
    @classmethod
    def get_available_algorithms(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available algorithms
        
        Returns:
            Dictionary with algorithm information
        """
        return cls._algorithms.copy()
    
    @classmethod
    def create_algorithm(cls, algorithm_type: str, **kwargs) -> MLAlgorithmBase:
        """
        Create an instance of the specified algorithm
        
        Args:
            algorithm_type: Type of algorithm to create
            **kwargs: Algorithm-specific parameters
            
        Returns:
            Initialized algorithm instance
            
        Raises:
            ValueError: If algorithm type is not supported
        """
        if algorithm_type not in cls._algorithms:
            available = list(cls._algorithms.keys())
            raise ValueError(f"Algorithm '{algorithm_type}' not supported. Available: {available}")
        
        algorithm_info = cls._algorithms[algorithm_type]
        algorithm_class = algorithm_info['class']
        
        # Apply default parameters
        params = cls._get_default_parameters(algorithm_type)
        params.update(kwargs)
        
        # Validate parameters
        cls._validate_parameters(algorithm_type, params)
        
        # Create and return algorithm instance
        return algorithm_class(**params)
    
    @classmethod
    def _get_default_parameters(cls, algorithm_type: str) -> Dict[str, Any]:
        """Get default parameters for an algorithm"""
        algorithm_info = cls._algorithms[algorithm_type]
        defaults = {}
        
        for param_name, param_info in algorithm_info.get('parameters', {}).items():
            defaults[param_name] = param_info['default']
        
        return defaults
    
    @classmethod
    def _validate_parameters(cls, algorithm_type: str, params: Dict[str, Any]):
        """Validate parameters for an algorithm"""
        algorithm_info = cls._algorithms[algorithm_type]
        param_definitions = algorithm_info.get('parameters', {})
        
        for param_name, param_value in params.items():
            if param_name in param_definitions:
                param_def = param_definitions[param_name]
                
                # Type validation
                expected_type = param_def['type']
                if not isinstance(param_value, expected_type):
                    raise ValueError(f"Parameter '{param_name}' must be of type {expected_type.__name__}")
                
                # Range validation
                if 'range' in param_def:
                    min_val, max_val = param_def['range']
                    if not (min_val <= param_value <= max_val):
                        raise ValueError(f"Parameter '{param_name}' must be between {min_val} and {max_val}")
    
    @classmethod
    def get_algorithm_info(cls, algorithm_type: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific algorithm
        
        Args:
            algorithm_type: Type of algorithm
            
        Returns:
            Algorithm information dictionary
        """
        if algorithm_type not in cls._algorithms:
            raise ValueError(f"Algorithm '{algorithm_type}' not supported")
        
        return cls._algorithms[algorithm_type].copy()
    
    @classmethod
    def register_algorithm(cls, algorithm_type: str, algorithm_class: Type[MLAlgorithmBase], 
                          algorithm_info: Dict[str, Any]):
        """
        Register a new algorithm type
        
        Args:
            algorithm_type: Unique identifier for the algorithm
            algorithm_class: Class implementing the algorithm
            algorithm_info: Information about the algorithm
        """
        if not issubclass(algorithm_class, MLAlgorithmBase):
            raise ValueError("Algorithm class must inherit from MLAlgorithmBase")
        
        cls._algorithms[algorithm_type] = {
            'class': algorithm_class,
            **algorithm_info
        }
    
    @classmethod
    def get_recommended_algorithm(cls, use_case: str = "general") -> str:
        """
        Get recommended algorithm for a specific use case
        
        Args:
            use_case: Description of the use case
            
        Returns:
            Recommended algorithm type
        """
        # For now, always recommend Random Forest as it's the only available algorithm
        # Future implementation will analyze use case and recommend best algorithm
        return 'random_forest'


# Convenience function for backward compatibility
def create_default_classifier() -> MLAlgorithmBase:
    """
    Create a default classifier instance for backward compatibility
    
    Returns:
        Default Random Forest algorithm instance
    """
    return MLAlgorithmFactory.create_algorithm('random_forest')


# Convenience function for direct Random Forest creation
def create_random_forest(**kwargs) -> RandomForestAlgorithm:
    """
    Create a Random Forest algorithm instance
    
    Args:
        **kwargs: Random Forest parameters
        
    Returns:
        Random Forest algorithm instance
    """
    return MLAlgorithmFactory.create_algorithm('random_forest', **kwargs)