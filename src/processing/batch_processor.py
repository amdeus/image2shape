#!/usr/bin/env python3
"""
Batch processor for large-scale drone image analysis
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime
import json
import time

class BatchProcessor:
    """
    High-performance batch processor for large-scale drone image analysis
    
    This class provides a complete framework for processing hundreds or thousands
    of drone images using trained machine learning models. It's optimized for
    memory efficiency and provides real-time progress feedback.
    
    Key Features:
    - Scalable processing: Handles 1000+ images efficiently
    - Memory management: Processes images individually to avoid memory overflow
    - Progress tracking: Real-time progress updates with ETA calculation
    - Error handling: Graceful handling of corrupted or invalid images
    - Result aggregation: Comprehensive statistics and result compilation
    
    Processing Pipeline:
    1. Load and validate each image
    2. Extract pixel features using grid sampling
    3. Apply trained ML model for classification
    4. Collect feature coordinates and confidence scores
    5. Aggregate results and generate statistics
    
    Performance Targets:
    - Processing speed: 1-2 seconds per 44MP image
    - Memory usage: <4GB total for large batches
    - Error tolerance: Continue processing despite individual image failures
    - Progress feedback: Sub-second updates with accurate ETA
    """
    
    def __init__(self, feature_extractor, classifier):
        """Initialize batch processor"""
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.results = {}
        self.processing_stats = {
            'total_images': 0,
            'processed_images': 0,
            'total_pixels_classified': 0,
            'total_features_detected': 0,
            'start_time': None,
            'end_time': None,
            'processing_time': 0
        }
    
    def process_all_images(self, 
                          image_folder: str, 
                          image_files: List[str],
                          progress_callback: Optional[Callable] = None,
                          max_pixels_per_image: int = 5000) -> Dict:
        """Process all images in batch for feature detection"""
        # Starting batch processing - progress will be tracked via callback
        
        self.processing_stats['total_images'] = len(image_files)
        self.processing_stats['start_time'] = datetime.now()
        self.results = {}
        
        for i, image_file in enumerate(image_files):
            try:
                if progress_callback:
                    progress = (i / len(image_files)) * 100
                    eta = self._calculate_eta(i, len(image_files))
                    progress_callback(progress, f"Processing {image_file}", eta)
                
                image_path = os.path.join(image_folder, image_file)
                result = self._process_single_image(image_path, max_pixels_per_image)
                
                if result:
                    self.results[image_path] = result
                    self.processing_stats['processed_images'] += 1
                    self.processing_stats['total_pixels_classified'] += result['total_pixels']
                    self.processing_stats['total_features_detected'] += result['feature_count']
                
            except Exception as e:
                # Error processing image - continue with next image
                continue
        
        self.processing_stats['end_time'] = datetime.now()
        self.processing_stats['processing_time'] = (
            self.processing_stats['end_time'] - self.processing_stats['start_time']
        ).total_seconds()
        
        if progress_callback:
            progress_callback(100, "Batch processing complete!", "0:00")
        
        return {'results': self.results, 'stats': self.processing_stats}
    
    def _process_single_image(self, image_path: str, max_pixels: int) -> Optional[Dict]:
        """Process a single image for feature detection"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            features, coordinates = self.feature_extractor.extract_pixel_features_from_image_grid(
                image_rgb, max_pixels=max_pixels
            )
            
            if len(features) == 0:
                return None
            
            predictions = self.classifier.predict(features)
            probabilities = self.classifier.predict_proba(features)
            
            feature_indices = np.where(predictions == 1)[0]
            feature_coordinates = [coordinates[i] for i in feature_indices]
            feature_probabilities = probabilities[feature_indices]
            confidence_scores = np.max(feature_probabilities, axis=1)
            
            return {
                'image_path': image_path,
                'total_pixels': len(features),
                'feature_count': len(feature_coordinates),
                'feature_coordinates': feature_coordinates,
                'confidence_scores': confidence_scores.tolist(),
                'image_dimensions': image_rgb.shape[:2],
                'processing_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            # Error processing individual image
            return None
    
    def _calculate_eta(self, current_index: int, total_images: int) -> str:
        """Calculate estimated time remaining"""
        if current_index == 0:
            return "Calculating..."
        
        elapsed = (datetime.now() - self.processing_stats['start_time']).total_seconds()
        avg_time_per_image = elapsed / current_index
        remaining_images = total_images - current_index
        eta_seconds = remaining_images * avg_time_per_image
        
        minutes = int(eta_seconds // 60)
        seconds = int(eta_seconds % 60)
        return f"{minutes}:{seconds:02d}"
    
    
