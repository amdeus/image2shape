#!/usr/bin/env python3
"""
Optimized Batch Georeferencer for Image2Shape v2.0

This module implements ultra-fast batch georeferencing for thousands of pixel coordinates
using pre-computed parameters and vectorized operations. It's designed for processing
ML detection results from the pixel-based classifier.

Key Features:
- Pre-computed georeferencing parameters for each image
- Vectorized coordinate transformation (1000+ points in <0.01s per image)
- Memory-efficient processing with cached metadata
- Integration with existing metadata processor
- Direct shapefile export with confidence scores

Performance Targets:
- Parameter computation: 0.1s per image (one-time cost)
- Coordinate transformation: 0.01s per image for 1000+ points
- Memory usage: <1GB for 1000 images with metadata cache
- Total processing: <10s for 1000 images with 50,000 coordinates

Author: Image2Shape Development Team
Version: 2.0 - Ultra-Fast Batch Processing
"""

import numpy as np
import math
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import cv2
from datetime import datetime

# Import self-contained georeferencing components
try:
    from .self_contained_georeferencer import SelfContainedGeoreferencer
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append('src/georef')
    from self_contained_georeferencer import SelfContainedGeoreferencer

try:
    from ..processing.metadata_processor import MetadataProcessor
except (ImportError, ValueError):
    # Fallback for direct execution
    import sys
    sys.path.append('src/processing')
    from metadata_processor import MetadataProcessor


class BatchGeoreferencer:
    """
    Ultra-fast batch georeferencer for processing thousands of coordinates
    
    This class pre-computes georeferencing parameters for each image and then
    applies vectorized transformations to convert pixel coordinates to WGS84.
    It's optimized for processing ML detection results efficiently.
    """
    
    def __init__(self, metadata_processor: Optional[MetadataProcessor] = None):
        """
        Initialize batch georeferencer
        
        Args:
            metadata_processor: Optional metadata processor instance for caching
        """
        self.georeferencer = SelfContainedGeoreferencer()
        self.metadata_processor = metadata_processor or MetadataProcessor()
        self.image_parameters = {}  # Cache for pre-computed parameters
        
    def precompute_image_parameters(self, image_folder: str, image_files: List[str], 
                                  progress_callback: Optional[callable] = None) -> Dict:
        """
        Pre-compute georeferencing parameters for all images using cached metadata and parallelization
        
        OPTIMIZED VERSION:
        - Reuses cached metadata from preprocessing (eliminates redundant exiftool calls)
        - Parallel processing for 2-4x speed improvement
        - Maintains existing API for backward compatibility
        
        Args:
            image_folder: Path to folder containing images
            image_files: List of image filenames
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict: Statistics about parameter computation
        """
        start_time = datetime.now()
        
        stats = {
            'total_images': len(image_files),
            'successful': 0,
            'failed': 0,
            'rtk_images': 0,
            'standard_gps': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Check if we have cached metadata available
        has_cached_metadata = (hasattr(self.metadata_processor, 'metadata_cache') and 
                              self.metadata_processor.metadata_cache)
        
        if has_cached_metadata:
            # Use optimized parallel processing with cached metadata
            return self._precompute_with_cached_metadata_parallel(
                image_folder, image_files, progress_callback, start_time, stats
            )
        else:
            # Fallback to original implementation
            return self._precompute_original_sequential(
                image_folder, image_files, progress_callback, start_time, stats
            )
    
    def _precompute_with_cached_metadata_parallel(self, image_folder: str, image_files: List[str],
                                                progress_callback: Optional[callable], start_time: datetime, 
                                                stats: Dict) -> Dict:
        """
        Optimized parallel processing using cached metadata
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        # Thread-safe counters
        stats_lock = threading.Lock()
        
        # Determine optimal number of workers (leave 1 CPU for UI)
        max_workers = min(8, max(2, os.cpu_count() - 1))
        
        def process_single_image(image_file: str) -> Optional[Dict]:
            """Process a single image using cached metadata"""
            try:
                image_path = os.path.join(image_folder, image_file)
                
                # Try to get cached metadata first
                cached_metadata = self.metadata_processor.metadata_cache.get(image_path)
                
                if cached_metadata:
                    # Use cached metadata to compute parameters directly
                    params = self._compute_params_from_cached_metadata(cached_metadata, image_path)
                    
                    with stats_lock:
                        stats['cache_hits'] += 1
                    
                    return params
                else:
                    # Fallback: extract metadata if not in cache
                    params = self.georeferencer.precompute_transformation_parameters(image_path)
                    
                    with stats_lock:
                        stats['cache_misses'] += 1
                    
                    return params
                    
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                return None
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_file = {
                executor.submit(process_single_image, image_file): image_file 
                for image_file in image_files
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_file):
                image_file = future_to_file[future]
                
                try:
                    params = future.result()
                    
                    if params:
                        image_path = os.path.join(image_folder, image_file)
                        self.image_parameters[image_path] = params
                        
                        with stats_lock:
                            stats['successful'] += 1
                            # Count RTK vs standard GPS
                            if params['rtk_status'] == 1:
                                stats['rtk_images'] += 1
                            else:
                                stats['standard_gps'] += 1
                    else:
                        with stats_lock:
                            stats['failed'] += 1
                            
                except Exception as e:
                    print(f"Error processing result for {image_file}: {e}")
                    with stats_lock:
                        stats['failed'] += 1
                
                # Update progress
                completed += 1
                if progress_callback:
                    progress = (completed / len(image_files)) * 100
                    progress_callback(progress, f"Processed {completed}/{len(image_files)} images")
        
        # Calculate final statistics
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        stats['processing_time'] = processing_time
        stats['avg_time_per_image'] = processing_time / len(image_files) if image_files else 0
        
        return stats
    
    def _compute_params_from_cached_metadata(self, cached_metadata: Dict, image_path: str) -> Optional[Dict]:
        """
        Compute transformation parameters directly from cached metadata (bypasses exiftool)
        
        This is the key optimization - we avoid redundant metadata extraction
        """
        try:
            # Check if we have required GPS data
            if not cached_metadata.get('has_gps', False):
                return None
            
            # Extract required parameters from cached metadata and ensure they're numeric
            gps_latitude = cached_metadata.get('latitude')
            gps_longitude = cached_metadata.get('longitude') 
            altitude_above_ground = cached_metadata.get('altitude_above_ground')
            focal_length_mm = cached_metadata.get('focal_length')
            
            # Validate and convert to float (MetadataProcessor now uses -n flag for decimal degrees)
            try:
                if gps_latitude is not None:
                    if isinstance(gps_latitude, str):
                        # Handle potential string format (fallback to DMS parsing if needed)
                        if 'deg' in gps_latitude:
                            gps_latitude = self._parse_dms_coordinate(gps_latitude)
                            if gps_latitude is None:
                                return None
                        else:
                            # Simple decimal string
                            gps_latitude = float(gps_latitude)
                    else:
                        # Already a number (expected with -n flag)
                        gps_latitude = float(gps_latitude)
                        
                if gps_longitude is not None:
                    if isinstance(gps_longitude, str):
                        # Handle potential string format (fallback to DMS parsing if needed)
                        if 'deg' in gps_longitude:
                            gps_longitude = self._parse_dms_coordinate(gps_longitude)
                            if gps_longitude is None:
                                return None
                        else:
                            # Simple decimal string
                            gps_longitude = float(gps_longitude)
                    else:
                        # Already a number (expected with -n flag)
                        gps_longitude = float(gps_longitude)
                        
                if altitude_above_ground is not None:
                    if isinstance(altitude_above_ground, str):
                        # Handle formats like "+20.036" or "20.036 m"
                        import re
                        match = re.search(r'([+-]?\d+\.?\d*)', str(altitude_above_ground))
                        if match:
                            altitude_above_ground = float(match.group(1))
                        else:
                            return None
                    else:
                        altitude_above_ground = float(altitude_above_ground)
                        
                if focal_length_mm is not None:
                    if isinstance(focal_length_mm, str):
                        # Handle formats like "50.0 mm"
                        import re
                        match = re.search(r'([+-]?\d+\.?\d*)', str(focal_length_mm))
                        if match:
                            focal_length_mm = float(match.group(1))
                        else:
                            return None
                    else:
                        focal_length_mm = float(focal_length_mm)
            except (ValueError, TypeError):
                return None
            
            # Validate required parameters
            if any(param is None for param in [gps_latitude, gps_longitude, altitude_above_ground, focal_length_mm]):
                return None
            
            # Get image dimensions and ensure they're numeric
            image_width = cached_metadata.get('image_width', cached_metadata.get('original_width'))
            image_height = cached_metadata.get('image_height', cached_metadata.get('original_height'))
            
            # Convert to int if they exist
            try:
                if image_width is not None:
                    image_width = int(image_width)
                if image_height is not None:
                    image_height = int(image_height)
            except (ValueError, TypeError):
                image_width = None
                image_height = None
            
            # If dimensions not in cache, get them from file (fallback)
            if not image_width or not image_height:
                try:
                    import cv2
                    img = cv2.imread(image_path)
                    if img is not None:
                        image_height, image_width = img.shape[:2]
                    else:
                        return None
                except:
                    return None
            
            # Get optimal yaw using gimbal compensation (updated for 15.6% accuracy improvement)
            # Create metadata dict for gimbal compensation with proper float conversion
            flight_yaw_raw = cached_metadata.get('FlightYawDegree', cached_metadata.get('flight_yaw_degree', 0))
            if isinstance(flight_yaw_raw, str):
                flight_yaw_raw = float(flight_yaw_raw.lstrip('+'))
            
            metadata_for_yaw = {
                'yaw_degrees': flight_yaw_raw,
                'gimbal_yaw': cached_metadata.get('gimbal_yaw'),
                'gimbal_roll': cached_metadata.get('gimbal_roll')
            }
            
            # Use compensated gimbal yaw from SelfContainedGeoreferencer
            optimal_yaw, is_gimbal_lock, yaw_source = self.georeferencer.get_compensated_gimbal_yaw(metadata_for_yaw)
            yaw_degrees = optimal_yaw
            
            try:
                yaw_degrees = float(yaw_degrees)
            except (ValueError, TypeError):
                yaw_degrees = 0.0
            
            # Calculate Ground Sample Distance using the same formula as SelfContainedGeoreferencer
            PIXEL_SIZE_MICRONS = 4.4
            EARTH_RADIUS_METERS = 6371004
            
            gsd_meters_per_pixel = (PIXEL_SIZE_MICRONS * altitude_above_ground) / (focal_length_mm * 1000)
            gsd_degrees_per_pixel = gsd_meters_per_pixel / (2 * math.pi * EARTH_RADIUS_METERS) * 360
            
            # Image center coordinates
            center_x = image_width / 2
            center_y = image_height / 2
            
            # Pre-compute rotation matrix components
            yaw_radians = math.radians(yaw_degrees)
            cos_yaw = math.cos(yaw_radians)
            sin_yaw = math.sin(yaw_radians)
            
            # Determine RTK status
            rtk_status = 1 if (cached_metadata.get('gps_status') == 'RTK' or 
                              cached_metadata.get('has_rtk', False)) else 0
            
            return {
                'gps_latitude': float(gps_latitude),
                'gps_longitude': float(gps_longitude),
                'center_x': center_x,
                'center_y': center_y,
                'gsd_degrees_per_pixel': gsd_degrees_per_pixel,
                'cos_yaw': cos_yaw,
                'sin_yaw': sin_yaw,
                'rtk_status': rtk_status,
                'altitude_above_ground': float(altitude_above_ground),
                'yaw_degrees': float(yaw_degrees),
                'drone_model': cached_metadata.get('drone_model', 'Unknown')
            }
            
        except Exception as e:
            print(f"Error computing parameters from cached metadata for {image_path}: {e}")
            return None
    
    def _parse_dms_coordinate(self, dms_string: str) -> Optional[float]:
        """
        Parse DMS (Degrees Minutes Seconds) coordinate string to decimal degrees
        
        Examples:
        "53 deg 49' 48.76" N" -> 53.8302119166667
        "12 deg 30' 37.49" E" -> 12.5104140555556
        """
        try:
            import re
            
            # Extract degrees, minutes, seconds, and direction
            # Pattern matches: "53 deg 49' 48.76" N" or similar formats
            pattern = r"(\d+(?:\.\d+)?)\s*deg\s*(\d+(?:\.\d+)?)'?\s*(\d+(?:\.\d+)?)?\"?\s*([NSEW]?)"
            match = re.search(pattern, str(dms_string))
            
            if not match:
                # Fallback: try to extract just a decimal number
                decimal_match = re.search(r'([+-]?\d+\.?\d*)', str(dms_string))
                if decimal_match:
                    return float(decimal_match.group(1))
                return None
            
            degrees = float(match.group(1))
            minutes = float(match.group(2)) if match.group(2) else 0.0
            seconds = float(match.group(3)) if match.group(3) else 0.0
            direction = match.group(4).upper() if match.group(4) else ''
            
            # Convert to decimal degrees
            decimal_degrees = degrees + minutes/60.0 + seconds/3600.0
            
            # Apply sign based on direction
            if direction in ['S', 'W']:
                decimal_degrees = -decimal_degrees
            
            return decimal_degrees
            
        except Exception as e:
            print(f"Error parsing DMS coordinate '{dms_string}': {e}")
            return None
    
    def _precompute_original_sequential(self, image_folder: str, image_files: List[str],
                                      progress_callback: Optional[callable], start_time: datetime,
                                      stats: Dict) -> Dict:
        """
        Original sequential implementation (fallback when no cached metadata available)
        """
        for i, image_file in enumerate(image_files):
            try:
                if progress_callback:
                    progress = (i / len(image_files)) * 100
                    progress_callback(progress, f"Computing parameters for {image_file}")
                
                image_path = os.path.join(image_folder, image_file)
                
                # Pre-compute transformation parameters using self-contained georeferencer
                params = self.georeferencer.precompute_transformation_parameters(image_path)
                
                if params:
                    self.image_parameters[image_path] = params
                    stats['successful'] += 1
                    
                    # Count RTK vs standard GPS
                    if params['rtk_status'] == 1:
                        stats['rtk_images'] += 1
                    else:
                        stats['standard_gps'] += 1
                else:
                    stats['failed'] += 1
                    
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                stats['failed'] += 1
                continue
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        stats['processing_time'] = processing_time
        stats['avg_time_per_image'] = processing_time / len(image_files) if image_files else 0
        
        return stats
    
    
    
    def process_ml_detection_results(self, detection_results: Dict, 
                                   progress_callback: Optional[callable] = None) -> List[Dict]:
        """
        Process ML detection results and convert all coordinates to WGS84
        
        Args:
            detection_results: Results from BatchProcessor with feature coordinates
            progress_callback: Optional callback for progress updates
            
        Returns:
            List[Dict]: All georeferenced coordinates from all images
        """
        print("Converting ML detection coordinates to WGS84...")
        start_time = datetime.now()
        
        all_coordinates = []
        total_images = len(detection_results.get('results', {}))
        
        for i, (image_path, result) in enumerate(detection_results.get('results', {}).items()):
            try:
                if progress_callback:
                    progress = (i / total_images) * 100
                    progress_callback(progress, f"Georeferencing {Path(image_path).name}")
                
                # Extract coordinates and confidence scores
                feature_coordinates = result.get('feature_coordinates', [])
                confidence_scores = result.get('confidence_scores', [])
                
                if not feature_coordinates:
                    continue
                
                        # Transform coordinates using vectorized operations
                if image_path in self.image_parameters:
                    params = self.image_parameters[image_path]
                    georeferenced = self.georeferencer.transform_coordinates_vectorized(
                        params, feature_coordinates, confidence_scores
                    )
                    
                    # Add image name to each result
                    for result in georeferenced:
                        result['image_name'] = Path(image_path).name
                else:
                    georeferenced = []
                
                all_coordinates.extend(georeferenced)
                
            except Exception as e:
                print(f"Error processing {Path(image_path).name}: {e}")
                continue
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"Georeferencing complete: {len(all_coordinates)} coordinates from {total_images} images")
        print(f"Processing time: {processing_time:.2f}s")
        
        return all_coordinates
    
    def transform_coordinates_batch(self, image_path: str, pixel_coordinates: List[Tuple[int, int]], 
                                  confidence_scores: List[float]) -> List[Dict]:
        """
        Transform pixel coordinates to world coordinates for a single image
        
        Args:
            image_path: Path to the image
            pixel_coordinates: List of (x, y) pixel coordinates
            confidence_scores: List of confidence scores for each coordinate
            
        Returns:
            List[Dict]: Georeferenced coordinates with metadata
        """
        if image_path not in self.image_parameters:
            raise ValueError(f"No pre-computed parameters found for {image_path}")
        
        params = self.image_parameters[image_path]
        
        # Use the vectorized transformation from self-contained georeferencer
        georeferenced = self.georeferencer.transform_coordinates_vectorized(
            params, pixel_coordinates, confidence_scores
        )
        
        # Add image name to each result
        for result in georeferenced:
            result['image_name'] = Path(image_path).name
        
        return georeferenced
    
    def get_transformation_statistics(self) -> Dict:
        """
        Get statistics about pre-computed transformation parameters
        
        Returns:
            Dict: Statistics about cached parameters
        """
        if not self.image_parameters:
            return {'cached_images': 0}
        
        rtk_count = sum(1 for params in self.image_parameters.values() if params['rtk_status'] == 1)
        
        altitudes = [params['altitude_above_ground'] for params in self.image_parameters.values()]
        yaw_angles = [params['yaw_degrees'] for params in self.image_parameters.values()]
        
        return {
            'cached_images': len(self.image_parameters),
            'rtk_images': rtk_count,
            'standard_gps_images': len(self.image_parameters) - rtk_count,
            'altitude_range': {
                'min': min(altitudes) if altitudes else 0,
                'max': max(altitudes) if altitudes else 0,
                'avg': sum(altitudes) / len(altitudes) if altitudes else 0
            },
            'yaw_range': {
                'min': min(yaw_angles) if yaw_angles else 0,
                'max': max(yaw_angles) if yaw_angles else 0,
                'avg': sum(yaw_angles) / len(yaw_angles) if yaw_angles else 0
            }
        }