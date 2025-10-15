#!/usr/bin/env python3
"""
Self-Contained Georeferencer for Image2Shape v2.0

This module implements a complete, self-contained georeferencing solution
that doesn't rely on external dr.py imports. It extracts all necessary
metadata using exiftool and implements the photogrammetric transformation
directly.

Key Features:
- No external module dependencies (except standard libraries)
- Direct EXIF metadata extraction using exiftool
- Complete photogrammetric transformation implementation
- Optimized for batch processing with pre-computed parameters
- DJI drone compatibility with RTK support

Author: Image2Shape Development Team
Version: 2.0 - Self-Contained Implementation
"""

import subprocess
import json
import math
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class SelfContainedGeoreferencer:
    """
    Complete georeferencing solution without external dependencies
    
    This class implements the full photogrammetric transformation pipeline
    using only standard libraries and exiftool for metadata extraction.
    """
    
    # DJI P1 sensor specifications
    PIXEL_SIZE_MICRONS = 4.4        # Physical pixel size on sensor
    EARTH_RADIUS_METERS = 6371004   # Earth radius for degree conversion
    
    def __init__(self):
        """Initialize the self-contained georeferencer"""
        self.metadata_cache = {}
        
    def extract_complete_metadata(self, image_path: str) -> Dict:
        """
        Extract all required metadata using exiftool directly
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dict: Complete metadata for georeferencing
        """
        if image_path in self.metadata_cache:
            return self.metadata_cache[image_path]
        
        try:
            # Use exiftool to extract all relevant metadata
            result = subprocess.run([
                'exiftool', '-j', '-n',
                '-GPSLatitude', '-GPSLongitude', '-GPSAltitude',
                '-RelativeAltitude', '-AbsoluteAltitude',
                '-GimbalYawDegree', '-GimbalPitchDegree', '-GimbalRollDegree',
                '-FlightYawDegree', '-FlightPitchDegree', '-FlightRollDegree',
                '-FocalLength', '-FocalLengthIn35mmFormat',
                '-ImageWidth', '-ImageHeight', '-ExifImageWidth', '-ExifImageHeight',
                '-Make', '-Model', '-CameraModelName',
                '-GpsStatus', '-RtkFlag', '-RtkStdLat', '-RtkStdLon', '-RtkStdHgt',
                '-DateTimeOriginal', '-CreateDate',
                str(image_path)
            ], capture_output=True, text=True, check=True)
            
            metadata_list = json.loads(result.stdout)
            if not metadata_list:
                raise ValueError("No metadata found")
            
            raw_metadata = metadata_list[0]
            
            # Process metadata into standardized format
            processed = self._process_raw_metadata(raw_metadata, image_path)
            
            self.metadata_cache[image_path] = processed
            return processed
            
        except Exception as e:
            print(f"Error extracting metadata from {image_path}: {e}")
            return self._get_empty_metadata()
    
    def _process_raw_metadata(self, raw_metadata: Dict, image_path: str) -> Dict:
        """
        Process raw exiftool metadata into standardized format
        
        Args:
            raw_metadata: Raw metadata from exiftool
            image_path: Path to image file
            
        Returns:
            Dict: Processed metadata
        """
        # Get image dimensions from file if not in metadata
        try:
            img = cv2.imread(str(image_path))
            if img is not None:
                height_px, width_px = img.shape[:2]
            else:
                width_px = raw_metadata.get('ImageWidth', raw_metadata.get('ExifImageWidth', 0))
                height_px = raw_metadata.get('ImageHeight', raw_metadata.get('ExifImageHeight', 0))
        except:
            width_px = raw_metadata.get('ImageWidth', raw_metadata.get('ExifImageWidth', 0))
            height_px = raw_metadata.get('ImageHeight', raw_metadata.get('ExifImageHeight', 0))
        
        # Extract GPS coordinates
        gps_lat = raw_metadata.get('GPSLatitude')
        gps_lon = raw_metadata.get('GPSLongitude')
        has_gps = gps_lat is not None and gps_lon is not None
        
        # Extract altitude (prefer relative altitude for drone work)
        altitude = (raw_metadata.get('RelativeAltitude') or 
                   raw_metadata.get('AbsoluteAltitude') or 
                   raw_metadata.get('GPSAltitude'))
        
        # Extract flight orientation
        yaw_degrees = (raw_metadata.get('FlightYawDegree') or 0)
        pitch_degrees = (raw_metadata.get('FlightPitchDegree') or 0)
        roll_degrees = (raw_metadata.get('FlightRollDegree') or 0)
        
        # Extract gimbal orientation for advanced georeferencing
        gimbal_yaw = raw_metadata.get('GimbalYawDegree')
        gimbal_pitch = raw_metadata.get('GimbalPitchDegree')
        gimbal_roll = raw_metadata.get('GimbalRollDegree')
        
        # Safe float conversion for gimbal data
        if isinstance(gimbal_yaw, str):
            gimbal_yaw = float(gimbal_yaw.lstrip('+')) if gimbal_yaw else None
        if isinstance(gimbal_pitch, str):
            gimbal_pitch = float(gimbal_pitch.lstrip('+')) if gimbal_pitch else None
        if isinstance(gimbal_roll, str):
            gimbal_roll = float(gimbal_roll.lstrip('+')) if gimbal_roll else None
        
        # Extract focal length
        focal_length_mm = (raw_metadata.get('FocalLength') or 
                          raw_metadata.get('FocalLengthIn35mmFormat'))
        
        # Extract RTK information
        gps_status = raw_metadata.get('GpsStatus', 'Unknown')
        rtk_flag = raw_metadata.get('RtkFlag')
        has_rtk = rtk_flag is not None or gps_status == 'RTK'
        
        # Extract drone information
        make = raw_metadata.get('Make', '')
        model = raw_metadata.get('Model', raw_metadata.get('CameraModelName', ''))
        drone_model = f"{make} {model}".strip()
        
        return {
            'has_gps': has_gps,
            'gps_latitude': float(gps_lat) if gps_lat is not None else None,
            'gps_longitude': float(gps_lon) if gps_lon is not None else None,
            'altitude_above_ground_m': float(altitude) if altitude is not None else None,
            'yaw_degrees': float(yaw_degrees),
            'pitch_degrees': float(pitch_degrees),
            'roll_degrees': float(roll_degrees),
            'gimbal_yaw': gimbal_yaw,
            'gimbal_pitch': gimbal_pitch,
            'gimbal_roll': gimbal_roll,
            'focal_length_mm': float(focal_length_mm) if focal_length_mm is not None else None,
            'image_width_px': int(width_px),
            'image_height_px': int(height_px),
            'gps_status': gps_status,
            'has_rtk': has_rtk,
            'rtk_flag': rtk_flag,
            'drone_model': drone_model,
            'rtk_std_lat': raw_metadata.get('RtkStdLat'),
            'rtk_std_lon': raw_metadata.get('RtkStdLon'),
            'rtk_std_hgt': raw_metadata.get('RtkStdHgt'),
            'datetime_original': raw_metadata.get('DateTimeOriginal', raw_metadata.get('CreateDate'))
        }
    
    def _get_empty_metadata(self) -> Dict:
        """Return empty metadata structure"""
        return {
            'has_gps': False,
            'gps_latitude': None,
            'gps_longitude': None,
            'altitude_above_ground_m': None,
            'yaw_degrees': 0,
            'pitch_degrees': 0,
            'roll_degrees': 0,
            'focal_length_mm': None,
            'image_width_px': 0,
            'image_height_px': 0,
            'gps_status': 'Unknown',
            'has_rtk': False,
            'rtk_flag': None,
            'drone_model': 'Unknown'
        }
    
    def calculate_ground_sample_distance(self, altitude_m: float, focal_length_mm: float) -> Tuple[float, float]:
        """
        Calculate Ground Sample Distance (GSD)
        
        Args:
            altitude_m: Flight altitude above ground in meters
            focal_length_mm: Camera focal length in millimeters
            
        Returns:
            Tuple: (gsd_meters_per_pixel, gsd_degrees_per_pixel)
        """
        # Calculate GSD in meters per pixel
        gsd_meters_per_pixel = (self.PIXEL_SIZE_MICRONS * altitude_m) / (focal_length_mm * 1000)
        
        # Convert to degrees per pixel for WGS84
        gsd_degrees_per_pixel = gsd_meters_per_pixel / (2 * math.pi * self.EARTH_RADIUS_METERS) * 360
        
        return gsd_meters_per_pixel, gsd_degrees_per_pixel
    
    def apply_yaw_rotation(self, dx_pixels: float, dy_pixels: float, yaw_degrees: float) -> Tuple[float, float]:
        """
        Apply yaw rotation to correct for drone heading
        
        Args:
            dx_pixels: X offset from image center in pixels
            dy_pixels: Y offset from image center in pixels
            yaw_degrees: Drone yaw angle in degrees
            
        Returns:
            Tuple: (dx_rotated, dy_rotated)
        """
        # Convert to radians - use positive yaw for correct rotation direction
        yaw_radians = math.radians(yaw_degrees)
        
        # Apply 2D rotation
        cos_yaw = math.cos(yaw_radians)
        sin_yaw = math.sin(yaw_radians)
        
        dx_rotated = dx_pixels * cos_yaw - dy_pixels * sin_yaw
        dy_rotated = dx_pixels * sin_yaw + dy_pixels * cos_yaw
        
        return dx_rotated, dy_rotated

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-180, 180] range"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def _detect_gimbal_lock(self, gimbal_roll: float, tolerance: float = 10.0) -> bool:
        """
        Detect gimbal lock condition
        
        Args:
            gimbal_roll: Gimbal roll angle in degrees
            tolerance: Tolerance for gimbal lock detection in degrees
            
        Returns:
            True if gimbal lock detected, False otherwise
        """
        if gimbal_roll is None:
            return False
        return abs(abs(gimbal_roll) - 180) < tolerance

    def get_compensated_gimbal_yaw(self, metadata: Dict) -> Tuple[float, bool, str]:
        """
        Get optimal yaw angle for georeferencing with gimbal compensation
        
        Args:
            metadata: Image metadata dictionary
            
        Returns:
            Tuple of (yaw_angle, is_gimbal_lock, yaw_source)
        """
        # Extract yaw data
        flight_yaw = metadata.get('yaw_degrees', 0)
        gimbal_yaw = metadata.get('gimbal_yaw')
        gimbal_roll = metadata.get('gimbal_roll')
        
        # If no gimbal data available, use flight yaw
        if gimbal_yaw is None:
            return flight_yaw, False, "flight_yaw"
        
        # Check for gimbal lock
        is_gimbal_lock = self._detect_gimbal_lock(gimbal_roll)
        
        if is_gimbal_lock:
            # Apply +180° compensation for gimbal lock
            compensated_yaw = self._normalize_angle(gimbal_yaw + 180)
            return compensated_yaw, True, "compensated_gimbal_yaw"
        else:
            # Use gimbal yaw directly for normal positions
            return gimbal_yaw, False, "direct_gimbal_yaw"
    
    def pixel_to_world_coordinates(self, image_path: str, pixel_x: int, pixel_y: int) -> Optional[Dict]:
        """
        Convert pixel coordinates to WGS84 world coordinates
        
        Args:
            image_path: Path to image file
            pixel_x: X pixel coordinate
            pixel_y: Y pixel coordinate
            
        Returns:
            Dict: Georeferencing result or None if failed
        """
        # Extract metadata
        metadata = self.extract_complete_metadata(image_path)
        
        if not metadata['has_gps']:
            return None
        
        # Validate required parameters
        required_params = ['gps_latitude', 'gps_longitude', 'altitude_above_ground_m', 'focal_length_mm']
        if any(metadata[param] is None for param in required_params):
            return None
        
        # Calculate Ground Sample Distance
        gsd_meters, gsd_degrees = self.calculate_ground_sample_distance(
            metadata['altitude_above_ground_m'], 
            metadata['focal_length_mm']
        )
        
        # Calculate pixel offset from image center
        center_x = metadata['image_width_px'] / 2
        center_y = metadata['image_height_px'] / 2
        
        dx_pixels = pixel_x - center_x
        dy_pixels = pixel_y - center_y
        
        # Get optimal yaw angle with gimbal compensation
        optimal_yaw, is_gimbal_lock, yaw_source = self.get_compensated_gimbal_yaw(metadata)
        
        
        # Apply yaw rotation using optimal yaw
        dx_rotated, dy_rotated = self.apply_yaw_rotation(
            dx_pixels, dy_pixels, optimal_yaw
        )
        
        # Convert to geographic degrees with latitude-corrected longitude scaling
        # CRITICAL FIX: Account for longitude compression at this latitude
        lat_radians = math.radians(metadata['gps_latitude'])
        lon_scale_factor = math.cos(lat_radians)  # Longitude compression factor
        
        dx_degrees = dx_rotated * gsd_degrees / lon_scale_factor  # Corrected longitude scaling
        dy_degrees = dy_rotated * gsd_degrees                     # Latitude unchanged
        
        # Calculate final world coordinates
        world_longitude = metadata['gps_longitude'] + dx_degrees
        world_latitude = metadata['gps_latitude'] - dy_degrees  # Negative: image Y increases downward
        
        # Convert RTK status to binary flag
        rtk_status = 1 if (metadata['gps_status'] == 'RTK' or metadata['has_rtk']) else 0
        
        return {
            'image_name': Path(image_path).name,
            'pixel_x': pixel_x,
            'pixel_y': pixel_y,
            'world_longitude': world_longitude,
            'world_latitude': world_latitude,
            'drone_gps_longitude': metadata['gps_longitude'],
            'drone_gps_latitude': metadata['gps_latitude'],
            'altitude_above_ground_m': metadata['altitude_above_ground_m'],
            'rtk_status': rtk_status,
            'yaw_degrees': optimal_yaw,
            'yaw_source': yaw_source,
            'is_gimbal_lock': is_gimbal_lock,
            'gsd_meters_per_pixel': gsd_meters,
            'drone_model': metadata['drone_model']
        }
    
    def precompute_transformation_parameters(self, image_path: str) -> Optional[Dict]:
        """
        Pre-compute transformation parameters for batch processing
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dict: Pre-computed parameters or None if failed
        """
        metadata = self.extract_complete_metadata(image_path)
        
        if not metadata['has_gps']:
            return None
        
        # Validate required parameters
        required_params = ['gps_latitude', 'gps_longitude', 'altitude_above_ground_m', 'focal_length_mm']
        if any(metadata[param] is None for param in required_params):
            return None
        
        try:
            # Calculate Ground Sample Distance
            gsd_meters, gsd_degrees = self.calculate_ground_sample_distance(
                metadata['altitude_above_ground_m'], 
                metadata['focal_length_mm']
            )
            
            # Image center coordinates
            center_x = metadata['image_width_px'] / 2
            center_y = metadata['image_height_px'] / 2
            
            # Get optimal yaw using gimbal compensation for precomputed parameters
            optimal_yaw, is_gimbal_lock, yaw_source = self.get_compensated_gimbal_yaw(metadata)
            
            
            # Pre-compute rotation matrix components using optimal yaw
            yaw_radians = math.radians(optimal_yaw)
            cos_yaw = math.cos(yaw_radians)
            sin_yaw = math.sin(yaw_radians)
            
            return {
                'gps_latitude': metadata['gps_latitude'],
                'gps_longitude': metadata['gps_longitude'],
                'center_x': center_x,
                'center_y': center_y,
                'gsd_degrees_per_pixel': gsd_degrees,
                'cos_yaw': cos_yaw,
                'sin_yaw': sin_yaw,
                'rtk_status': 1 if (metadata['gps_status'] == 'RTK' or metadata['has_rtk']) else 0,
                'altitude_above_ground': metadata['altitude_above_ground_m'],
                'yaw_degrees': metadata['yaw_degrees'],
                'drone_model': metadata['drone_model']
            }
            
        except Exception as e:
            print(f"Error computing transformation parameters for {image_path}: {e}")
            return None
    
    def transform_coordinates_vectorized(self, params: Dict, pixel_coordinates: List[Tuple[int, int]], 
                                       confidence_scores: Optional[List[float]] = None) -> List[Dict]:
        """
        Transform multiple coordinates using pre-computed parameters (vectorized)
        
        Args:
            params: Pre-computed transformation parameters
            pixel_coordinates: List of (x, y) pixel coordinates
            confidence_scores: Optional confidence scores
            
        Returns:
            List[Dict]: Georeferenced coordinates
        """
        if not pixel_coordinates:
            return []
        
        # Convert to numpy arrays for vectorized operations
        coords_array = np.array(pixel_coordinates)
        pixel_x = coords_array[:, 0]
        pixel_y = coords_array[:, 1]
        
        # Calculate pixel offsets from image center (vectorized)
        dx_pixels = pixel_x - params['center_x']
        dy_pixels = pixel_y - params['center_y']
        
        # Apply yaw rotation (vectorized)
        dx_rotated = dx_pixels * params['cos_yaw'] - dy_pixels * params['sin_yaw']
        dy_rotated = dx_pixels * params['sin_yaw'] + dy_pixels * params['cos_yaw']
        
        # Convert to geographic degrees with latitude-corrected longitude scaling (vectorized)
        # CRITICAL FIX: Account for longitude compression at this latitude
        lat_radians = math.radians(params['gps_latitude'])
        lon_scale_factor = math.cos(lat_radians)  # Longitude compression factor
        
        dx_degrees = dx_rotated * params['gsd_degrees_per_pixel'] / lon_scale_factor  # Corrected longitude scaling
        dy_degrees = dy_rotated * params['gsd_degrees_per_pixel']                     # Latitude unchanged
        
        # Calculate final world coordinates (vectorized)
        world_longitude = params['gps_longitude'] + dx_degrees
        world_latitude = params['gps_latitude'] - dy_degrees
        
        # Build result list
        results = []
        for i in range(len(pixel_coordinates)):
            result = {
                'pixel_x': int(pixel_x[i]),
                'pixel_y': int(pixel_y[i]),
                'world_longitude': float(world_longitude[i]),
                'world_latitude': float(world_latitude[i]),
                'drone_gps_longitude': params['gps_longitude'],
                'drone_gps_latitude': params['gps_latitude'],
                'altitude_above_ground_m': params['altitude_above_ground'],
                'rtk_status': params['rtk_status'],
                'yaw_degrees': params['yaw_degrees']
            }
            
            # Add confidence score if provided
            if confidence_scores and i < len(confidence_scores):
                # Handle numpy arrays and ensure scalar conversion
                conf_score = confidence_scores[i]
                if hasattr(conf_score, 'item'):  # numpy scalar
                    result['confidence_score'] = float(conf_score.item())
                elif hasattr(conf_score, '__len__') and len(conf_score) == 1:  # single-element array
                    result['confidence_score'] = float(conf_score[0])
                else:
                    result['confidence_score'] = float(conf_score)
            
            results.append(result)
        
        return results