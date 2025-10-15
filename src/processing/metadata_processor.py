#!/usr/bin/env python3
"""
Metadata processor for extracting GPS and RTK information from drone images
"""

import subprocess
import json
import os
from typing import Dict, List, Optional

class MetadataProcessor:
    def __init__(self):
        self.metadata_cache = {}
        
    def extract_metadata(self, image_path: str) -> Dict:
        """Extract metadata from an image using exiftool"""
        if image_path in self.metadata_cache:
            return self.metadata_cache[image_path]
            
        try:
            # Run exiftool to get JSON output with all relevant fields
            # CRITICAL: Use -n flag to get decimal degrees instead of DMS format for GPS coordinates
            # CRITICAL: Include specific flight orientation fields for accurate georeferencing
            result = subprocess.run([
                'exiftool', '-j', '-n', '-GPS*', '-RTK*', '-*GPS*', '-*RTK*', 
                '-Gps*', '-Rtk*', '-Drone*', '-Absolute*', '-Relative*',
                '-FocalLength*', '-Camera*', '-Model*', '-Make*', '-Image*',
                '-DateTime*', '-Create*', '-Gimbal*', '-Flight*',
                '-FlightYawDegree', '-FlightPitchDegree', '-FlightRollDegree',
                '-GimbalYawDegree', '-GimbalPitchDegree', '-GimbalRollDegree',
                image_path
            ], capture_output=True, text=True, check=True)
            
            metadata_list = json.loads(result.stdout)
            if metadata_list:
                metadata = metadata_list[0]
                
                # Process and standardize the metadata
                processed = self._process_metadata(metadata)
                self.metadata_cache[image_path] = processed
                return processed
                
        except Exception as e:
            print(f"Error extracting metadata from {image_path}: {e}")
            
        # Return empty metadata if extraction fails
        empty_metadata = {
            'has_gps': False,
            'has_rtk': False,
            'gps_status': 'Unknown',
            'rtk_flag': None,
            'latitude': None,
            'longitude': None,
            'altitude': None,
            'rtk_std_lat': None,
            'rtk_std_lon': None,
            'rtk_std_hgt': None,
            'rtk_diff_age': None,
            'drone_model': 'Unknown',
            'rtk_status': 'Standard'
        }
        self.metadata_cache[image_path] = empty_metadata
        return empty_metadata
        
    def _safe_float_conversion(self, value):
        """Safely convert string values with + prefix to float"""
        if isinstance(value, str):
            clean_value = value.lstrip('+')
            try:
                return float(clean_value)
            except ValueError:
                return None
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return None

    def _process_metadata(self, raw_metadata: Dict) -> Dict:
        """Process raw metadata into standardized format"""
        processed = {
            'has_gps': False,
            'has_rtk': False,
            'gps_status': 'Unknown',
            'rtk_flag': None,
            'latitude': None,
            'longitude': None,
            'altitude': None,
            'altitude_above_ground': None,
            'rtk_std_lat': None,
            'rtk_std_lon': None,
            'rtk_std_hgt': None,
            'rtk_diff_age': None,
            'drone_model': 'Unknown',
            'focal_length': None,
            'focal_length_35mm': None,
            'camera_model': None,
            'image_width': None,
            'image_height': None,
            'datetime_original': None,
            'gimbal_pitch': None,
            'gimbal_roll': None,
            'gimbal_yaw': None,
            'rtk_status': 'Standard'
        }
        
        # Extract GPS information
        if 'GPSLatitude' in raw_metadata and 'GPSLongitude' in raw_metadata:
            processed['has_gps'] = True
            processed['latitude'] = raw_metadata.get('GPSLatitude')
            processed['longitude'] = raw_metadata.get('GPSLongitude')
            
        if 'GPSAltitude' in raw_metadata:
            processed['altitude'] = raw_metadata.get('GPSAltitude')
            
        # Extract GPS status
        gps_status = raw_metadata.get('GPSStatus', raw_metadata.get('GpsStatus', 'Unknown'))
        processed['gps_status'] = gps_status
        
        # Check for RTK data
        rtk_flag = raw_metadata.get('RtkFlag', raw_metadata.get('Rtk Flag'))
        if rtk_flag is not None:
            processed['has_rtk'] = True
            processed['rtk_flag'] = rtk_flag
        
        # Set RTK status for consistency across modules
        if gps_status == 'RTK' or processed['has_rtk']:
            processed['rtk_status'] = 'RTK'
        else:
            processed['rtk_status'] = 'Standard'
            
        # RTK accuracy data
        processed['rtk_std_lat'] = raw_metadata.get('RtkStdLat', raw_metadata.get('Rtk Std Lat'))
        processed['rtk_std_lon'] = raw_metadata.get('RtkStdLon', raw_metadata.get('Rtk Std Lon'))
        processed['rtk_std_hgt'] = raw_metadata.get('RtkStdHgt', raw_metadata.get('Rtk Std Hgt'))
        processed['rtk_diff_age'] = raw_metadata.get('RtkDiffAge', raw_metadata.get('Rtk Diff Age'))
        
        # Drone information
        processed['drone_model'] = raw_metadata.get('DroneModel', raw_metadata.get('Drone Model', 'Unknown'))
        
        # Camera and lens information
        processed['focal_length'] = raw_metadata.get('FocalLength')
        processed['focal_length_35mm'] = raw_metadata.get('FocalLengthIn35mmFormat')
        processed['camera_model'] = raw_metadata.get('CameraModelName', raw_metadata.get('Model'))
        
        # Image dimensions
        processed['image_width'] = raw_metadata.get('ImageWidth', raw_metadata.get('ExifImageWidth'))
        processed['image_height'] = raw_metadata.get('ImageHeight', raw_metadata.get('ExifImageHeight'))
        
        # Timestamp
        processed['datetime_original'] = raw_metadata.get('DateTimeOriginal', raw_metadata.get('CreateDate'))
        
        # Altitude information
        if 'GPSAltitude' in raw_metadata:
            processed['altitude'] = raw_metadata.get('GPSAltitude')
        
        # Relative altitude (height above ground)
        processed['altitude_above_ground'] = raw_metadata.get('RelativeAltitude', raw_metadata.get('Relative Altitude'))
        
        # Gimbal information with safe float conversion
        processed['gimbal_pitch'] = self._safe_float_conversion(
            raw_metadata.get('GimbalPitchDegree', raw_metadata.get('Gimbal Pitch Degree'))
        )
        processed['gimbal_roll'] = self._safe_float_conversion(
            raw_metadata.get('GimbalRollDegree', raw_metadata.get('Gimbal Roll Degree'))
        )
        processed['gimbal_yaw'] = self._safe_float_conversion(
            raw_metadata.get('GimbalYawDegree', raw_metadata.get('Gimbal Yaw Degree'))
        )
        
        # Flight information (CRITICAL for accurate georeferencing)
        processed['FlightYawDegree'] = raw_metadata.get('FlightYawDegree', raw_metadata.get('Flight Yaw Degree'))
        processed['FlightPitchDegree'] = raw_metadata.get('FlightPitchDegree', raw_metadata.get('Flight Pitch Degree'))
        processed['FlightRollDegree'] = raw_metadata.get('FlightRollDegree', raw_metadata.get('Flight Roll Degree'))
        
        return processed
        
    def validate_images_metadata(self, image_folder: str, image_files: List[str]) -> Dict:
        """Validate metadata for a list of images"""
        validation_results = {
            'total_images': len(image_files),
            'gps_valid': 0,
            'rtk_count': 0,
            'rtk_quality_good': 0,
            'rtk_quality_fair': 0,
            'rtk_quality_poor': 0,
            'no_gps': 0,
            'drone_models': set(),
            'altitude_range': {'min': None, 'max': None},
            'rtk_accuracy_stats': {
                'lat_std_avg': 0,
                'lon_std_avg': 0,
                'hgt_std_avg': 0
            }
        }
        
        rtk_lat_stds = []
        rtk_lon_stds = []
        rtk_hgt_stds = []
        altitudes = []
        
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            metadata = self.extract_metadata(image_path)
            
            # Count GPS valid images
            if metadata['has_gps']:
                validation_results['gps_valid'] += 1
                
                # Collect altitude data
                if metadata['altitude'] is not None:
                    try:
                        alt_value = float(str(metadata['altitude']).split()[0])  # Remove units
                        altitudes.append(alt_value)
                    except (ValueError, IndexError):
                        pass
            else:
                validation_results['no_gps'] += 1
                
            # Count RTK images and quality
            if metadata['has_rtk']:
                validation_results['rtk_count'] += 1
                
                # Assess RTK quality based on standard deviations
                lat_std = metadata['rtk_std_lat']
                lon_std = metadata['rtk_std_lon']
                hgt_std = metadata['rtk_std_hgt']
                
                if lat_std is not None and lon_std is not None:
                    rtk_lat_stds.append(float(lat_std))
                    rtk_lon_stds.append(float(lon_std))
                    
                    # Quality assessment (typical RTK accuracy thresholds)
                    max_std = max(float(lat_std), float(lon_std))
                    if max_std < 0.02:  # < 2cm
                        validation_results['rtk_quality_good'] += 1
                    elif max_std < 0.05:  # < 5cm
                        validation_results['rtk_quality_fair'] += 1
                    else:
                        validation_results['rtk_quality_poor'] += 1
                        
                if hgt_std is not None:
                    rtk_hgt_stds.append(float(hgt_std))
                    
            # Collect drone models
            if metadata['drone_model'] != 'Unknown':
                validation_results['drone_models'].add(metadata['drone_model'])
                
        # Calculate statistics
        if altitudes:
            validation_results['altitude_range']['min'] = min(altitudes)
            validation_results['altitude_range']['max'] = max(altitudes)
            
        if rtk_lat_stds:
            validation_results['rtk_accuracy_stats']['lat_std_avg'] = sum(rtk_lat_stds) / len(rtk_lat_stds)
        if rtk_lon_stds:
            validation_results['rtk_accuracy_stats']['lon_std_avg'] = sum(rtk_lon_stds) / len(rtk_lon_stds)
        if rtk_hgt_stds:
            validation_results['rtk_accuracy_stats']['hgt_std_avg'] = sum(rtk_hgt_stds) / len(rtk_hgt_stds)
            
        return validation_results
        
    def format_validation_summary(self, validation_results: Dict) -> str:
        """Format validation results into a readable summary"""
        total = validation_results['total_images']
        gps_valid = validation_results['gps_valid']
        rtk_count = validation_results['rtk_count']
        
        summary = f"""Metadata Validation Summary:

📊 Image Statistics:
   Total Images: {total}
   GPS Valid: {gps_valid} ({gps_valid/total*100:.1f}%)
   No GPS: {validation_results['no_gps']}

🛰️ RTK GPS Quality:
   RTK Images: {rtk_count} ({rtk_count/total*100:.1f}%)
   High Quality (< 2cm): {validation_results['rtk_quality_good']}
   Fair Quality (2-5cm): {validation_results['rtk_quality_fair']}
   Poor Quality (> 5cm): {validation_results['rtk_quality_poor']}

📐 RTK Accuracy (Average):
   Latitude Std: {validation_results['rtk_accuracy_stats']['lat_std_avg']:.4f}m
   Longitude Std: {validation_results['rtk_accuracy_stats']['lon_std_avg']:.4f}m
   Height Std: {validation_results['rtk_accuracy_stats']['hgt_std_avg']:.4f}m

✈️ Flight Information:
   Drone Models: {', '.join(validation_results['drone_models']) if validation_results['drone_models'] else 'Unknown'}"""

        if validation_results['altitude_range']['min'] is not None:
            alt_min = validation_results['altitude_range']['min']
            alt_max = validation_results['altitude_range']['max']
            summary += f"\n   Altitude Range: {alt_min:.1f}m - {alt_max:.1f}m"
            
        return summary