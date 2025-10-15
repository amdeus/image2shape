#!/usr/bin/env python3
"""
Image Footprint Calculator for Image2Shape

This module calculates image footprint polygons from corner coordinates
for the combined export option. It transforms image corner pixels to
world coordinates to create footprint polygons.

Key Features:
- Image corner coordinate calculation
- Footprint polygon generation using existing georeferencing
- Area calculation in square meters
- Integration with existing coordinate transformation pipeline

Author: Image2Shape Development Team
Version: 2.0 - Enhanced Export Options
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from shapely.geometry import Polygon
from pathlib import Path


class FootprintCalculator:
    """
    Calculate image footprint polygons from corner coordinates
    
    This class uses the existing georeferencing pipeline to transform
    image corner coordinates to world coordinates and create footprint polygons.
    """
    
    def __init__(self, georeferencer):
        """
        Initialize footprint calculator
        
        Args:
            georeferencer: Instance of SelfContainedGeoreferencer or BatchGeoreferencer
        """
        self.georeferencer = georeferencer
    
    def calculate_image_footprint(self, image_path: str, 
                                 georeferencing_params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Calculate image footprint polygon from corner coordinates
        
        Args:
            image_path: Path to image file
            georeferencing_params: Optional pre-computed georeferencing parameters
            
        Returns:
            Dict: Footprint data with polygon geometry and metadata
        """
        try:
            # Get image metadata and dimensions
            if georeferencing_params:
                # Use pre-computed parameters
                image_width = int(georeferencing_params['center_x'] * 2)
                image_height = int(georeferencing_params['center_y'] * 2)
                params = georeferencing_params
            else:
                # Extract metadata directly
                metadata = self.georeferencer.extract_complete_metadata(image_path)
                if not metadata['has_gps']:
                    return None
                
                image_width = metadata['image_width_px']
                image_height = metadata['image_height_px']
                
                # Compute transformation parameters
                params = self.georeferencer.precompute_transformation_parameters(image_path)
                if not params:
                    return None
            
            # Define corner coordinates (clockwise from top-left)
            corners = [
                (0, 0),                    # Top-left
                (image_width, 0),          # Top-right  
                (image_width, image_height), # Bottom-right
                (0, image_height)          # Bottom-left
            ]
            
            # Transform corner coordinates to world coordinates
            if hasattr(self.georeferencer, 'transform_coordinates_vectorized'):
                # Use batch georeferencer
                world_corners = self.georeferencer.transform_coordinates_vectorized(
                    params, corners, [1.0] * len(corners)  # Dummy confidence scores
                )
            else:
                # Use self-contained georeferencer
                world_corners = []
                for corner in corners:
                    world_coord = self.georeferencer.pixel_to_world_coordinates(
                        image_path, corner[0], corner[1]
                    )
                    if world_coord:
                        world_corners.append(world_coord)
                    else:
                        return None
            
            if len(world_corners) != 4:
                return None
            
            # Create polygon from world coordinates
            polygon_coords = [
                (coord['world_longitude'], coord['world_latitude']) 
                for coord in world_corners
            ]
            
            # Ensure polygon is closed
            if polygon_coords[0] != polygon_coords[-1]:
                polygon_coords.append(polygon_coords[0])
            
            footprint_polygon = Polygon(polygon_coords)
            
            # Calculate area in square meters
            area_m2 = self._calculate_footprint_area_m2(footprint_polygon)
            
            # Get representative metadata from first corner
            base_coord = world_corners[0]
            
            return {
                'geometry': footprint_polygon,
                'image_name': Path(image_path).name,
                'area_m2': area_m2,
                'altitude_m': base_coord.get('altitude_above_ground_m', 0),
                'yaw_degrees': base_coord.get('yaw_degrees', 0),
                'rtk_status': base_coord.get('rtk_status', 0),
                'drone_longitude': base_coord.get('drone_gps_longitude', 0),
                'drone_latitude': base_coord.get('drone_gps_latitude', 0),
                'corner_coordinates': polygon_coords[:-1]  # Exclude duplicate last point
            }
            
        except Exception as e:
            print(f"Error calculating footprint for {image_path}: {e}")
            return None
    
    def calculate_batch_footprints(self, image_paths: List[str], 
                                  batch_georeferencer, 
                                  progress_callback: Optional[callable] = None) -> List[Dict]:
        """
        Calculate footprints for multiple images using batch georeferencer
        
        Args:
            image_paths: List of image file paths
            batch_georeferencer: BatchGeoreferencer instance with pre-computed parameters
            progress_callback: Optional progress callback
            
        Returns:
            List[Dict]: List of footprint data dictionaries
        """
        footprints = []
        
        for i, image_path in enumerate(image_paths):
            if progress_callback:
                progress = (i / len(image_paths)) * 100
                progress_callback(progress, f"Calculating footprint for {Path(image_path).name}")
            
            # Get pre-computed parameters
            if image_path in batch_georeferencer.image_parameters:
                params = batch_georeferencer.image_parameters[image_path]
                footprint = self.calculate_image_footprint(image_path, params)
                
                if footprint:
                    footprints.append(footprint)
            else:
                print(f"No georeferencing parameters found for {image_path}")
        
        return footprints
    
    def _calculate_footprint_area_m2(self, polygon: Polygon) -> float:
        """
        Calculate polygon area in square meters using approximate method
        
        Args:
            polygon: Shapely polygon geometry
            
        Returns:
            float: Area in square meters
        """
        # Get polygon bounds
        minx, miny, maxx, maxy = polygon.bounds
        
        # Average latitude for area calculation
        avg_lat = (miny + maxy) / 2
        
        # Degrees to meters conversion factors
        lat_m_per_deg = 111000  # Approximately 111 km per degree latitude
        lon_m_per_deg = 111000 * math.cos(math.radians(avg_lat))
        
        # Convert polygon coordinates to meters (approximate)
        coords_m = []
        for x, y in polygon.exterior.coords:
            x_m = (x - minx) * lon_m_per_deg
            y_m = (y - miny) * lat_m_per_deg
            coords_m.append((x_m, y_m))
        
        # Calculate area using shoelace formula
        n = len(coords_m)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += coords_m[i][0] * coords_m[j][1]
            area -= coords_m[j][0] * coords_m[i][1]
        
        return abs(area) / 2.0
    
    def get_drone_positions_from_images(self, image_paths: List[str], 
                                       batch_georeferencer) -> List[Dict]:
        """
        Extract drone GPS positions from image metadata
        
        Args:
            image_paths: List of image file paths
            batch_georeferencer: BatchGeoreferencer instance
            
        Returns:
            List[Dict]: List of drone position data
        """
        drone_positions = []
        
        for image_path in image_paths:
            if image_path in batch_georeferencer.image_parameters:
                params = batch_georeferencer.image_parameters[image_path]
                
                drone_position = {
                    'image_name': Path(image_path).name,
                    'longitude': params['gps_longitude'],
                    'latitude': params['gps_latitude'],
                    'altitude_m': params.get('altitude_above_ground', 0),
                    'rtk_status': params.get('rtk_status', 0),
                    'yaw_degrees': params.get('yaw_degrees', 0)
                }
                
                drone_positions.append(drone_position)
        
        return drone_positions