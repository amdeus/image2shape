#!/usr/bin/env python3
"""
Polygon Processor for Image2Shape Export

This module handles polygon buffering and merging operations for the
feature polygon export option. It converts point features to buffered
polygons and merges overlapping ones.

Key Features:
- Point to polygon buffering with configurable radius
- Automatic merging of overlapping polygons
- Coordinate system aware buffering (meters to degrees conversion)
- Efficient spatial operations using Shapely

Author: Image2Shape Development Team
Version: 2.0 - Enhanced Export Options
"""

import math
import numpy as np
from typing import List, Dict, Tuple
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import geopandas as gpd


class PolygonProcessor:
    """
    Process point features into buffered and merged polygons
    
    This class handles the conversion of point features to polygon features
    by buffering each point and merging overlapping polygons.
    """
    
    # Earth radius for degree conversion
    EARTH_RADIUS_METERS = 6371004
    
    def __init__(self):
        """Initialize polygon processor"""
        pass
    
    def create_buffered_polygons(self, georeferenced_coords: List[Dict], 
                                buffer_size_m: float) -> List[Dict]:
        """
        Convert point coordinates to buffered polygons with merging
        
        Args:
            georeferenced_coords: List of georeferenced detection results
            buffer_size_m: Buffer size in meters
            
        Returns:
            List[Dict]: Polygon features with merged overlapping areas
        """
        if not georeferenced_coords:
            return []
        
        print(f"Creating buffered polygons with {buffer_size_m}m buffer...")
        
        # Convert points to buffered polygons
        buffered_polygons = []
        polygon_data = []
        
        for i, coord in enumerate(georeferenced_coords):
            # Create point geometry
            point = Point(coord['world_longitude'], coord['world_latitude'])
            
            # Calculate buffer size in degrees at this latitude
            buffer_degrees = self._meters_to_degrees(
                buffer_size_m, coord['world_latitude']
            )
            
            # Create buffered polygon
            buffered_polygon = point.buffer(buffer_degrees)
            buffered_polygons.append(buffered_polygon)
            
            # Store associated data
            polygon_data.append({
                'original_coord': coord,
                'polygon_index': i
            })
        
        # Merge overlapping polygons
        print(f"Merging {len(buffered_polygons)} buffered polygons...")
        merged_polygons = self._merge_overlapping_polygons(
            buffered_polygons, polygon_data
        )
        
        print(f"Created {len(merged_polygons)} merged polygon features")
        return merged_polygons
    
    def _meters_to_degrees(self, meters: float, latitude: float) -> float:
        """
        Convert meters to degrees at given latitude
        
        Args:
            meters: Distance in meters
            latitude: Latitude in degrees
            
        Returns:
            float: Distance in degrees
        """
        # Convert meters to degrees
        degrees_per_meter = 360 / (2 * math.pi * self.EARTH_RADIUS_METERS)
        
        # Adjust for latitude compression (longitude only)
        lat_radians = math.radians(latitude)
        lon_compression = math.cos(lat_radians)
        
        # Use average of lat/lon degree conversion for circular buffer
        lat_degrees = meters * degrees_per_meter
        lon_degrees = meters * degrees_per_meter / lon_compression
        
        # Use average for circular buffer approximation
        return (lat_degrees + lon_degrees) / 2
    
    def _merge_overlapping_polygons(self, polygons: List[Polygon], 
                                  polygon_data: List[Dict]) -> List[Dict]:
        """
        Merge overlapping polygons and combine their attributes
        
        Args:
            polygons: List of Shapely polygon geometries
            polygon_data: List of associated data for each polygon
            
        Returns:
            List[Dict]: Merged polygon features with combined attributes
        """
        if not polygons:
            return []
        
        # Create GeoDataFrame for efficient spatial operations
        gdf = gpd.GeoDataFrame(polygon_data, geometry=polygons)
        
        # Find overlapping groups using spatial index
        merged_groups = self._find_overlapping_groups(gdf)
        
        # Create merged polygon features
        merged_features = []
        
        for group_indices in merged_groups:
            # Get polygons and data for this group
            group_polygons = [polygons[i] for i in group_indices]
            group_data = [polygon_data[i] for i in group_indices]
            
            # Merge polygons using unary_union
            merged_geometry = unary_union(group_polygons)
            
            # Handle both single polygons and multipolygons
            if hasattr(merged_geometry, 'geoms'):
                # MultiPolygon - create separate features for each part
                for geom in merged_geometry.geoms:
                    if isinstance(geom, Polygon):
                        merged_feature = self._create_merged_feature(
                            geom, group_data, len(group_indices)
                        )
                        merged_features.append(merged_feature)
            else:
                # Single Polygon
                merged_feature = self._create_merged_feature(
                    merged_geometry, group_data, len(group_indices)
                )
                merged_features.append(merged_feature)
        
        return merged_features
    
    def _find_overlapping_groups(self, gdf: gpd.GeoDataFrame) -> List[List[int]]:
        """
        Find groups of overlapping polygons using spatial indexing
        
        Args:
            gdf: GeoDataFrame with polygon geometries
            
        Returns:
            List[List[int]]: Groups of overlapping polygon indices
        """
        n_polygons = len(gdf)
        visited = set()
        groups = []
        
        for i in range(n_polygons):
            if i in visited:
                continue
            
            # Start new group
            current_group = [i]
            visited.add(i)
            to_check = [i]
            
            # Find all connected overlapping polygons
            while to_check:
                current_idx = to_check.pop(0)
                current_geom = gdf.iloc[current_idx].geometry
                
                # Check intersection with all unvisited polygons
                for j in range(n_polygons):
                    if j in visited:
                        continue
                    
                    other_geom = gdf.iloc[j].geometry
                    
                    # Check if polygons overlap (not just touch)
                    if current_geom.intersects(other_geom) and not current_geom.touches(other_geom):
                        current_group.append(j)
                        visited.add(j)
                        to_check.append(j)
            
            groups.append(current_group)
        
        return groups
    
    def _create_merged_feature(self, geometry: Polygon, group_data: List[Dict], 
                             merge_count: int) -> Dict:
        """
        Create merged polygon feature with combined attributes
        
        Args:
            geometry: Merged polygon geometry
            group_data: List of original feature data
            merge_count: Number of polygons that were merged
            
        Returns:
            Dict: Merged polygon feature
        """
        # Calculate polygon area in square meters
        area_m2 = self._calculate_polygon_area_m2(geometry)
        
        # Combine attributes from all merged features
        confidence_scores = []
        image_names = set()
        rtk_statuses = []
        
        for data in group_data:
            coord = data['original_coord']
            if 'confidence_score' in coord:
                confidence_scores.append(coord['confidence_score'])
            if 'image_name' in coord:
                image_names.add(coord['image_name'])
            if 'rtk_status' in coord:
                rtk_statuses.append(coord['rtk_status'])
        
        # Calculate combined statistics
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        max_confidence = np.max(confidence_scores) if confidence_scores else 0.0
        
        # Convert RTK statuses to numeric values for calculation
        rtk_numeric = []
        for status in rtk_statuses:
            if isinstance(status, str):
                rtk_numeric.append(1 if status == 'RTK' else 0)
            else:
                rtk_numeric.append(int(status))
        rtk_percentage = (sum(rtk_numeric) / len(rtk_numeric) * 100) if rtk_numeric else 0.0
        
        # Get centroid for representative coordinates
        centroid = geometry.centroid
        
        # Use first feature's metadata as base
        base_coord = group_data[0]['original_coord']
        
        return {
            'geometry': geometry,
            'world_longitude': centroid.x,
            'world_latitude': centroid.y,
            'drone_gps_longitude': base_coord.get('drone_gps_longitude', centroid.x),
            'drone_gps_latitude': base_coord.get('drone_gps_latitude', centroid.y),
            'altitude_above_ground_m': base_coord.get('altitude_above_ground_m', 0),
            'rtk_status': 1 if rtk_percentage > 50 else 0,
            'yaw_degrees': base_coord.get('yaw_degrees', 0),
            'confidence_score': avg_confidence,
            'max_confidence': max_confidence,
            'feature_count': merge_count,
            'area_m2': area_m2,
            'image_names': ','.join(sorted(image_names)),
            'rtk_percentage': rtk_percentage
        }
    
    def _calculate_polygon_area_m2(self, polygon: Polygon) -> float:
        """
        Calculate polygon area in square meters
        
        Args:
            polygon: Shapely polygon geometry
            
        Returns:
            float: Area in square meters
        """
        # Get polygon bounds
        minx, miny, maxx, maxy = polygon.bounds
        
        # Calculate approximate area using UTM projection
        # This is an approximation - for precise area calculation,
        # we would need to reproject to an equal-area projection
        
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