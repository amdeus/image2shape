#!/usr/bin/env python3
"""
Optimized Polygon Processor for Image2Shape Export
High-performance polygon processing for large datasets (40k+ points)

This module provides optimized polygon buffering and merging operations
with spatial preprocessing, parallel processing, and memory efficiency.

Key Optimizations:
- Spatial preprocessing to remove duplicates and cluster dense areas
- Parallel polygon buffering using multiprocessing
- R-tree spatial indexing for O(n log n) merging instead of O(n²)
- Memory-efficient streaming processing
- Adaptive chunk sizing based on available memory

Performance Targets:
- 40k points: <1 minute total processing (vs 3-6 minutes current)
- Memory usage: <1GB peak (vs 2-4GB current)
- Scalability: Linear scaling to 100k+ points

Author: Image2Shape Development Team
Version: 2.1 - Performance Optimized
"""

import math
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import logging

from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import geopandas as gpd

# Optional dependencies with graceful fallback
try:
    from rtree import index
    RTREE_AVAILABLE = True
except ImportError:
    RTREE_AVAILABLE = False
    logging.warning("rtree not available - falling back to slower spatial operations")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - memory monitoring disabled")

from .polygon_processor import PolygonProcessor  # Import base class


class OptimizedPolygonProcessor(PolygonProcessor):
    """
    High-performance polygon processor for large datasets
    
    Extends the base PolygonProcessor with advanced optimizations:
    - Spatial preprocessing and deduplication
    - Parallel buffering operations
    - Spatial index-based merging
    - Memory-efficient processing
    """
    
    def __init__(self, enable_preprocessing=True, enable_parallel=True, 
                 n_workers=None, memory_limit_gb=2.0, chunk_size=2000):
        """
        Initialize optimized polygon processor
        
        Args:
            enable_preprocessing: Enable spatial preprocessing and deduplication
            enable_parallel: Enable parallel polygon buffering
            n_workers: Number of worker processes (None = auto-detect)
            memory_limit_gb: Memory limit in GB for adaptive processing
            chunk_size: Default chunk size for parallel processing
        """
        super().__init__()
        
        self.enable_preprocessing = enable_preprocessing
        self.enable_parallel = enable_parallel
        self.n_workers = n_workers or min(cpu_count(), 8)  # Cap at 8 workers
        self.memory_limit_gb = memory_limit_gb
        self.chunk_size = chunk_size
        
        # Performance tracking
        self.performance_stats = {
            'preprocessing_time': 0.0,
            'buffering_time': 0.0,
            'merging_time': 0.0,
            'total_time': 0.0,
            'points_processed': 0,
            'points_after_preprocessing': 0,
            'polygons_created': 0,
            'polygons_after_merging': 0,
            'memory_peak_mb': 0.0
        }
    
    def create_buffered_polygons(self, georeferenced_coords: List[Dict], 
                                buffer_size_m: float,
                                progress_callback: Optional[Callable] = None) -> List[Dict]:
        """
        Create buffered polygons with advanced optimizations
        
        Args:
            georeferenced_coords: List of georeferenced detection results
            buffer_size_m: Buffer size in meters
            progress_callback: Optional progress reporting callback
            
        Returns:
            List[Dict]: Optimized polygon features with merged overlapping areas
        """
        if not georeferenced_coords:
            return []
        
        start_time = time.time()
        self.performance_stats['points_processed'] = len(georeferenced_coords)
        
        try:
            # Phase 1: Spatial Preprocessing
            if self.enable_preprocessing:
                if progress_callback:
                    progress_callback(10, "Preprocessing coordinates...")
                
                preprocessed_coords = self._preprocess_coordinates(
                    georeferenced_coords, buffer_size_m
                )
                self.performance_stats['points_after_preprocessing'] = len(preprocessed_coords)
                
                reduction_pct = (1 - len(preprocessed_coords)/len(georeferenced_coords))*100
                if reduction_pct > 1:  # Only show if significant reduction
                    pass  # Removed verbose preprocessing output
            else:
                preprocessed_coords = georeferenced_coords
                self.performance_stats['points_after_preprocessing'] = len(preprocessed_coords)
            
            # Phase 2: Parallel Polygon Buffering
            if progress_callback:
                progress_callback(30, "Creating buffered polygons...")
            
            buffering_start = time.time()
            if self.enable_parallel and len(preprocessed_coords) > 1000:
                buffered_polygons, polygon_data = self._create_polygons_parallel(
                    preprocessed_coords, buffer_size_m, progress_callback
                )
            else:
                buffered_polygons, polygon_data = self._create_polygons_sequential(
                    preprocessed_coords, buffer_size_m
                )
            
            self.performance_stats['buffering_time'] = time.time() - buffering_start
            self.performance_stats['polygons_created'] = len(buffered_polygons)
            
            # Phase 3: Optimized Merging
            if progress_callback:
                progress_callback(70, "Merging overlapping polygons...")
            
            merging_start = time.time()
            if RTREE_AVAILABLE and len(buffered_polygons) > 500:
                merged_polygons = self._merge_with_spatial_index(
                    buffered_polygons, polygon_data, progress_callback
                )
            else:
                merged_polygons = self._merge_overlapping_polygons(
                    buffered_polygons, polygon_data
                )
            
            self.performance_stats['merging_time'] = time.time() - merging_start
            self.performance_stats['polygons_after_merging'] = len(merged_polygons)
            
            # Final statistics
            self.performance_stats['total_time'] = time.time() - start_time
            self.performance_stats['memory_peak_mb'] = self._get_memory_usage_mb()
            
            if progress_callback:
                progress_callback(100, "Polygon processing complete!")
            
            return merged_polygons
            
        except Exception as e:
            print(f"Warning: Optimized processing failed ({e}), falling back to standard processing")
            # Fallback to parent class implementation
            return super().create_buffered_polygons(georeferenced_coords, buffer_size_m)
    
    def _preprocess_coordinates(self, coords: List[Dict], buffer_size_m: float) -> List[Dict]:
        """
        Preprocess coordinates to remove duplicates and optimize for buffering
        
        Args:
            coords: Original coordinates
            buffer_size_m: Buffer size for tolerance calculations
            
        Returns:
            List[Dict]: Preprocessed coordinates
        """
        preprocessing_start = time.time()
        
        # Step 1: Remove exact duplicates
        unique_coords = self._remove_exact_duplicates(coords)
        
        # Step 2: Remove near-duplicates within buffer tolerance
        tolerance_m = buffer_size_m * 0.5  # Half buffer size as tolerance
        deduplicated_coords = self._spatial_deduplication(unique_coords, tolerance_m)
        
        # Step 3: Cluster very dense areas (optional aggressive optimization)
        if len(deduplicated_coords) > 20000:  # Only for very large datasets
            cluster_radius_m = buffer_size_m * 0.8
            clustered_coords = self._cluster_dense_areas(deduplicated_coords, cluster_radius_m)
        else:
            clustered_coords = deduplicated_coords
        
        self.performance_stats['preprocessing_time'] = time.time() - preprocessing_start
        return clustered_coords
    
    def _remove_exact_duplicates(self, coords: List[Dict]) -> List[Dict]:
        """Remove coordinates with identical lat/lon values"""
        seen = set()
        unique_coords = []
        
        for coord in coords:
            # Round to 8 decimal places to catch near-identical coordinates
            lat = round(coord['world_latitude'], 8)
            lon = round(coord['world_longitude'], 8)
            key = (lat, lon)
            if key not in seen:
                seen.add(key)
                unique_coords.append(coord)
        
        return unique_coords
    
    def _spatial_deduplication(self, coords: List[Dict], tolerance_m: float) -> List[Dict]:
        """
        Remove coordinates within tolerance distance using spatial hashing
        
        Args:
            coords: Input coordinates
            tolerance_m: Tolerance distance in meters
            
        Returns:
            List[Dict]: Deduplicated coordinates
        """
        if not coords:
            return coords
        
        # Convert tolerance to degrees (approximate)
        avg_lat = np.mean([c['world_latitude'] for c in coords])
        tolerance_deg = tolerance_m / 111000  # Rough conversion
        
        # Create spatial hash grid
        grid_size = tolerance_deg
        spatial_hash = defaultdict(list)
        
        for i, coord in enumerate(coords):
            lat, lon = coord['world_latitude'], coord['world_longitude']
            grid_x = int(lon / grid_size)
            grid_y = int(lat / grid_size)
            spatial_hash[(grid_x, grid_y)].append((i, coord))
        
        # Keep one representative point per grid cell
        deduplicated = []
        for cell_coords in spatial_hash.values():
            if len(cell_coords) == 1:
                deduplicated.append(cell_coords[0][1])
            else:
                # Keep the point with highest confidence if available
                best_coord = max(cell_coords, 
                               key=lambda x: x[1].get('confidence_score', 0.5))[1]
                deduplicated.append(best_coord)
        
        return deduplicated
    
    def _cluster_dense_areas(self, coords: List[Dict], cluster_radius_m: float) -> List[Dict]:
        """
        Replace very dense clusters with representative points
        
        Args:
            coords: Input coordinates
            cluster_radius_m: Clustering radius in meters
            
        Returns:
            List[Dict]: Coordinates with dense areas clustered
        """
        # This is an aggressive optimization for extremely dense datasets
        # Use DBSCAN clustering to identify dense areas
        try:
            from sklearn.cluster import DBSCAN
            
            # Extract coordinates for clustering
            points = np.array([[c['world_latitude'], c['world_longitude']] for c in coords])
            
            # Convert radius to degrees (approximate)
            eps_deg = cluster_radius_m / 111000
            
            # Perform clustering
            clustering = DBSCAN(eps=eps_deg, min_samples=5).fit(points)
            labels = clustering.labels_
            
            clustered_coords = []
            cluster_groups = defaultdict(list)
            
            # Group points by cluster
            for i, label in enumerate(labels):
                if label == -1:  # Noise points (not in clusters)
                    clustered_coords.append(coords[i])
                else:
                    cluster_groups[label].append(coords[i])
            
            # Replace each cluster with representative point
            for cluster_points in cluster_groups.values():
                if len(cluster_points) > 3:  # Only cluster if significant density
                    # Use centroid with highest confidence
                    best_point = max(cluster_points, 
                                   key=lambda x: x.get('confidence_score', 0.5))
                    clustered_coords.append(best_point)
                else:
                    clustered_coords.extend(cluster_points)
            
            return clustered_coords
            
        except ImportError:
            # sklearn not available, skip clustering
            return coords
    
    def _create_polygons_parallel(self, coords: List[Dict], buffer_size_m: float,
                                 progress_callback: Optional[Callable] = None) -> Tuple[List[Polygon], List[Dict]]:
        """
        Create buffered polygons using parallel processing
        
        Args:
            coords: Preprocessed coordinates
            buffer_size_m: Buffer size in meters
            progress_callback: Progress reporting callback
            
        Returns:
            Tuple[List[Polygon], List[Dict]]: Buffered polygons and associated data
        """
        # Calculate optimal chunk size based on dataset size and workers
        optimal_chunk_size = max(100, len(coords) // (self.n_workers * 4))
        chunk_size = min(self.chunk_size, optimal_chunk_size)
        
        # Split coordinates into chunks
        chunks = [coords[i:i + chunk_size] for i in range(0, len(coords), chunk_size)]
        
        try:
            # Process chunks in parallel
            with Pool(processes=self.n_workers) as pool:
                # Create arguments for each chunk
                chunk_args = [(chunk, buffer_size_m, i) for i, chunk in enumerate(chunks)]
                results = pool.starmap(_process_polygon_chunk_worker, chunk_args)
            
            # Combine results
            all_polygons = []
            all_data = []
            
            for polygons, data in results:
                all_polygons.extend(polygons)
                all_data.extend(data)
            
            return all_polygons, all_data
            
        except Exception as e:
            print(f"Warning: Parallel processing failed, using sequential processing")
            return self._create_polygons_sequential(coords, buffer_size_m)
    
    def _process_polygon_chunk(self, coords_chunk: List[Dict], 
                              buffer_size_m: float) -> Tuple[List[Polygon], List[Dict]]:
        """
        Process a chunk of coordinates into buffered polygons
        
        Args:
            coords_chunk: Chunk of coordinates to process
            buffer_size_m: Buffer size in meters
            
        Returns:
            Tuple[List[Polygon], List[Dict]]: Polygons and data for this chunk
        """
        polygons = []
        polygon_data = []
        
        for i, coord in enumerate(coords_chunk):
            try:
                # Create point geometry
                point = Point(coord['world_longitude'], coord['world_latitude'])
                
                # Calculate buffer size in degrees
                buffer_degrees = self._meters_to_degrees(
                    buffer_size_m, coord['world_latitude']
                )
                
                # Create buffered polygon
                buffered_polygon = point.buffer(buffer_degrees)
                polygons.append(buffered_polygon)
                
                # Store associated data
                polygon_data.append({
                    'original_coord': coord,
                    'polygon_index': len(polygon_data)
                })
                
            except Exception as e:
                print(f"Error processing coordinate {i}: {e}")
                continue
        
        return polygons, polygon_data
    
    def _create_polygons_sequential(self, coords: List[Dict], 
                                   buffer_size_m: float) -> Tuple[List[Polygon], List[Dict]]:
        """
        Create buffered polygons sequentially (fallback method)
        
        Args:
            coords: Coordinates to process
            buffer_size_m: Buffer size in meters
            
        Returns:
            Tuple[List[Polygon], List[Dict]]: Buffered polygons and associated data
        """
        return self._process_polygon_chunk(coords, buffer_size_m)
    
    def _merge_with_spatial_index(self, polygons: List[Polygon], polygon_data: List[Dict],
                                 progress_callback: Optional[Callable] = None) -> List[Dict]:
        """
        Merge overlapping polygons using R-tree spatial indexing for O(n log n) performance
        
        Args:
            polygons: List of buffered polygons
            polygon_data: Associated data for each polygon
            progress_callback: Progress reporting callback
            
        Returns:
            List[Dict]: Merged polygon features
        """
        if not RTREE_AVAILABLE:
            return self._merge_overlapping_polygons(polygons, polygon_data)
        
        # Build spatial index
        idx = index.Index()
        for i, polygon in enumerate(polygons):
            try:
                idx.insert(i, polygon.bounds)
            except Exception as e:
                continue
        
        # Find overlapping groups using spatial index
        visited = set()
        merged_groups = []
        processed_count = 0
        
        for i, polygon in enumerate(polygons):
            if i in visited:
                continue
            
            # Start new group
            current_group = [i]
            visited.add(i)
            to_check = [i]
            
            # Find all connected overlapping polygons using spatial index
            while to_check:
                current_idx = to_check.pop(0)
                current_polygon = polygons[current_idx]
                
                try:
                    # Query spatial index for potential overlaps
                    candidates = list(idx.intersection(current_polygon.bounds))
                    
                    for candidate_idx in candidates:
                        if candidate_idx in visited:
                            continue
                        
                        candidate_polygon = polygons[candidate_idx]
                        
                        # Check actual intersection (not just bounding box)
                        try:
                            if (current_polygon.intersects(candidate_polygon) and 
                                not current_polygon.touches(candidate_polygon)):
                                current_group.append(candidate_idx)
                                visited.add(candidate_idx)
                                to_check.append(candidate_idx)
                        except Exception as e:
                            # Skip problematic polygon intersections
                            continue
                            
                except Exception as e:
                    continue
            
            merged_groups.append(current_group)
            processed_count += len(current_group)
            
            # Progress reporting every 1000 processed polygons
            if progress_callback and processed_count % 1000 == 0:
                progress = 70 + (processed_count / len(polygons)) * 20
                progress_callback(progress, f"Merging: {processed_count:,}/{len(polygons):,} processed")
        
        # Create merged polygon features
        merged_features = []
        for group_idx, group_indices in enumerate(merged_groups):
            try:
                group_polygons = [polygons[i] for i in group_indices]
                group_data = [polygon_data[i] for i in group_indices]
                
                # Merge polygons using unary_union
                if len(group_polygons) == 1:
                    merged_geometry = group_polygons[0]
                else:
                    merged_geometry = unary_union(group_polygons)
                
                # Handle both single polygons and multipolygons
                if hasattr(merged_geometry, 'geoms'):
                    for geom in merged_geometry.geoms:
                        if isinstance(geom, Polygon):
                            merged_feature = self._create_merged_feature(
                                geom, group_data, len(group_indices)
                            )
                            merged_features.append(merged_feature)
                else:
                    merged_feature = self._create_merged_feature(
                        merged_geometry, group_data, len(group_indices)
                    )
                    merged_features.append(merged_feature)
                    
            except Exception as e:
                # Add individual polygons if merging fails
                for idx in group_indices:
                    try:
                        individual_feature = self._create_merged_feature(
                            polygons[idx], [polygon_data[idx]], 1
                        )
                        merged_features.append(individual_feature)
                    except:
                        continue
        
        return merged_features
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        if not PSUTIL_AVAILABLE:
            return 0.0
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _log_performance_stats(self):
        """Log performance statistics"""
        stats = self.performance_stats
        
        print(f"\n🚀 Optimized Polygon Processing Performance:")
        print(f"   Points processed: {stats['points_processed']:,}")
        print(f"   After preprocessing: {stats['points_after_preprocessing']:,} "
              f"({(1 - stats['points_after_preprocessing']/stats['points_processed'])*100:.1f}% reduction)")
        print(f"   Polygons created: {stats['polygons_created']:,}")
        print(f"   After merging: {stats['polygons_after_merging']:,}")
        print(f"   Preprocessing time: {stats['preprocessing_time']:.2f}s")
        print(f"   Buffering time: {stats['buffering_time']:.2f}s")
        print(f"   Merging time: {stats['merging_time']:.2f}s")
        print(f"   Total time: {stats['total_time']:.2f}s")
        print(f"   Peak memory: {stats['memory_peak_mb']:.1f} MB")
        print(f"   Processing rate: {stats['points_processed']/stats['total_time']:.0f} points/second")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics dictionary"""
        return self.performance_stats.copy()


def _process_polygon_chunk_worker(coords_chunk: List[Dict], buffer_size_m: float, chunk_id: int = 0) -> Tuple[List[Polygon], List[Dict]]:
    """
    Worker function for parallel polygon processing
    (Must be at module level for multiprocessing)
    """
    polygons = []
    polygon_data = []
    
    for i, coord in enumerate(coords_chunk):
        try:
            # Create point geometry
            point = Point(coord['world_longitude'], coord['world_latitude'])
            
            # Calculate buffer size in degrees at this latitude
            lat_radians = math.radians(coord['world_latitude'])
            degrees_per_meter = 360 / (2 * math.pi * 6371004)  # Earth radius
            lat_degrees = buffer_size_m * degrees_per_meter
            lon_degrees = buffer_size_m * degrees_per_meter / math.cos(lat_radians)
            buffer_degrees = (lat_degrees + lon_degrees) / 2  # Average for circular buffer
            
            # Create buffered polygon
            buffered_polygon = point.buffer(buffer_degrees)
            polygons.append(buffered_polygon)
            
            # Store associated data
            polygon_data.append({
                'original_coord': coord,
                'polygon_index': len(polygon_data)
            })
            
        except Exception as e:
            print(f"Error processing coordinate {i} in chunk {chunk_id}: {e}")
            continue
    
    return polygons, polygon_data