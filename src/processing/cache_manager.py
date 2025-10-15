#!/usr/bin/env python3
"""
Cache Management System for Image2Shape

Manages application data directory, display image cache, and metadata cache
for optimal performance during drone image processing.

Key Features:
- Application data directory management (~/.image2shape/)
- Hash-based folder identification for unique cache keys
- Display image cache with 1/3 scale optimization
- Metadata cache for instant georeferencing
- Automatic cleanup and size monitoring

Author: Image2Shape Development Team
Version: 3.0 - Pre-Caching & Optimization
"""

import os
import json
import hashlib
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import cv2

class CacheManager:
    """
    Manages caching system for Image2Shape application
    """
    
    def __init__(self):
        """Initialize cache manager with application data directory"""
        self.app_data_dir = self._get_app_data_directory()
        self.cache_dir = self.app_data_dir / "cache"
        self.models_dir = self.app_data_dir / "models"
        self.logs_dir = self.app_data_dir / "logs"
        
        # Cache size limits (in MB)
        self.max_total_cache_size = 2048  # 2GB total
        self.max_folder_cache_size = 500  # 500MB per folder
        
        self._ensure_directories()
    
    def _get_app_data_directory(self) -> Path:
        """Get application data directory path"""
        home = Path.home()
        app_data = home / ".image2shape"
        return app_data
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        for directory in [self.app_data_dir, self.cache_dir, self.models_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_folder_hash(self, folder_path: str) -> str:
        """
        Generate unique hash for folder path
        
        Args:
            folder_path: Path to image folder
            
        Returns:
            Unique hash string for the folder
        """
        # Use absolute path for consistent hashing
        abs_path = os.path.abspath(folder_path)
        folder_hash = hashlib.md5(abs_path.encode()).hexdigest()[:12]
        return folder_hash
    
    def get_cache_folder(self, image_folder: str) -> Path:
        """
        Get cache folder path for specific image folder
        
        Args:
            image_folder: Path to source image folder
            
        Returns:
            Path to cache folder for this image set
        """
        folder_hash = self.get_folder_hash(image_folder)
        cache_folder = self.cache_dir / folder_hash
        cache_folder.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (cache_folder / "display_images").mkdir(exist_ok=True)
        
        return cache_folder
    
    def get_metadata_cache_path(self, image_folder: str) -> Path:
        """Get path to metadata cache file"""
        cache_folder = self.get_cache_folder(image_folder)
        return cache_folder / "metadata.json"
    
    def get_display_image_path(self, image_folder: str, image_filename: str) -> Path:
        """Get path to cached display image"""
        cache_folder = self.get_cache_folder(image_folder)
        display_filename = f"{Path(image_filename).stem}_display.jpg"
        return cache_folder / "display_images" / display_filename
    
    def save_metadata_cache(self, image_folder: str, metadata_cache: Dict):
        """
        Save metadata cache to file
        
        Args:
            image_folder: Source image folder path
            metadata_cache: Dictionary of metadata for all images
        """
        cache_path = self.get_metadata_cache_path(image_folder)
        
        # Convert any non-serializable objects to strings
        serializable_cache = {}
        for image_path, metadata in metadata_cache.items():
            serializable_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    serializable_metadata[key] = value
                else:
                    serializable_metadata[key] = str(value)
            serializable_cache[image_path] = serializable_metadata
        
        # Add cache metadata
        cache_data = {
            'cache_version': '3.0',
            'created_timestamp': time.time(),
            'source_folder': image_folder,
            'image_count': len(metadata_cache),
            'metadata': serializable_cache
        }
        
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
    
    def load_metadata_cache(self, image_folder: str) -> Optional[Dict]:
        """
        Load metadata cache from file
        
        Args:
            image_folder: Source image folder path
            
        Returns:
            Metadata cache dictionary or None if not found/invalid
        """
        cache_path = self.get_metadata_cache_path(image_folder)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            # Validate cache version and data
            if cache_data.get('cache_version') != '3.0':
                return None
            
            return cache_data.get('metadata', {})
            
        except (json.JSONDecodeError, KeyError):
            return None
    
    def create_display_image(self, source_image_path: str, image_folder: str) -> Optional[str]:
        """
        Create and cache 1/3 scale display image
        
        Args:
            source_image_path: Path to original image
            image_folder: Source folder for cache organization
            
        Returns:
            Path to cached display image or None if failed
        """
        try:
            image_filename = os.path.basename(source_image_path)
            cache_path = self.get_display_image_path(image_folder, image_filename)
            
            # Check if cached version exists and is newer than source
            if (cache_path.exists() and 
                cache_path.stat().st_mtime > Path(source_image_path).stat().st_mtime):
                return str(cache_path)
            
            # Load and resize image
            img = cv2.imread(source_image_path)
            if img is None:
                return None
            
            # Resize to 1/3 scale
            height, width = img.shape[:2]
            new_height, new_width = height // 3, width // 3
            
            resized = cv2.resize(img, (new_width, new_height), 
                               interpolation=cv2.INTER_AREA)
            
            # Save with high quality JPEG compression
            cv2.imwrite(str(cache_path), resized, 
                       [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            return str(cache_path)
            
        except Exception as e:
            print(f"Error creating display image for {source_image_path}: {e}")
            return None
    
    def get_cache_size(self, cache_folder: Optional[Path] = None) -> float:
        """
        Get cache size in MB
        
        Args:
            cache_folder: Specific cache folder, or None for total cache size
            
        Returns:
            Cache size in megabytes
        """
        if cache_folder is None:
            cache_folder = self.cache_dir
        
        if not cache_folder.exists():
            return 0.0
        
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(cache_folder):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    continue
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def cleanup_old_caches(self, max_age_days: int = 30):
        """
        Clean up old cache folders
        
        Args:
            max_age_days: Maximum age of cache folders to keep
        """
        if not self.cache_dir.exists():
            return
        
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        for cache_folder in self.cache_dir.iterdir():
            if not cache_folder.is_dir():
                continue
            
            # Check folder age
            folder_age = current_time - cache_folder.stat().st_mtime
            
            if folder_age > max_age_seconds:
                try:
                    shutil.rmtree(cache_folder)
                    print(f"Removed old cache folder: {cache_folder.name}")
                except Exception as e:
                    print(f"Error removing cache folder {cache_folder}: {e}")
    
    def cleanup_large_caches(self):
        """Clean up caches if total size exceeds limit"""
        total_size = self.get_cache_size()
        
        if total_size <= self.max_total_cache_size:
            return
        
        print(f"Cache size ({total_size:.1f}MB) exceeds limit ({self.max_total_cache_size}MB)")
        
        # Get all cache folders with their ages
        cache_folders = []
        for cache_folder in self.cache_dir.iterdir():
            if cache_folder.is_dir():
                age = time.time() - cache_folder.stat().st_mtime
                size = self.get_cache_size(cache_folder)
                cache_folders.append((age, size, cache_folder))
        
        # Sort by age (oldest first)
        cache_folders.sort(key=lambda x: x[0], reverse=True)
        
        # Remove oldest folders until under limit
        for age, size, cache_folder in cache_folders:
            try:
                shutil.rmtree(cache_folder)
                total_size -= size
                print(f"Removed cache folder: {cache_folder.name} ({size:.1f}MB)")
                
                if total_size <= self.max_total_cache_size:
                    break
                    
            except Exception as e:
                print(f"Error removing cache folder {cache_folder}: {e}")
    
    def get_cache_info(self, image_folder: str) -> Dict:
        """
        Get cache information for specific folder
        
        Args:
            image_folder: Source image folder path
            
        Returns:
            Dictionary with cache information
        """
        cache_folder = self.get_cache_folder(image_folder)
        metadata_cache_path = self.get_metadata_cache_path(image_folder)
        
        # Count display images
        display_images_dir = cache_folder / "display_images"
        display_image_count = 0
        if display_images_dir.exists():
            display_image_count = len(list(display_images_dir.glob("*.jpg")))
        
        return {
            'cache_folder': str(cache_folder),
            'cache_size_mb': self.get_cache_size(cache_folder),
            'has_metadata_cache': metadata_cache_path.exists(),
            'display_image_count': display_image_count,
            'folder_hash': self.get_folder_hash(image_folder)
        }
    
    def clear_folder_cache(self, image_folder: str):
        """
        Clear cache for specific folder
        
        Args:
            image_folder: Source image folder path
        """
        cache_folder = self.get_cache_folder(image_folder)
        
        if cache_folder.exists():
            try:
                shutil.rmtree(cache_folder)
                print(f"Cleared cache for folder: {image_folder}")
            except Exception as e:
                print(f"Error clearing cache: {e}")
    
    def get_total_cache_info(self) -> Dict:
        """Get information about total cache usage"""
        total_size = self.get_cache_size()
        
        # Count cache folders
        folder_count = 0
        if self.cache_dir.exists():
            folder_count = len([d for d in self.cache_dir.iterdir() if d.is_dir()])
        
        return {
            'total_size_mb': total_size,
            'max_size_mb': self.max_total_cache_size,
            'usage_percent': (total_size / self.max_total_cache_size) * 100,
            'folder_count': folder_count,
            'cache_directory': str(self.cache_dir)
        }