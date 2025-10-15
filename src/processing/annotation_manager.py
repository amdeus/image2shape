#!/usr/bin/env python3
"""
Annotation manager for saving and loading image annotations
"""

import json
import os
from typing import List, Dict, Tuple

class AnnotationManager:
    def __init__(self, annotations_dir="data/annotations"):
        self.annotations_dir = annotations_dir
        self.annotations = {}  # image_path -> list of annotations
        
        # Create annotations directory if it doesn't exist
        os.makedirs(self.annotations_dir, exist_ok=True)
        
    def add_annotation(self, image_path: str, bbox: Tuple[int, int, int, int], annotation_type: str):
        """Add an annotation for an image"""
        if image_path not in self.annotations:
            self.annotations[image_path] = []
            
        annotation = {
            'bbox': bbox,  # (x1, y1, x2, y2)
            'type': annotation_type,  # 'feature' or 'background'
            'timestamp': self._get_timestamp()
        }
        
        self.annotations[image_path].append(annotation)
        self.save_annotations(image_path)
        
    def get_annotations(self, image_path: str) -> List[Dict]:
        """Get all annotations for an image"""
        return self.annotations.get(image_path, [])
        
    def clear_annotations(self, image_path: str):
        """Clear all annotations for an image"""
        if image_path in self.annotations:
            self.annotations[image_path] = []
            self.save_annotations(image_path)
    
    def clear_all_annotations(self):
        """Clear all annotations from all images"""
        # Get all image paths that have annotations
        image_paths = list(self.annotations.keys())
        
        # Clear the annotations dictionary
        self.annotations.clear()
        
        # Remove all annotation files
        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            annotation_file = os.path.join(self.annotations_dir, f"{image_name}.json")
            if os.path.exists(annotation_file):
                try:
                    os.remove(annotation_file)
                except Exception as e:
                    # Log warning but continue - non-critical error
                    pass
            
    def save_annotations(self, image_path: str):
        """Save annotations for an image to JSON file"""
        if image_path not in self.annotations:
            return
            
        # Create filename based on image name
        image_name = os.path.basename(image_path)
        annotation_file = os.path.join(self.annotations_dir, f"{image_name}.json")
        
        annotation_data = {
            'image_path': image_path,
            'annotations': self.annotations[image_path]
        }
        
        try:
            with open(annotation_file, 'w') as f:
                json.dump(annotation_data, f, indent=2)
        except Exception as e:
            # Log error but continue - annotation save failed
            pass
            
    def load_annotations(self, image_path: str):
        """Load annotations for an image from JSON file"""
        image_name = os.path.basename(image_path)
        annotation_file = os.path.join(self.annotations_dir, f"{image_name}.json")
        
        if not os.path.exists(annotation_file):
            self.annotations[image_path] = []
            return
            
        try:
            with open(annotation_file, 'r') as f:
                annotation_data = json.load(f)
                self.annotations[image_path] = annotation_data.get('annotations', [])
        except Exception as e:
            # Error loading annotations - start with empty list
            self.annotations[image_path] = []
            
    def get_all_annotations(self) -> Dict[str, List[Dict]]:
        """Get all annotations for all images"""
        return self.annotations.copy()
        
    def get_training_data(self) -> Tuple[List, List]:
        """Extract training data from annotations"""
        features = []
        labels = []
        
        for image_path, annotations in self.annotations.items():
            for annotation in annotations:
                bbox = annotation['bbox']
                annotation_type = annotation['type']
                
                # Add to training data
                features.append({
                    'image_path': image_path,
                    'bbox': bbox
                })
                
                # Convert type to numeric label
                label = 1 if annotation_type == 'feature' else 0
                labels.append(label)
                
        return features, labels
        
    def get_annotation_count(self) -> Dict[str, int]:
        """Get count of annotations by type"""
        feature_count = 0
        background_count = 0
        
        for annotations in self.annotations.values():
            for annotation in annotations:
                if annotation['type'] == 'feature':
                    feature_count += 1
                else:
                    background_count += 1
                    
        return {
            'feature': feature_count,
            'background': background_count,
            'total': feature_count + background_count
        }
        
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        import datetime
        return datetime.datetime.now().isoformat()
        
    def export_annotations(self, output_file: str):
        """Export all annotations to a single file"""
        export_data = {
            'annotations': self.annotations,
            'export_timestamp': self._get_timestamp(),
            'annotation_counts': self.get_annotation_count()
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            # Annotations exported successfully
            pass
        except Exception as e:
            # Error exporting annotations
            pass
            
    def import_annotations(self, input_file: str):
        """Import annotations from a file"""
        try:
            with open(input_file, 'r') as f:
                import_data = json.load(f)
                self.annotations.update(import_data.get('annotations', {}))
            # Annotations imported successfully
            pass
        except Exception as e:
            # Error importing annotations
            pass