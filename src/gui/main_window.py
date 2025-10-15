#!/usr/bin/env python3
"""
Main window for Image2Shape application with pixel-based ML integration

This module provides the main GUI interface for the Image2Shape application,
featuring revolutionary pixel-based machine learning for drone image analysis.

Key Features:
- Pixel-level ML training with 30,000+ samples per session
- Interactive confidence and clustering controls
- Real-time prediction filtering and visualization
- Advanced spatial clustering for noise reduction
- Comprehensive metadata processing and validation

Author: Image2Shape Development Team
Version: 2.0 - Pixel-Based ML Revolution
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
from datetime import datetime
from pathlib import Path

# Import other modules
from gui.image_viewer import ImageViewer
from gui.map_widget import MapWidget
from gui.preprocessing_dialog import PreprocessingDialog
from gui.export_dialog import show_export_dialog
from processing.annotation_manager import AnnotationManager
from processing.metadata_processor import MetadataProcessor
from processing.batch_processor import BatchProcessor
from processing.parallel_batch_processor import ParallelBatchProcessor
from processing.cache_manager import CacheManager
from ml.feature_extractor import FeatureExtractor
from ml.parallel_feature_extractor import ParallelFeatureExtractor
from ml.random_forest_classifier import DroneImageClassifier
from ml.algorithm_factory import MLAlgorithmFactory, create_default_classifier
from gui.ml_algorithm_dialog import show_ml_algorithm_dialog, show_ml_training_progress
from gui.documentation_viewer import show_readme, show_user_guide

class Image2ShapeApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Image2Shape - Drone Image Feature Detection")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 700)
        
        # Configure fonts
        self.setup_fonts()
        
        # Initialize components
        self.image_folders = []  # List of folder paths
        self.image_files = []    # List of full image paths
        self.current_image_index = 0
        self.annotation_manager = AnnotationManager()
        self.metadata_processor = MetadataProcessor()
        self.cache_manager = CacheManager()
        self.all_metadata = {}
        
        # Cache data
        self.metadata_cache = {}
        self.image_cache = {}
        
        # ML components - Use parallel feature extractor for better performance
        self.feature_extractor = ParallelFeatureExtractor()
        self.classifier = create_default_classifier()  # Use new architecture with backward compatibility
        self.batch_processor = BatchProcessor(self.feature_extractor, self.classifier)
        self.model_trained = False
        self.current_algorithm_type = 'random_forest'  # Track current algorithm
        self.show_predictions = True  # Default to True
        
        # Processing time tracking
        self.processing_times = {
            'load_folder': 0.0,
            'train_model': 0.0,
            'export': 0.0
        }
        
        # Prediction storage and filtering - moved to image viewer
        self.current_predictions = None
        
        self.setup_ui()
        self.setup_bindings()
        self.root.bind('<Configure>', self.on_window_resize)
        
    def setup_fonts(self):
        """Configure fonts for better readability"""
        self.default_font = ('Arial', 10)
        self.heading_font = ('Arial', 11, 'bold')
        self.metadata_font = ('Courier New', 9)
        self.button_font = ('Arial', 9)
        
        style = ttk.Style()
        style.configure('TButton', font=self.button_font)
        style.configure('TLabel', font=self.default_font)
        style.configure('Heading.TLabel', font=self.heading_font)
        
        self.root.option_add('*Font', self.default_font)
        
    def setup_ui(self):
        """Set up the user interface"""
        self.create_menu()
        self.create_toolbar()
        self.create_main_content()
        self.create_status_bar()
        
    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Folder", command=self.load_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        process_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Process", menu=process_menu)
        process_menu.add_command(label="Train Model", command=self.train_model)
        process_menu.add_command(label="Export Results", command=self.show_export_dialog)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)
        help_menu.add_command(label="README", command=self.show_readme)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about)
        
    def create_toolbar(self):
        """Create toolbar with main buttons"""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        ttk.Button(toolbar, text="Load Folder", command=self.load_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Train Model", command=self.train_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Export Results", command=self.show_export_dialog).pack(side=tk.LEFT, padx=5)
        
        # Removed Clear buttons and image counter - moved to image viewer
        
    def create_main_content(self):
        """Create main content area with resizable panels"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel with fixed width (450px)
        left_frame = ttk.Frame(paned, width=450)
        left_frame.pack_propagate(False)  # Maintain fixed width
        paned.add(left_frame, weight=0)
        
        left_paned = ttk.PanedWindow(left_frame, orient=tk.VERTICAL)
        left_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Images section
        images_frame = ttk.LabelFrame(left_paned, text="Images")
        left_paned.add(images_frame, weight=1)
        
        list_container = ttk.Frame(images_frame)
        list_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.image_listbox = tk.Listbox(list_container, font=self.default_font)
        scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.image_listbox.yview)
        self.image_listbox.config(yscrollcommand=scrollbar.set)
        
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)
        
        # Map section
        map_frame = ttk.Frame(left_paned)
        left_paned.add(map_frame, weight=1)
        self.map_widget = MapWidget(map_frame)
        
        # Set up map click callback
        self.map_widget.set_click_callback(self.on_map_image_click)
        
        # Metadata section
        metadata_frame = ttk.LabelFrame(left_paned, text="Metadata Summary")
        left_paned.add(metadata_frame, weight=1)
        
        metadata_container = ttk.Frame(metadata_frame)
        metadata_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.metadata_text = tk.Text(
            metadata_container,
            wrap=tk.WORD,
            font=self.metadata_font,
            bg='#f8f9fa',
            fg='#212529',
            selectbackground='#007acc',
            relief=tk.FLAT,
            borderwidth=1
        )
        
        metadata_scrollbar = ttk.Scrollbar(metadata_container, orient=tk.VERTICAL, command=self.metadata_text.yview)
        self.metadata_text.config(yscrollcommand=metadata_scrollbar.set)
        
        self.metadata_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        metadata_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Right panel - Image viewer
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=4)
        
        
        self.image_viewer = ImageViewer(right_frame, self.annotation_manager)
        self.image_viewer.set_main_window(self)  # Set reference for button callbacks
        
        # Set up prediction controls in image viewer
        self.image_viewer.setup_prediction_controls()
        
    def create_status_bar(self):
        """Create status bar"""
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def setup_bindings(self):
        """Set up keyboard bindings"""
        self.root.bind('<Left>', lambda e: self.previous_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('<Control-o>', lambda e: self.load_folder())
        self.root.focus_set()
        
    def on_window_resize(self, event):
        """Handle window resize events"""
        if event.widget == self.root:
            if hasattr(self, 'image_viewer') and self.image_viewer.original_image is not None:
                self.root.after(50, self.image_viewer.fit_to_window)
    
    def on_map_image_click(self, image_path):
        """Handle clicks on map dots to navigate to specific images"""
        if not self.image_files:
            return
        
        # Find the index of the clicked image (image_path is already full path)
        try:
            image_index = self.image_files.index(image_path)
            self.current_image_index = image_index
            self.load_current_image()
        except ValueError:
            # Image not found in current list
            print(f"Image {os.path.basename(image_path)} not found in current image list")
        
    def load_folder(self):
        """Enhanced folder loading with preprocessing dialog"""
        start_time = datetime.now()
        
        folder_path = filedialog.askdirectory(title="Select Drone Images Folder")
        if not folder_path:
            return
        
        # Get image files
        image_files = self.get_image_files(folder_path)
        if not image_files:
            messagebox.showerror("Error", "No valid image files found in folder")
            return
        
        # Show preprocessing dialog
        preprocessing_dialog = PreprocessingDialog(self.root, folder_path, image_files)
        
        # Wait for preprocessing to complete
        self.root.wait_window(preprocessing_dialog.dialog)
        
        # Check if preprocessing was successful
        if preprocessing_dialog.success and preprocessing_dialog.metadata_cache:
            self.setup_folder_data(
                folder_path, 
                image_files, 
                preprocessing_dialog.metadata_cache,
                preprocessing_dialog.image_cache,
                preprocessing_dialog.summary
            )
            
            # Record processing time
            end_time = datetime.now()
            self.processing_times['load_folder'] = (end_time - start_time).total_seconds()
        else:
            if not preprocessing_dialog.cancelled:
                messagebox.showerror("Error", "Preprocessing failed or was cancelled")
    
    def get_image_files(self, folder_path):
        """Get list of supported image files from folder"""
        extensions = ('.jpg', '.jpeg', '.png', '.tiff', '.tif')
        image_files = []
        
        try:
            for file in os.listdir(folder_path):
                if file.lower().endswith(extensions):
                    image_files.append(file)
        except Exception as e:
            print(f"Error reading folder {folder_path}: {e}")
            return []
        
        image_files.sort()
        return image_files
    
    def setup_folder_data(self, folder_path, image_files, metadata_cache, image_cache, summary):
        """Setup application with preprocessed data (additive - adds to existing data)"""
        # Initialize if first folder
        if not hasattr(self, 'metadata_cache'):
            self.metadata_cache = {}
            self.image_cache = {}
        
        # Add new folder to list
        if folder_path not in self.image_folders:
            self.image_folders.append(folder_path)
        
        # Add new images with full paths
        new_image_paths = [os.path.join(folder_path, f) for f in image_files]
        
        # Filter out duplicates (in case same folder loaded twice)
        for img_path in new_image_paths:
            if img_path not in self.image_files:
                self.image_files.append(img_path)
        
        # Merge caches (new data overwrites old if same keys)
        self.metadata_cache.update(metadata_cache)
        self.image_cache.update(image_cache)
        
        # Update processor caches
        self.metadata_processor.metadata_cache = self.metadata_cache
        self.image_viewer.set_image_cache(self.image_cache)
        
        # Initialize batch processor with cached metadata
        self.batch_processor = BatchProcessor(
            self.feature_extractor, 
            self.classifier
        )
        
        # Update UI
        self.update_image_list()
        self.display_combined_summary()
        
        # Load all existing annotations
        self.load_all_existing_annotations()
        
        # Validate metadata for display
        self.validate_metadata()
        
        # Enable processing controls
        self.enable_processing_controls()
        
        # Load current image with proper delay to ensure UI is ready (only if no image currently loaded)
        if self.current_image_index >= len(self.image_files):
            self.current_image_index = len(self.image_files) - len(new_image_paths)  # First image of new folder
        
        self.root.after(100, self.load_current_image)
        
        total_images = len(self.image_files)
        new_images = len(new_image_paths)
        self.status_var.set(f"Added {new_images} images. Total: {total_images} images from {len(self.image_folders)} folders")
    
    def display_preprocessing_summary(self, summary):
        """Display preprocessing results in metadata area"""
        summary_text = f"""Folder Loaded Successfully

Dataset Summary:
   Total Images: {summary['total_images']}
   Valid Metadata: {summary['valid_metadata']} ({summary['valid_metadata']/summary['total_images']*100:.1f}%)
   Valid Display Images: {summary['valid_display_images']}

GPS Information:
   GPS Coordinates: {summary['gps_coordinates']} valid
   RTK Positioning: {summary['rtk_positioning']} images

Cache Information:
   Cache Size: {summary['cache_size_mb']:.1f} MB
   Processing Time: {summary['processing_time']:.1f} seconds

Ready for annotation and feature detection!

ML Training Status:
   Annotations: 0 total (add annotations to train model)
   Model Status: Not Trained
"""
        
        # Update metadata display
        self.metadata_text.delete(1.0, tk.END)
        self.metadata_text.insert(1.0, summary_text)
    
    def enable_processing_controls(self):
        """Enable processing controls after successful folder load"""
        # All controls should already be enabled, but we can add specific logic here if needed
        pass
    
    def load_all_existing_annotations(self):
        """Load all existing annotation files for the current folder"""
        if not self.image_files:
            return
        
        loaded_count = 0
        
        for image_path in self.image_files:
            try:
                self.annotation_manager.load_annotations(image_path)
                annotations = self.annotation_manager.get_annotations(image_path)
                if annotations:
                    loaded_count += len(annotations)
            except Exception as e:
                print(f"Error loading annotations for {os.path.basename(image_path)}: {e}")
        
        if loaded_count > 0:
            print(f"Loaded {loaded_count} annotations from {len([p for p in self.image_files if self.annotation_manager.get_annotations(p)])} images")
        
        # Update the display
        self.display_combined_summary()
    
        
            
    def update_image_list(self):
        """Update the image listbox"""
        self.image_listbox.delete(0, tk.END)
        for img_path in self.image_files:
            # Show folder name + image name for clarity
            folder_name = os.path.basename(os.path.dirname(img_path))
            image_name = os.path.basename(img_path)
            display_name = f"{folder_name}/{image_name}"
            self.image_listbox.insert(tk.END, display_name)
        self.update_image_counter()
        
    def update_image_counter(self):
        """Update image counter display"""
        if self.image_files:
            # Update counters in image viewer
            self.image_viewer.update_image_counter(self.current_image_index, len(self.image_files))
        else:
            # Update counter in image viewer
            self.image_viewer.update_image_counter(0, 0)
            
    def previous_image(self):
        """Navigate to previous image"""
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
            
    def next_image(self):
        """Navigate to next image"""
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_current_image()
            
    def on_image_select(self, event):
        """Handle image selection from listbox"""
        selection = self.image_listbox.curselection()
        if selection:
            self.current_image_index = selection[0]
            self.load_current_image()
            
    def load_current_image(self):
        """Load the current image in the viewer"""
        if not self.image_files or self.current_image_index >= len(self.image_files):
            return
            
        image_path = self.image_files[self.current_image_index]  # Already full path
        self.image_viewer.load_image(image_path)
        
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(self.current_image_index)
        self.image_listbox.see(self.current_image_index)
        
        self.update_image_counter()
        
        # Load annotations for current image
        self.annotation_manager.load_annotations(image_path)
        self.image_viewer.update_annotations()
        
        # Also refresh the metadata display to show current annotation counts
        self.display_combined_summary()
        
        self.root.after(100, self.image_viewer.fit_to_window)
        
        if hasattr(self, 'map_widget'):
            current_image_path = self.image_files[self.current_image_index] if self.image_files else None
            self.map_widget.highlight_current_image(self.current_image_index, current_image_path)
            
        # Auto-update predictions if they're enabled and model is trained (with delay for proper loading)
        if self.show_predictions and self.model_trained:
            self.root.after(150, self.update_predictions)
            
    def validate_metadata(self):
        """Validate metadata for loaded images and cache all data"""
        if not self.image_files:
            return
            
        self.status_var.set("Extracting metadata...")
        self.root.update()
        
        try:
            # Use existing metadata cache instead of re-extracting
            if hasattr(self, 'metadata_cache') and self.metadata_cache:
                self.all_metadata = self.metadata_cache.copy()
            else:
                self.all_metadata = {}
                for i, image_path in enumerate(self.image_files):
                    self.all_metadata[image_path] = self.metadata_processor.extract_metadata(image_path)
                
                progress = (i + 1) / len(self.image_files) * 100
                self.status_var.set(f"Extracting metadata... {progress:.0f}%")
                self.root.update()
            
            # Use combined summary for multi-folder setup
            self.display_combined_summary()
            
            self.map_widget.update_locations(self.all_metadata)
            
            self.status_var.set("Metadata extraction complete")
            
        except Exception as e:
            error_summary = f"""Metadata Extraction Error:

Error: {str(e)}

This might happen if exiftool is not installed
or if the images don't contain GPS metadata.

Total Images: {len(self.image_files)}
Status: Unable to extract metadata"""
            
            self.metadata_text.delete(1.0, tk.END)
            self.metadata_text.insert(1.0, error_summary)
            
            self.status_var.set("Metadata extraction failed")
            
    def create_enhanced_summary(self, validation_results):
        """Create simplified metadata summary with camera/lens info"""
        total = validation_results['total_images']
        gps_valid = validation_results['gps_valid']
        rtk_count = validation_results['rtk_count']
        
        camera_info = "Unknown"
        focal_length = "Unknown"
        image_size = "Unknown"
        
        for metadata in self.all_metadata.values():
            if metadata.get('camera_model'):
                camera_info = metadata['camera_model']
            if metadata.get('focal_length'):
                focal_length = f"{metadata['focal_length']} mm"
                if metadata.get('focal_length_35mm'):
                    focal_length += f" ({metadata['focal_length_35mm']} mm equiv.)"
            if metadata.get('image_width') and metadata.get('image_height'):
                image_size = f"{metadata['image_width']} x {metadata['image_height']}"
            break
        
        annotation_counts = self.annotation_manager.get_annotation_count()
        
        summary = f"""Dataset Summary:
   Total Images: {total}
   GPS Valid: {gps_valid} ({gps_valid/total*100:.1f}%)
   RTK Images: {rtk_count} ({rtk_count/total*100:.1f}%)

Camera & Lens Info:
   Camera: {camera_info}
   Focal Length: {focal_length}
   Image Size: {image_size}

ML Training Data:
   Annotations: {annotation_counts['total']} total
   Features: {annotation_counts['feature']} samples
   Background: {annotation_counts['background']} samples
   Model Status: {'Trained' if self.model_trained else 'Not Trained'}"""

        if validation_results['altitude_range']['min'] is not None:
            alt_min = validation_results['altitude_range']['min']
            alt_max = validation_results['altitude_range']['max']
            summary += f"\n   Altitude Range: {alt_min:.1f}m - {alt_max:.1f}m"
            
        if self.image_files and self.current_image_index < len(self.image_files):
            current_image_path = self.image_files[self.current_image_index]
            if current_image_path in self.metadata_cache:
                current_meta = self.metadata_cache[current_image_path]
                summary += f"\n\nCurrent Image Details:"
                summary += f"\n   File: {os.path.basename(current_image_path)}"
                summary += f"\n   Folder: {os.path.basename(os.path.dirname(current_image_path))}"
                if current_meta.get('altitude_above_ground'):
                    summary += f"\n   Height AGL: {current_meta['altitude_above_ground']}"
                if current_meta.get('datetime_original'):
                    summary += f"\n   Timestamp: {current_meta['datetime_original']}"
                if current_meta.get('gimbal_pitch'):
                    summary += f"\n   Gimbal Pitch: {current_meta['gimbal_pitch']} deg"
                    
        return summary
    
    def display_combined_summary(self):
        """Display combined summary for all loaded folders"""
        if not self.image_files:
            return
        
        # Create combined validation results
        total_images = len(self.image_files)
        gps_valid = sum(1 for path in self.image_files if path in self.metadata_cache and 
                       self.metadata_cache[path].get('latitude') and self.metadata_cache[path].get('longitude'))
        rtk_count = sum(1 for path in self.image_files if path in self.metadata_cache and 
                       (self.metadata_cache[path].get('rtk_status') == 'RTK' or 
                        self.metadata_cache[path].get('gps_status') == 'RTK'))
        
        # Get camera info from first available metadata
        camera_info = "Unknown"
        focal_length = "Unknown"
        image_size = "Unknown"
        
        for img_path in self.image_files:
            if img_path in self.metadata_cache:
                metadata = self.metadata_cache[img_path]
                if metadata.get('camera_model'):
                    camera_info = metadata['camera_model']
                if metadata.get('focal_length'):
                    focal_length = f"{metadata['focal_length']} mm"
                    if metadata.get('focal_length_35mm'):
                        focal_length += f" ({metadata['focal_length_35mm']} mm equiv.)"
                if metadata.get('image_width') and metadata.get('image_height'):
                    image_size = f"{metadata['image_width']} x {metadata['image_height']}"
                break
        
        annotation_counts = self.annotation_manager.get_annotation_count()
        
        summary = f"""Combined Dataset Summary:
   Total Images: {total_images} from {len(self.image_folders)} folders
   GPS Valid: {gps_valid} ({gps_valid/total_images*100:.1f}%)
   RTK Images: {rtk_count} ({rtk_count/total_images*100:.1f}%)

Folders Loaded:"""
        
        for i, folder in enumerate(self.image_folders, 1):
            folder_name = os.path.basename(folder)
            folder_images = sum(1 for path in self.image_files if os.path.dirname(path) == folder)
            summary += f"\n   {i}. {folder_name} ({folder_images} images)"

        summary += f"""

Camera & Lens Info:
   Camera: {camera_info}
   Focal Length: {focal_length}
   Image Size: {image_size}

ML Training Data:
   Annotations: {annotation_counts['total']} total
   Features: {annotation_counts['feature']} samples
   Background: {annotation_counts['background']} samples
   Model Status: {'Trained' if self.model_trained else 'Not Trained'}"""

        if self.image_files and self.current_image_index < len(self.image_files):
            current_image_path = self.image_files[self.current_image_index]
            if current_image_path in self.metadata_cache:
                current_meta = self.metadata_cache[current_image_path]
                summary += f"\n\nCurrent Image Details:"
                summary += f"\n   File: {os.path.basename(current_image_path)}"
                summary += f"\n   Folder: {os.path.basename(os.path.dirname(current_image_path))}"
                if current_meta.get('altitude_above_ground'):
                    summary += f"\n   Height AGL: {current_meta['altitude_above_ground']}"
                if current_meta.get('datetime_original'):
                    summary += f"\n   Timestamp: {current_meta['datetime_original']}"
                if current_meta.get('gimbal_pitch'):
                    summary += f"\n   Gimbal Pitch: {current_meta['gimbal_pitch']} deg"
        
        # Update metadata display
        self.metadata_text.delete(1.0, tk.END)
        self.metadata_text.insert(1.0, summary)
        
        # Update map with all locations
        self.map_widget.update_locations(self.metadata_cache)

    def train_model(self):
        """Train machine learning model using current annotations with algorithm selection"""
        try:
            # Check if we have annotations
            all_annotations = self.annotation_manager.get_all_annotations()
            annotation_counts = self.annotation_manager.get_annotation_count()
            
            if not all_annotations:
                messagebox.showwarning("No Annotations", 
                                     "Please create some annotations first by drawing rectangles on images.\n\n"
                                     "Left-click + drag = Background (cyan)\n"
                                     "Right-click + drag = Features (red)")
                return
            
            # Show ML Algorithm Selection Dialog
            dialog_result = show_ml_algorithm_dialog(
                parent=self.root,
                current_algorithm=self.current_algorithm_type,
                annotation_count=annotation_counts['total']
            )
            
            if not dialog_result or not dialog_result.get('confirmed', False):
                # User cancelled the dialog
                return
            
            # Get selected algorithm and parameters
            algorithm_type = dialog_result['algorithm_type']
            algorithm_params = dialog_result['parameters']
            
            self.status_var.set(f"Initializing {algorithm_type} algorithm...")
            self.root.update()
            
            # Create new algorithm instance if different from current
            if algorithm_type != self.current_algorithm_type:
                self.classifier = MLAlgorithmFactory.create_algorithm(algorithm_type, **algorithm_params)
                self.current_algorithm_type = algorithm_type
                self.model_trained = False
                
                # Update batch processor with new classifier
                self.batch_processor = BatchProcessor(self.feature_extractor, self.classifier)
            
            # Show training progress dialog
            try:
                algorithm_info = MLAlgorithmFactory.get_algorithm_info(algorithm_type)
                algorithm_name = algorithm_info['name']
            except:
                algorithm_name = algorithm_type.replace('_', ' ').title()
            
            progress_dialog = show_ml_training_progress(self.root, algorithm_name)
            
            try:
                # Start training
                start_time = datetime.now()
                
                # Prepare training data with progress tracking
                def training_progress(progress, message):
                    progress_dialog.update_progress(progress * 0.8, f"Extracting features: {message}")
                
                progress_dialog.update_progress(10, "Preparing training data...")
                
                if hasattr(self.feature_extractor, 'prepare_training_data_parallel'):
                    X, y = self.feature_extractor.prepare_training_data_parallel(all_annotations, training_progress)
                else:
                    X, y = self.feature_extractor.prepare_training_data(all_annotations)
                feature_names = self.feature_extractor.get_feature_names()
                
                # Train the model
                progress_dialog.update_progress(85, f"Training {algorithm_name} model...")
                training_results = self.classifier.train(X, y, feature_names)
                
                progress_dialog.update_progress(95, "Saving model...")
                
                # Save the model
                os.makedirs("data/models", exist_ok=True)
                model_path = f"data/models/{algorithm_type}_classifier"
                self.classifier.save_model(model_path)
                
                progress_dialog.update_progress(100, "Training complete!")
                
                self.model_trained = True
                
                # Record processing time
                end_time = datetime.now()
                self.processing_times['train_model'] = (end_time - start_time).total_seconds()
                
                # Close progress dialog
                progress_dialog.close()
                
                # Update metadata display to show model is trained
                self.display_combined_summary()
                
                # Auto-update predictions if they're enabled
                if self.show_predictions and hasattr(self, 'image_viewer') and self.image_viewer.predictions_enabled:
                    self.update_predictions()
                
                # Show results
                accuracy = training_results['accuracy']
                cv_accuracy = training_results['cv_accuracy']
                n_samples = training_results['n_samples']
                
                result_message = f"""{algorithm_name} Training Complete!

Algorithm: {algorithm_name}
Training Results:
- Samples: {n_samples:,} pixels
- Features: {training_results['n_features']} per pixel (R,G,B,H,S,V)
- Training Accuracy: {accuracy:.1%}
- Cross-Validation: {cv_accuracy:.1%}
- Training Time: {self.processing_times['train_model']:.1f} seconds

Model saved to: data/models/{algorithm_type}_classifier.joblib

Predictions updated automatically!"""
                
                messagebox.showinfo("Training Complete", result_message)
                self.status_var.set(f"{algorithm_name} training complete")
            
            except Exception as e:
                progress_dialog.close()
                raise e
            
        except Exception as e:
            # Close progress dialog if it exists
            if 'progress_dialog' in locals():
                progress_dialog.close()
            
            error_message = f"Training failed: {str(e)}\n\nPlease check that you have valid annotations and try again."
            messagebox.showerror("Training Error", error_message)
            self.status_var.set("Training failed")

    def toggle_predictions(self):
        """Toggle prediction display"""
        if self.show_predictions:
            if not self.model_trained:
                # Try to load existing models (check multiple algorithm types)
                model_loaded = False
                
                # Check for algorithm-specific models first
                for algo_type in ['random_forest', 'knn', 'cnn']:
                    model_path = f"data/models/{algo_type}_classifier"
                    if os.path.exists(f"{model_path}.joblib"):
                        try:
                            # Create algorithm instance and load model
                            self.classifier = MLAlgorithmFactory.create_algorithm(algo_type)
                            self.classifier.load_model(model_path)
                            self.current_algorithm_type = algo_type
                            self.model_trained = True
                            model_loaded = True
                            self.status_var.set(f"{algo_type.replace('_', ' ').title()} model loaded successfully")
                            break
                        except Exception as e:
                            continue
                
                # Fallback to legacy model path for backward compatibility
                if not model_loaded:
                    legacy_model_path = "data/models/drone_classifier"
                    if os.path.exists(f"{legacy_model_path}.joblib"):
                        try:
                            # Load as Random Forest (legacy compatibility)
                            self.classifier = MLAlgorithmFactory.create_algorithm('random_forest')
                            self.classifier.load_model(legacy_model_path)
                            self.current_algorithm_type = 'random_forest'
                            self.model_trained = True
                            model_loaded = True
                            self.status_var.set("Legacy Random Forest model loaded successfully")
                        except Exception as e:
                            pass
                
                if not model_loaded:
                    messagebox.showwarning("No Model", 
                                         "No trained model found. Please train a model first.")
                    self.image_viewer.predictions_enabled = False
                    self.image_viewer.predictions_button.config(text="Show Predictions")
                    return
            
            self.update_predictions()
            self.status_var.set("Predictions enabled")
        else:
            self.image_viewer.clear_predictions()
            self.status_var.set("Predictions disabled")
    
    def update_predictions(self):
        """Update predictions for current image with filtering"""
        if not self.model_trained or self.image_viewer.original_image is None:
            return
            
        try:
            self.status_var.set("Generating predictions...")
            self.root.update()
            
            # Extract pixel features from current image
            features, coordinates = self.feature_extractor.extract_pixel_features_from_image_grid(
                self.image_viewer.original_image, max_pixels=5000
            )
            
            if len(features) > 0:
                # Get predictions and probabilities
                predictions = self.classifier.predict(features)
                probabilities = self.classifier.predict_proba(features)
                
                # Apply confidence filtering
                confidence_threshold = self.image_viewer.confidence_var.get()
                min_cluster_size = self.image_viewer.min_cluster_var.get()
                
                filtered_predictions, filtered_coords, filtered_probs = self.filter_predictions(
                    predictions, coordinates, probabilities, confidence_threshold, min_cluster_size
                )
                
                # Store current predictions for export
                self.current_predictions = (predictions, coordinates, probabilities)
                
                # Send filtered results to image viewer (filtered_probs now contains confidence scores)
                self.image_viewer.display_predictions(filtered_predictions, filtered_probs, filtered_coords)
                
                original_count = sum(predictions)
                filtered_count = len(filtered_predictions)
                prediction_text = f"Predictions: {filtered_count}/{original_count} features (filtered)"
                self.status_var.set(prediction_text)
                
                # Update filtered counter in image viewer
                self.image_viewer.update_filtered_counter(filtered_count, original_count)
            else:
                self.status_var.set("No predictions generated")
                # Reset filtered counter when no predictions
                self.image_viewer.update_filtered_counter(0, 0)
                
        except Exception as e:
            print(f"Prediction error: {e}")
            self.status_var.set("Prediction failed")
    
    def filter_predictions(self, predictions, coordinates, probabilities, confidence_threshold, min_cluster_size):
        """Filter predictions to reduce false positives"""
        import numpy as np
        from collections import defaultdict
        
        # Step 1: Confidence filtering
        feature_indices = np.where(predictions == 1)[0]
        high_conf_indices = []
        
        for idx in feature_indices:
            confidence = np.max(probabilities[idx])
            if confidence >= confidence_threshold:
                high_conf_indices.append(idx)
        
        if not high_conf_indices:
            return [], [], []
        
        # Step 2: Spatial clustering to remove isolated pixels
        filtered_coords = [coordinates[i] for i in high_conf_indices]
        filtered_probs = [probabilities[i] for i in high_conf_indices]
        
        if min_cluster_size <= 1:
            # No clustering, just extract confidence scores from filtered results
            confidence_scores = []
            for prob in filtered_probs:
                if hasattr(prob, '__len__') and len(prob) > 1:
                    confidence_scores.append(float(np.max(prob)))
                else:
                    confidence_scores.append(float(prob))
            return [1] * len(filtered_coords), filtered_coords, confidence_scores
        
        # Improved spatial clustering with larger radius
        clusters = []
        used = set()
        cluster_radius = 100  # Increased radius for better clustering
        
        for i, (x1, y1) in enumerate(filtered_coords):
            if i in used:
                continue
                
            cluster = [i]
            used.add(i)
            
            # Find nearby pixels (within cluster radius) - use iterative expansion
            to_check = [i]
            while to_check:
                current_idx = to_check.pop(0)
                current_x, current_y = filtered_coords[current_idx]
                
                for j, (x2, y2) in enumerate(filtered_coords):
                    if j in used:
                        continue
                        
                    distance = ((current_x - x2) ** 2 + (current_y - y2) ** 2) ** 0.5
                    if distance <= cluster_radius:
                        cluster.append(j)
                        used.add(j)
                        to_check.append(j)  # Check this point's neighbors too
            
            # Always add cluster, filter by size later
            clusters.append(cluster)
        
        # Extract coordinates and probabilities from valid clusters
        final_coords = []
        final_confidence_scores = []
        
        # Filter clusters by minimum size and collect results
        for cluster in clusters:
            if len(cluster) >= min_cluster_size:
                for idx in cluster:
                    final_coords.append(filtered_coords[idx])
                    # Extract confidence score (max probability) directly
                    prob_array = filtered_probs[idx]
                    if hasattr(prob_array, '__len__') and len(prob_array) > 1:
                        confidence_score = float(np.max(prob_array))
                    else:
                        confidence_score = float(prob_array)
                    final_confidence_scores.append(confidence_score)
        return [1] * len(final_coords), final_coords, final_confidence_scores
    
    
    def clear_current_annotations(self):
        """Clear annotations for current image only"""
        if not self.image_files or self.current_image_index >= len(self.image_files):
            messagebox.showwarning("No Image", "No image currently loaded.")
            return
        
        current_image_path = self.image_files[self.current_image_index]
        annotation_count = len(self.annotation_manager.get_annotations(current_image_path))
        
        if annotation_count == 0:
            messagebox.showinfo("No Annotations", "Current image has no annotations to clear.")
            return
        
        result = messagebox.askyesno(
            "Clear Current Image Annotations",
            f"Clear {annotation_count} annotations from current image?\n\n"
            f"Image: {self.image_files[self.current_image_index]}\n\n"
            f"This action cannot be undone."
        )
        
        if result:
            self.annotation_manager.clear_annotations(current_image_path)
            self.image_viewer.update_annotations()
            # Force immediate visual refresh of annotations
            self.image_viewer.canvas.delete("annotation")
            self.image_viewer.draw_annotations()
            
            # Update metadata display to reflect changes
            if hasattr(self, 'all_metadata') and self.all_metadata:
                # For multi-folder architecture, skip individual folder validation
                # The metadata display is already updated through other mechanisms
                pass
            
            self.status_var.set(f"Cleared {annotation_count} annotations from current image")
    
    def clear_all_annotations(self):
        """Clear annotations from all images"""
        total_annotations = self.annotation_manager.get_annotation_count()['total']
        
        if total_annotations == 0:
            messagebox.showinfo("No Annotations", "No annotations found to clear.")
            return
        
        result = messagebox.askyesno(
            "Clear All Annotations",
            f"Clear ALL {total_annotations} annotations from ALL images?\n\n"
            f"This will remove annotations from {len(self.annotation_manager.get_all_annotations())} images.\n\n"
            f"This action cannot be undone!"
        )
        
        if result:
            # Get count before clearing
            cleared_count = total_annotations
            cleared_images = len(self.annotation_manager.get_all_annotations())
            
            # Clear all annotations
            self.annotation_manager.clear_all_annotations()
            
            # Update current image display
            if self.image_files and self.current_image_index < len(self.image_files):
                self.image_viewer.update_annotations()
                # Force immediate visual refresh of annotations
                self.image_viewer.canvas.delete("annotation")
                self.image_viewer.draw_annotations()
            
            # Update metadata display to reflect changes
            if hasattr(self, 'all_metadata') and self.all_metadata:
                # For multi-folder architecture, skip individual folder validation
                # The metadata display is already updated through other mechanisms
                pass
            
            self.status_var.set(f"Cleared {cleared_count} annotations from {cleared_images} images")
            
            messagebox.showinfo(
                "Annotations Cleared",
                f"Successfully cleared {cleared_count} annotations from {cleared_images} images."
            )
    
    def show_export_dialog(self):
        """Show export configuration dialog and handle export"""
        # Determine default output directory
        default_output = str(Path.home() / "Desktop")
        
        # Check prerequisites
        if not hasattr(self, 'classifier') or not self.classifier.is_trained:
            # Check if we have current predictions for single image export
            if not hasattr(self, 'current_predictions') or not self.current_predictions:
                messagebox.showerror("Error", "Please train a model or run predictions first!")
                return
            else:
                # We have current predictions, so single image export is possible
                if not hasattr(self, 'current_predictions') or not self.current_predictions:
                    messagebox.showerror("Error", "No predictions available for current image!")
                    return
        
        # Show export configuration dialog
        config = show_export_dialog(self.root, default_output)
        
        if config is None:
            # User cancelled
            return
        
        # Execute export based on configuration
        try:
            # Use unified export method
            self.export_features(config)
        except Exception as e:
            messagebox.showerror("Export Error", f"Export failed: {str(e)}")
    
    def export_features(self, config):
        """Unified export method for both single image and batch export"""
        # Determine target images and export mode
        if config['current_image_only']:
            # Single image export
            if not hasattr(self, 'current_predictions') or not self.current_predictions:
                messagebox.showerror("Error", "No predictions available for current image!")
                return
            
            target_images = [self.image_files[self.current_image_index]]
            current_image_name = os.path.splitext(os.path.basename(self.image_files[self.current_image_index]))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_folder_name = f"single_export_{current_image_name}_{timestamp}"
            mode = "single"
        else:
            # Batch export
            if not hasattr(self, 'classifier') or not self.classifier.is_trained:
                messagebox.showerror("Error", "Please train a model first before batch processing!")
                return
                
            if not self.image_files:
                messagebox.showerror("Error", "No images loaded. Please load a folder first!")
                return
            
            target_images = self.image_files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_folder_name = f"batch_export_{timestamp}"
            mode = "batch"
        
        # Create export folder structure
        export_folder = os.path.join(config['output_folder'], export_folder_name)
        os.makedirs(export_folder, exist_ok=True)
        
        # Create shp subfolder
        shp_folder = os.path.join(export_folder, "shp")
        os.makedirs(shp_folder, exist_ok=True)
        
        # Set output path for shapefile in shp subfolder (without .shp extension)
        output_path = os.path.join(shp_folder, "features")
        
        # Cache filter values from main thread before starting background processing
        if hasattr(self, 'image_viewer'):
            try:
                self._cached_confidence_threshold = self.image_viewer.confidence_var.get()
                self._cached_min_cluster_size = self.image_viewer.min_cluster_var.get()
            except Exception as e:
                print(f"Warning: Could not cache filter values: {e}")
                self._cached_confidence_threshold = 0.9
                self._cached_min_cluster_size = 3
        else:
            self._cached_confidence_threshold = 0.9
            self._cached_min_cluster_size = 3
        
        # Process the export
        self._process_and_export(target_images, output_path, config, export_folder, mode)
    
    def _process_and_export(self, target_images, output_path, config, export_folder, mode):
        """Unified processing and export method for both single and batch"""
        start_time = datetime.now()
        
        # Import required modules
        try:
            import cv2
            from georef.batch_georeferencer import BatchGeoreferencer
            from export.shapefile_exporter import ShapefileExporter
        except ImportError as e:
            messagebox.showerror("Error", f"Failed to import export modules: {e}")
            return
        
        try:
            # Route to appropriate export method based on export type
            export_type = config.get('export_type', 'feature_polygons')
            
            if mode == "single":
                # Single image processing (keep on main thread - fast)
                self._process_single_image(target_images[0], output_path, config, export_folder, export_type)
            else:
                # Batch processing (move to background thread to prevent GUI freezing)
                import threading
                processing_thread = threading.Thread(
                    target=self._process_batch_images_threaded,
                    args=(target_images, output_path, config, export_folder, export_type),
                    daemon=True
                )
                processing_thread.start()
            
            # Record processing time
            end_time = datetime.now()
            self.processing_times['export'] = (end_time - start_time).total_seconds()
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Export failed: {str(e)}")
    
    def _process_single_image(self, image_path, output_path, config, export_folder, export_type):
        """Process single image export"""
        from georef.batch_georeferencer import BatchGeoreferencer
        from export.shapefile_exporter import ShapefileExporter
        
        # Initialize batch georeferencer
        batch_georeferencer = BatchGeoreferencer(self.metadata_processor)
        
        # Pre-compute parameters for current image
        current_folder = os.path.dirname(image_path)
        current_filename = os.path.basename(image_path)
        param_stats = batch_georeferencer.precompute_image_parameters(
            current_folder, [current_filename]
        )
        
        if param_stats['successful'] == 0:
            messagebox.showerror("Error", "Failed to extract georeferencing parameters from current image!")
            return
        
        # Get filtered predictions
        predictions, coordinates, probabilities = self.current_predictions
        
        # Apply current filtering settings - use cached values to avoid threading issues
        try:
            confidence_threshold = self.image_viewer.confidence_var.get()
            min_cluster_size = self.image_viewer.min_cluster_var.get()
        except Exception as e:
            print(f"Warning: Could not get filter values, using defaults: {e}")
            confidence_threshold = 0.9
            min_cluster_size = 3
        
        filtered_predictions, filtered_coordinates, filtered_probabilities = self.filter_predictions(
            predictions, coordinates, probabilities, confidence_threshold, min_cluster_size
        )
        
        if not filtered_coordinates:
            messagebox.showerror("Error", "No predictions pass the current filtering settings!")
            return
        
        # Convert to georeferenced coordinates
        if image_path in batch_georeferencer.image_parameters:
            params = batch_georeferencer.image_parameters[image_path]
            georeferenced_coords = batch_georeferencer.georeferencer.transform_coordinates_vectorized(
                params, filtered_coordinates, filtered_probabilities
            )
            
            # Add image name to each result
            for result in georeferenced_coords:
                result['image_name'] = os.path.basename(image_path)
        else:
            raise ValueError("No georeferencing parameters found for current image")
        
        # Export based on type
        shapefile_exporter = ShapefileExporter()
        
        if export_type == "feature_polygons":
            buffer_size_m = config.get('buffer_size_m', 0.5)
            export_stats = shapefile_exporter.export_feature_polygons(
                georeferenced_coords, output_path, buffer_size_m
            )
        elif export_type == "feature_points":
            export_stats = shapefile_exporter.export_feature_points(
                georeferenced_coords, output_path
            )
        elif export_type == "combined":
            # For single image combined export, create drone position and footprint
            from georef.footprint_calculator import FootprintCalculator
            
            footprint_calc = FootprintCalculator(batch_georeferencer.georeferencer)
            
            # Get drone position
            drone_positions = footprint_calc.get_drone_positions_from_images(
                [image_path], batch_georeferencer
            )
            
            # Get image footprint
            image_footprints = []
            if image_path in batch_georeferencer.image_parameters:
                params = batch_georeferencer.image_parameters[image_path]
                footprint = footprint_calc.calculate_image_footprint(image_path, params)
                if footprint:
                    image_footprints.append(footprint)
            
            export_stats = shapefile_exporter.export_combined_data(
                georeferenced_coords, drone_positions, image_footprints, output_path
            )
        else:
            # Default to feature points
            export_stats = shapefile_exporter.export_feature_points(
                georeferenced_coords, output_path
            )
        
        # Generate PDF report if requested
        if config['include_report']:
            self._generate_pdf_report(export_stats, georeferenced_coords, export_folder, "single", self.processing_times)
        
        # Show success message
        messagebox.showinfo("Export Complete", 
                          f"Single image export completed successfully!\n\n"
                          f"Export folder: {export_folder}\n"
                          f"Features detected: {len(georeferenced_coords):,}")
    
    def _process_batch_images_threaded(self, target_images, output_path, config, export_folder, export_type):
        """Thread-safe wrapper for batch processing - runs in background thread"""
        try:
            self._process_batch_images(target_images, output_path, config, export_folder, export_type)
        except Exception as e:
            # Handle errors in background thread - schedule on main thread
            error_msg = f"Batch export failed: {str(e)}"
            self.root.after(0, lambda: messagebox.showerror("Export Error", error_msg))
    
    def _process_batch_images(self, target_images, output_path, config, export_folder, export_type):
        """Process batch export for multiple images"""
        from georef.batch_georeferencer import BatchGeoreferencer
        from export.shapefile_exporter import ShapefileExporter
        import cv2
        import queue
        
        # Create progress dialog
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Batch Export")
        progress_window.geometry("600x400")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        # Center the dialog relative to parent window
        progress_window.update_idletasks()
        parent_x = self.root.winfo_x()
        parent_y = self.root.winfo_y()
        parent_width = self.root.winfo_width()
        parent_height = self.root.winfo_height()
        
        # Calculate center position
        x = parent_x + (parent_width // 2) - (600 // 2)
        y = parent_y + (parent_height // 2) - (400 // 2)
        
        # Ensure dialog doesn't go off screen
        screen_width = progress_window.winfo_screenwidth()
        screen_height = progress_window.winfo_screenheight()
        x = max(0, min(x, screen_width - 600))
        y = max(0, min(y, screen_height - 400))
        
        progress_window.geometry(f"600x400+{x}+{y}")
        
        # Prevent window from being closed during processing
        progress_window.protocol("WM_DELETE_WINDOW", lambda: None)
        
        # Progress widgets
        progress_frame = ttk.Frame(progress_window)
        progress_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        progress_label = ttk.Label(progress_frame, text="Initializing batch export...")
        progress_label.pack(pady=(0, 10))
        
        progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        # Log text area
        log_text = tk.Text(progress_frame, height=15, wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(progress_frame, orient=tk.VERTICAL, command=log_text.yview)
        log_text.configure(yscrollcommand=log_scrollbar.set)
        
        log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Thread-safe progress update system
        update_queue = queue.Queue()
        is_processing = True
        
        def process_gui_updates():
            """Process queued GUI updates on the main thread"""
            try:
                while True:
                    try:
                        update_data = update_queue.get_nowait()
                        
                        # Apply updates to GUI
                        if 'percent' in update_data:
                            progress_bar['value'] = update_data['percent']
                        if 'message' in update_data:
                            progress_label.config(text=update_data['message'])
                        if 'log_message' in update_data and update_data['log_message']:
                            log_text.insert(tk.END, update_data['log_message'] + "\n")
                            log_text.see(tk.END)
                            
                    except queue.Empty:
                        break
                        
            except Exception as e:
                print(f"Error processing GUI updates: {e}")
            
            # Schedule next update check if still processing
            if is_processing:
                self.root.after(50, process_gui_updates)
                # Keep progress window visible and on top
                try:
                    progress_window.lift()
                    progress_window.update_idletasks()
                except:
                    pass  # Window might be destroyed
        
        def update_progress(percent, message, log_message=""):
            """Thread-safe progress update function"""
            try:
                update_queue.put({
                    'percent': percent,
                    'message': message,
                    'log_message': log_message
                })
            except Exception as e:
                print(f"Error queuing progress update: {e}")
        
        # Start GUI update processor
        process_gui_updates()
        
        # Keep window visible and responsive
        progress_window.lift()
        progress_window.focus_force()
        progress_window.update_idletasks()
        
        try:
            # Initialize batch georeferencer
            update_progress(5, "Initializing batch processing...")
            batch_georeferencer = BatchGeoreferencer(self.metadata_processor)
            
            # Pre-compute georeferencing parameters
            update_progress(10, "Pre-computing georeferencing parameters...")
            folder_files = {}
            for img_path in target_images:
                folder = os.path.dirname(img_path)
                filename = os.path.basename(img_path)
                if folder not in folder_files:
                    folder_files[folder] = []
                folder_files[folder].append(filename)
            
            total_successful = 0
            for folder, files in folder_files.items():
                folder_stats = batch_georeferencer.precompute_image_parameters(folder, files)
                total_successful += folder_stats['successful']
            
            update_progress(20, f"Parameters computed: {total_successful}/{len(target_images)} images")
            
            if total_successful == 0:
                raise ValueError("No images could be georeferenced. Check GPS metadata.")
            
            # Process all images with ML using parallel processing
            update_progress(25, "Processing images with parallel ML model...")
            all_georeferenced_coords = []
            
            # Initialize parallel processor
            from processing.parallel_batch_processor import ParallelBatchProcessor
            
            # Check if classifier is trained
            if not hasattr(self.classifier, 'is_trained') or not self.classifier.is_trained:
                raise ValueError("ML model is not trained! Please train the model first:\n1. Add some annotations (red=features, cyan=background)\n2. Click 'Train Model' in the Process menu\n3. Then try batch export again")
            
            parallel_processor = ParallelBatchProcessor(self.feature_extractor, self.classifier)
            
            # Group images by folder for efficient parallel processing
            folder_groups = {}
            for img_path in target_images:
                if img_path in batch_georeferencer.image_parameters:
                    folder = os.path.dirname(img_path)
                    if folder not in folder_groups:
                        folder_groups[folder] = []
                    folder_groups[folder].append(os.path.basename(img_path))
            
            # Process each folder group in parallel
            total_processed = 0
            for folder_idx, (folder, image_files) in enumerate(folder_groups.items()):
                try:
                    folder_name = os.path.basename(folder)
                    update_progress(25 + (folder_idx / len(folder_groups)) * 50, 
                                  f"Processing folder {folder_name} ({len(image_files)} images)...")
                    
                    # Parallel ML processing for current folder
                    def folder_progress_callback(progress, status, eta):
                        overall_progress = 25 + (folder_idx / len(folder_groups)) * 50 + (progress / len(folder_groups)) * 0.5
                        update_progress(overall_progress, f"{folder_name}: {status}", f"{folder_name}: {status}")
                    
                    # Process folder in parallel
                    ml_results = parallel_processor.process_all_images_parallel(
                        folder, image_files, folder_progress_callback, max_pixels_per_image=5000
                    )
                    
                    # Process ML results and apply filtering + georeferencing
                    for img_path, result in ml_results['results'].items():
                        try:
                            if img_path not in batch_georeferencer.image_parameters:
                                continue
                            
                            # Extract coordinates and confidence scores from parallel results
                            feature_coordinates = result.get('feature_coordinates', [])
                            confidence_scores = result.get('confidence_scores', [])
                            
                            # Debug: Log what we got from parallel processing
                            update_progress(25 + (folder_idx / len(folder_groups)) * 50, "", 
                                          f"{os.path.basename(img_path)}: Found {len(feature_coordinates)} raw features")
                            
                            if not feature_coordinates:
                                update_progress(25 + (folder_idx / len(folder_groups)) * 50, "", 
                                              f"{os.path.basename(img_path)}: No features from parallel processing")
                                continue
                            
                            # Apply confidence and clustering filters (same as single image export)
                            # Get filter values safely from main thread before parallel processing
                            confidence_threshold = getattr(self, '_cached_confidence_threshold', 0.9)
                            min_cluster_size = getattr(self, '_cached_min_cluster_size', 3)
                            
                            # Filter by confidence threshold first
                            if confidence_threshold > 0:
                                before_count = len(feature_coordinates)
                                filtered_indices = [i for i, score in enumerate(confidence_scores) 
                                                  if score >= confidence_threshold]
                                feature_coordinates = [feature_coordinates[i] for i in filtered_indices]
                                confidence_scores = [confidence_scores[i] for i in filtered_indices]
                                after_count = len(feature_coordinates)
                                if before_count != after_count:
                                    update_progress(25 + (folder_idx / len(folder_groups)) * 50, "", 
                                                  f"Confidence filter: {after_count}/{before_count} features kept")
                            
                            # Apply spatial clustering
                            if min_cluster_size > 1:
                                before_count = len(feature_coordinates)
                                filtered_coordinates, filtered_confidences = self._apply_spatial_clustering(
                                    feature_coordinates, confidence_scores, min_cluster_size
                                )
                                after_count = len(filtered_coordinates)
                                if before_count != after_count:
                                    update_progress(25 + (folder_idx / len(folder_groups)) * 50, "", 
                                                  f"Clustering filter: {after_count}/{before_count} features kept")
                            else:
                                filtered_coordinates = feature_coordinates
                                filtered_confidences = confidence_scores
                            
                            if not filtered_coordinates:
                                continue
                            
                            # Georeference filtered coordinates
                            params = batch_georeferencer.image_parameters[img_path]
                            georeferenced_coords = batch_georeferencer.georeferencer.transform_coordinates_vectorized(
                                params, filtered_coordinates, filtered_confidences
                            )
                            
                            # Add image name to each result
                            for geo_result in georeferenced_coords:
                                geo_result['image_name'] = os.path.basename(img_path)
                            
                            all_georeferenced_coords.extend(georeferenced_coords)
                            total_processed += 1
                            
                            if len(georeferenced_coords) > 0:
                                update_progress(25 + (folder_idx / len(folder_groups)) * 50, "", 
                                              f"{os.path.basename(img_path)}: {len(georeferenced_coords)} features detected")
                            
                        except Exception as e:
                            update_progress(25 + (folder_idx / len(folder_groups)) * 50, "", 
                                          f"{os.path.basename(img_path)}: Error - {str(e)}")
                            continue
                    
                    # Log folder completion
                    folder_features = sum(len(result['feature_coordinates']) for result in ml_results['results'].values())
                    update_progress(25 + ((folder_idx + 1) / len(folder_groups)) * 50, "", 
                                  f"Folder {folder_name} complete: {len(ml_results['results'])} images, {folder_features} features")
                    
                except Exception as e:
                    update_progress(25 + (folder_idx / len(folder_groups)) * 50, "", 
                                  f"Folder {folder_name}: Error - {str(e)}")
                    continue
            
            # Log parallel processing performance
            perf_report = parallel_processor.get_performance_report()
            if perf_report:
                processing_summary = perf_report.get('processing_summary', {})
                update_progress(75, "", 
                              f"Parallel processing complete:")
                update_progress(75, "", 
                              f"Images processed: {processing_summary.get('total_images_processed', 0)}")
                update_progress(75, "", 
                              f"Average time per image: {processing_summary.get('avg_time_per_image', 0):.2f}s")
                update_progress(75, "", 
                              f"Parallel efficiency: {processing_summary.get('parallel_efficiency', 0):.1f}%")
            
            # Export to shapefile based on type
            update_progress(85, "Exporting to shapefile...")
            
            if not all_georeferenced_coords:
                raise ValueError("No features detected. Try lowering confidence threshold.")
            
            shapefile_exporter = ShapefileExporter()
            
            if export_type == "feature_polygons":
                buffer_size_m = config.get('buffer_size_m', 0.5)
                update_progress(85, "", f"Creating buffered polygons with {buffer_size_m}m buffer...")
                export_stats = shapefile_exporter.export_feature_polygons(
                    all_georeferenced_coords, output_path, buffer_size_m
                )
            elif export_type == "feature_points":
                update_progress(85, "", f"Exporting {len(all_georeferenced_coords)} feature points...")
                export_stats = shapefile_exporter.export_feature_points(
                    all_georeferenced_coords, output_path
                )
            elif export_type == "combined":
                # For combined export, create drone positions and footprints
                update_progress(85, "", f"Preparing combined export data...")
                from georef.footprint_calculator import FootprintCalculator
                
                footprint_calc = FootprintCalculator(batch_georeferencer.georeferencer)
                
                # Get drone positions from all processed images
                processed_images = [img_path for img_path in target_images 
                                  if img_path in batch_georeferencer.image_parameters]
                
                update_progress(87, "", f"Extracting drone positions from {len(processed_images)} images...")
                drone_positions = footprint_calc.get_drone_positions_from_images(
                    processed_images, batch_georeferencer
                )
                
                # Calculate image footprints
                update_progress(90, "", f"Calculating image footprints...")
                image_footprints = footprint_calc.calculate_batch_footprints(
                    processed_images, batch_georeferencer
                )
                
                update_progress(93, "", f"Exporting combined data: {len(all_georeferenced_coords)} features, "
                              f"{len(drone_positions)} positions, {len(image_footprints)} footprints...")
                export_stats = shapefile_exporter.export_combined_data(
                    all_georeferenced_coords, drone_positions, image_footprints, output_path
                )
            else:
                # Default to feature points
                export_stats = shapefile_exporter.export_feature_points(
                    all_georeferenced_coords, output_path
                )
            
            # Generate PDF report if requested
            if config['include_report']:
                update_progress(95, "Generating PDF report...")
                update_progress(95, "", f"PDF report requested - generating...")
                # Pass processing summary if available
                processing_summary = perf_report.get('processing_summary', {}) if 'perf_report' in locals() and perf_report else None
                
                try:
                    # Combine processing summary with our tracked times
                    combined_summary = processing_summary.copy() if processing_summary else {}
                    combined_summary.update(self.processing_times)
                    self._generate_pdf_report(export_stats, all_georeferenced_coords, export_folder, "batch", combined_summary)
                    update_progress(95, "", f"PDF report generation completed")
                except Exception as pdf_error:
                    update_progress(95, "", f"PDF report generation failed: {str(pdf_error)}")
                    print(f"PDF generation error: {pdf_error}")
            else:
                update_progress(95, "", f"PDF report not requested (checkbox unchecked)")
            
            update_progress(100, "Batch export complete")
            update_progress(100, "", f"EXPORT SUMMARY:")
            update_progress(100, "", f"Total features detected: {len(all_georeferenced_coords):,}")
            update_progress(100, "", f"Export folder: {export_folder}")
            
            # Stop the GUI update processor
            is_processing = False
            
            # Re-enable window close button
            progress_window.protocol("WM_DELETE_WINDOW", progress_window.destroy)
            
            # Add close button
            def close_progress():
                nonlocal is_processing
                is_processing = False
                progress_window.destroy()
            
            close_button = ttk.Button(progress_frame, text="Close", command=close_progress)
            close_button.pack(pady=(10, 0))
            
            # Show success message (schedule on main thread)
            success_msg = (f"Batch export completed successfully!\n\n"
                          f"Features detected: {len(all_georeferenced_coords):,}\n"
                          f"Export folder: {export_folder}")
            self.root.after(0, lambda: messagebox.showinfo("Export Complete", success_msg))
            
        except Exception as e:
            # Stop the GUI update processor on error
            is_processing = False
            # Re-enable window close button
            if 'progress_window' in locals():
                progress_window.protocol("WM_DELETE_WINDOW", progress_window.destroy)
            # Schedule error dialog on main thread
            error_msg = f"Batch export failed: {str(e)}"
            self.root.after(0, lambda: messagebox.showerror("Export Error", error_msg))
            if 'progress_window' in locals():
                self.root.after(0, progress_window.destroy)
    
    def _generate_pdf_report(self, export_stats, georeferenced_coords, output_dir, mode, processing_summary=None):
        """Generate PDF report"""
        try:
            from export.compact_pdf_generator import CompactPDFGenerator
            pdf_generator = CompactPDFGenerator()
            
            # Ensure processing_summary includes our tracked times
            if processing_summary is None:
                processing_summary = self.processing_times.copy()
            elif isinstance(processing_summary, dict):
                processing_summary.update(self.processing_times)
            
            result = pdf_generator.generate_report(
                export_stats=export_stats,
                georeferenced_coords=georeferenced_coords,
                output_dir=output_dir,
                mode=mode,
                processing_summary=processing_summary,
                image_files=self.image_files,
                metadata_cache=self.metadata_cache
            )
            if result['success']:
                print(f"PDF report generated: {result['report_path']} ({result.get('file_size_mb', 'unknown')} MB)")
                return result['report_path']
            else:
                print(f"❌ PDF generation failed: {result.get('error', 'Unknown error')}")
                return None
        except Exception as e:
            print(f"❌ PDF generation exception: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _apply_spatial_clustering(self, coordinates, confidence_scores, min_cluster_size):
        """Apply spatial clustering to filter isolated detections"""
        import numpy as np
        
        if min_cluster_size <= 1 or len(coordinates) == 0:
            return coordinates, confidence_scores
        
        # Spatial clustering with optimized radius
        clusters = []
        used = set()
        cluster_radius = 100  # pixels
        
        for i, (x1, y1) in enumerate(coordinates):
            if i in used:
                continue
                
            cluster = [i]
            used.add(i)
            
            # Find nearby pixels using iterative expansion
            to_check = [i]
            while to_check:
                current_idx = to_check.pop(0)
                current_x, current_y = coordinates[current_idx]
                
                for j, (x2, y2) in enumerate(coordinates):
                    if j in used:
                        continue
                        
                    distance = ((current_x - x2) ** 2 + (current_y - y2) ** 2) ** 0.5
                    if distance <= cluster_radius:
                        cluster.append(j)
                        used.add(j)
                        to_check.append(j)
            
            clusters.append(cluster)
        
        # Extract coordinates from valid clusters
        filtered_coords = []
        filtered_confidences = []
        
        for cluster in clusters:
            if len(cluster) >= min_cluster_size:
                for idx in cluster:
                    filtered_coords.append(coordinates[idx])
                    filtered_confidences.append(confidence_scores[idx])
        
        return filtered_coords, filtered_confidences

    def show_user_guide(self):
        """Show User Guide documentation"""
        show_user_guide(self.root)
    
    def show_readme(self):
        """Show README documentation"""
        show_readme(self.root)
    
    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo("About", "Image2Shape v2.0 - Multi-algorithm ML for drone image analysis!")
        
    def run(self):
        """Start the application"""
        self.root.mainloop()
