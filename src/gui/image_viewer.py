#!/usr/bin/env python3
"""
Image viewer with annotation capabilities
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

class ImageViewer:
    def __init__(self, parent, annotation_manager):
        self.parent = parent
        self.annotation_manager = annotation_manager
        
        # Image data
        self.original_image = None
        self.display_image = None
        self.photo_image = None
        self.image_path = None
        
        # Cache support
        self.display_image_cache = {}
        self.current_display_scale = 1.0
        self.display_to_original_scale = 1.0
        self.image_loaded = False
        
        # Display properties
        self.scale_factor = 0.33
        self.zoom_level = 1.0
        
        # Annotation state
        self.annotation_mode = "background"
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_rect = None
        
        # Prediction display
        self.predictions_enabled = True  # Default to True
        self.prediction_overlays = []
        
        self.setup_ui()
        self.setup_bindings()
        
    def setup_ui(self):
        """Set up the image viewer UI"""
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.create_toolbar(main_frame)
        
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='gray')
        
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind canvas resize event
        self.canvas.bind('<Configure>', self.on_canvas_resize)
        
    def create_toolbar(self, parent):
        """Create two-row annotation and control toolbar"""
        # Main toolbar container
        toolbar_container = ttk.Frame(parent)
        toolbar_container.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # First row: Annotation mode buttons and clear buttons
        toolbar_row1 = ttk.Frame(toolbar_container)
        toolbar_row1.pack(side=tk.TOP, fill=tk.X, pady=(0, 3))
        
        # Annotation mode buttons
        button_frame = ttk.Frame(toolbar_row1)
        button_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(button_frame, text="Annotation Mode:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.bg_button = tk.Button(button_frame, text="Background", 
                                  bg="cyan", fg="black", relief="sunken",
                                  command=self.set_background_mode, width=12)
        self.bg_button.pack(side=tk.LEFT, padx=2)
        
        self.feature_button = tk.Button(button_frame, text="Feature", 
                                       bg="lightgray", fg="black", relief="raised",
                                       command=self.set_feature_mode, width=12)
        self.feature_button.pack(side=tk.LEFT, padx=2)
        
        # Add separator
        ttk.Separator(toolbar_row1, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Clear buttons
        ttk.Button(toolbar_row1, text="Clear Current", command=self.clear_current_annotations).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar_row1, text="Clear All", command=self.clear_all_annotations).pack(side=tk.LEFT, padx=5)
        
        # Second row: Prediction controls and counters
        toolbar_row2 = ttk.Frame(toolbar_container)
        toolbar_row2.pack(side=tk.TOP, fill=tk.X)
        
        # Prediction controls
        self.predictions_enabled = True  # Default to True
        self.predictions_button = ttk.Button(
            toolbar_row2, text="Hide Predictions", 
            command=self.toggle_predictions_button
        )
        self.predictions_button.pack(side=tk.LEFT, padx=5)
        
        # Confidence threshold control
        ttk.Label(toolbar_row2, text="Confidence:").pack(side=tk.LEFT, padx=(10, 2))
        self.confidence_var = tk.DoubleVar(value=0.9)
        self.confidence_scale = tk.Scale(
            toolbar_row2, from_=0.5, to=0.95, resolution=0.05,
            variable=self.confidence_var, orient=tk.HORIZONTAL,
            length=80  # Smaller to fit in toolbar
        )
        self.confidence_scale.pack(side=tk.LEFT, padx=2)
        
        # Minimum cluster size control
        ttk.Label(toolbar_row2, text="Min Size:").pack(side=tk.LEFT, padx=(5, 2))
        self.min_cluster_var = tk.IntVar(value=3)
        self.min_cluster_scale = tk.Scale(
            toolbar_row2, from_=1, to=10, resolution=1,
            variable=self.min_cluster_var, orient=tk.HORIZONTAL,
            length=60  # Smaller to fit in toolbar
        )
        self.min_cluster_scale.pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar_row2, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Counters section
        self.image_counter_label = ttk.Label(toolbar_row2, text="Image: 0/0")
        self.image_counter_label.pack(side=tk.LEFT, padx=5)
        
        self.filtered_counter_label = ttk.Label(toolbar_row2, text="Predictions: 0/0")
        self.filtered_counter_label.pack(side=tk.LEFT, padx=5)
        
        # Set up automatic updates
        self.confidence_var.trace('w', self.on_filter_change)
        self.min_cluster_var.trace('w', self.on_filter_change)
        self._update_pending = False
        
    def setup_prediction_controls(self):
        """Set up prediction controls in image viewer"""
        # Prediction controls are now integrated into the main toolbar
        # This method is kept for compatibility but does nothing
        pass
        
    def set_background_mode(self):
        """Set annotation mode to background"""
        self.annotation_mode = "background"
        # Update button appearances
        self.bg_button.config(relief="sunken", bg="cyan", fg="black")
        self.feature_button.config(relief="raised", bg="lightgray", fg="black")
        
    def set_feature_mode(self):
        """Set annotation mode to feature"""
        self.annotation_mode = "feature"
        # Update button appearances
        self.bg_button.config(relief="raised", bg="lightgray", fg="black")
        self.feature_button.config(relief="sunken", bg="red", fg="white")
        
    def setup_bindings(self):
        """Set up mouse and keyboard bindings"""
        # Mouse bindings - both left and right click now use the same handlers
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        self.canvas.bind("<Button-3>", self.on_mouse_down)
        self.canvas.bind("<B3-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_mouse_up)
        
        # Focus for canvas (but don't interfere with annotation)
        self.canvas.focus_set()
        
    # Removed zoom functionality
        
    def set_image_cache(self, image_cache):
        """Set the pre-generated display image cache"""
        self.display_image_cache = image_cache
    
    def load_image(self, image_path):
        """Load and display an image using cached version if available"""
        self.image_path = image_path
        
        if self.load_image_fast(image_path):
            return
        
        self.load_image_original(image_path)
    
    def load_image_fast(self, image_path):
        """Load image using cached 1/3 scale version for instant display"""
        if image_path in self.display_image_cache:
            cache_path = self.display_image_cache[image_path]
            if os.path.exists(cache_path):
                try:
                    img = cv2.imread(cache_path)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        self.display_cached_image(img_rgb, image_path)
                        return True
                except Exception as e:
                    print(f"Error loading cached image {cache_path}: {e}")
        
        return False
    
    def display_cached_image(self, img, original_path):
        """Display pre-scaled cached image with proper resizing"""
        # Load original image immediately for annotations (but don't display it)
        try:
            full_res_image = cv2.imread(original_path)
            if full_res_image is not None:
                self.original_image = cv2.cvtColor(full_res_image, cv2.COLOR_BGR2RGB)
            else:
                self.original_image = img  # Fallback to cached
        except:
            self.original_image = img  # Fallback to cached
        
        # Ensure canvas is properly sized before calculating display parameters
        self.canvas.update_idletasks()
        self.parent.update_idletasks()  # Update parent too
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 800
            canvas_height = 600
        
        # Calculate zoom to fit cached image in canvas
        img_height, img_width = img.shape[:2]
        zoom_x = canvas_width / img_width
        zoom_y = canvas_height / img_height
        self.zoom_level = min(zoom_x, zoom_y) * 1.0
        
        # Actually resize the cached image for display
        display_width = int(img_width * self.zoom_level)
        display_height = int(img_height * self.zoom_level)
        
        if display_width > 0 and display_height > 0:
            self.display_image = cv2.resize(img, (display_width, display_height))
        else:
            self.display_image = img
        
        # Set scale factors for coordinate conversion - CRITICAL for annotation accuracy
        if self.original_image is not None:
            orig_height, orig_width = self.original_image.shape[:2]
            disp_height, disp_width = self.display_image.shape[:2]
            self.display_to_original_scale = orig_width / disp_width
        else:
            self.display_to_original_scale = 3.0
        
        # Display the resized image
        pil_image = Image.fromarray(self.display_image)
        self.photo_image = ImageTk.PhotoImage(pil_image)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        self.image_loaded = True
        self.draw_annotations()
    
    # Removed unused methods - functionality integrated into display_cached_image
    
    def load_image_original(self, image_path):
        """Load and display an image using original method"""
        try:
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                raise ValueError("Could not load image")
                
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            
            self.current_display_scale = 1.0
            self.display_to_original_scale = 1.0
            self.image_loaded = True
            
            # Calculate correct size immediately for smooth loading
            self.fit_to_window_immediate()
            self.update_display()
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            self.image_loaded = False
            
    def update_display(self):
        """Update the display with current zoom and scale"""
        if self.original_image is None:
            return
            
        height, width = self.original_image.shape[:2]
        display_width = int(width * self.scale_factor * self.zoom_level)
        display_height = int(height * self.scale_factor * self.zoom_level)
        
        self.display_image = cv2.resize(self.original_image, (display_width, display_height))
        
        pil_image = Image.fromarray(self.display_image)
        self.photo_image = ImageTk.PhotoImage(pil_image)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
        
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        self.draw_annotations()
        self.draw_predictions()
        
    def draw_annotations(self):
        """Draw all annotations on the canvas"""
        if not self.image_path:
            return
            
        annotations = self.annotation_manager.get_annotations(self.image_path)
        
        for annotation in annotations:
            x1, y1, x2, y2 = annotation['bbox']
            annotation_type = annotation['type']
            
            # Convert from original coordinates to display coordinates
            display_x1 = int(x1 / self.display_to_original_scale)
            display_y1 = int(y1 / self.display_to_original_scale)
            display_x2 = int(x2 / self.display_to_original_scale)
            display_y2 = int(y2 / self.display_to_original_scale)
            
            color = "red" if annotation_type == "feature" else "cyan"
            
            self.canvas.create_rectangle(display_x1, display_y1, display_x2, display_y2,
                                       outline=color, width=2, tags="annotation")
            
    def update_annotations(self):
        """Update annotations display"""
        self.draw_annotations()
        
    def on_mouse_down(self, event):
        """Handle mouse button down - uses current annotation mode"""
        if not self.image_loaded:
            return
            
        self.drawing = True
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        
    def on_mouse_drag(self, event):
        """Handle mouse drag - uses current annotation mode"""
        if not self.drawing or self.original_image is None:
            return
            
        current_x = self.canvas.canvasx(event.x)
        current_y = self.canvas.canvasy(event.y)
        
        if self.current_rect:
            self.canvas.delete(self.current_rect)
            
        # Use color based on current annotation mode
        color = "cyan" if self.annotation_mode == "background" else "red"
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, current_x, current_y,
            outline=color, width=2, tags="temp"
        )
        
    def on_mouse_up(self, event):
        """Handle mouse button up - uses current annotation mode"""
        self._finish_annotation(event, self.annotation_mode)
        
    def _finish_annotation(self, event, annotation_type):
        """Common logic for finishing annotation"""
        if not self.drawing:
            return
        
        # Original image should already be loaded in display_cached_image()
        if self.original_image is None:
            return
            
        self.drawing = False
        
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        
        if self.current_rect:
            self.canvas.delete(self.current_rect)
            self.current_rect = None
            
        x1 = min(self.start_x, end_x)
        y1 = min(self.start_y, end_y)
        x2 = max(self.start_x, end_x)
        y2 = max(self.start_y, end_y)
        
        if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
            return
            
        # Use the display_to_original_scale for accurate coordinate conversion
        orig_x1 = int(x1 * self.display_to_original_scale)
        orig_y1 = int(y1 * self.display_to_original_scale)
        orig_x2 = int(x2 * self.display_to_original_scale)
        orig_y2 = int(y2 * self.display_to_original_scale)
        
        height, width = self.original_image.shape[:2]
        orig_x1 = max(0, min(orig_x1, width))
        orig_y1 = max(0, min(orig_y1, height))
        orig_x2 = max(0, min(orig_x2, width))
        orig_y2 = max(0, min(orig_y2, height))
        
        self.annotation_manager.add_annotation(
            self.image_path, 
            (orig_x1, orig_y1, orig_x2, orig_y2), 
            annotation_type
        )
        
        self.draw_annotations()
        
        # Update metadata display when annotation is added
        if hasattr(self, 'main_window') and self.main_window:
            self.main_window.display_combined_summary()
        
    # Removed all zoom functionality
        
    def fit_to_window_immediate(self):
        """Fit image to window immediately without delay"""
        if self.original_image is None:
            return
            
        # Force canvas update to get current dimensions
        self.canvas.update_idletasks()
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Use reasonable defaults if canvas not ready
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 800  # Default width
            canvas_height = 600  # Default height
            
        height, width = self.original_image.shape[:2]
        display_width = width * self.scale_factor
        display_height = height * self.scale_factor
        
        zoom_x = canvas_width / display_width
        zoom_y = canvas_height / display_height
        
        self.zoom_level = min(zoom_x, zoom_y) * 1.0  # 100% space usage
        
        # Don't call update_display() here to avoid double-rendering
        
    def fit_to_window(self):
        """Fit image to window (legacy method for compatibility)"""
        self.fit_to_window_immediate()
        self.update_display()
        
    def display_predictions(self, predictions, probabilities, coordinates):
        """Display ML predictions as overlay"""
        self.predictions_enabled = True
        
        # Clear existing predictions first
        self.clear_predictions()
        
        self.prediction_overlays = []
        
        for pred, prob, coord in zip(predictions, probabilities, coordinates):
            self.prediction_overlays.append({
                'prediction': pred,
                'probability': prob,
                'coordinate': coord
            })
        
        self.draw_predictions()
    
    def draw_predictions(self):
        """Draw prediction overlays on the canvas"""
        if not self.predictions_enabled or not self.prediction_overlays:
            return
        
        self.canvas.delete("prediction")
        
        scale = self.scale_factor * self.zoom_level
        
        for overlay in self.prediction_overlays:
            pred = overlay['prediction']
            prob = overlay['probability']
            coord = overlay['coordinate']
            
            if pred == 1:
                x, y = coord
                
                display_x = int(x * scale)
                display_y = int(y * scale)
                
                # Handle both confidence scores (float) and probability arrays
                if hasattr(prob, '__len__') and len(prob) > 1:
                    # This is a probability array [background_prob, feature_prob]
                    confidence = max(prob)
                else:
                    # This is already a confidence score (float)
                    confidence = float(prob)
                
                size = int(20 * confidence)
                
                if confidence > 0.8:
                    color = "#FF0000"
                elif confidence > 0.6:
                    color = "#FF6600"
                else:
                    color = "#FFAA00"
                
                self.canvas.create_oval(
                    display_x - size//2, display_y - size//2,
                    display_x + size//2, display_y + size//2,
                    fill=color, outline="white", width=1,
                    tags="prediction"
                )
    
    def clear_predictions(self):
        """Clear all prediction overlays"""
        self.prediction_overlays = []
        self.canvas.delete("prediction")
    
    def set_main_window(self, main_window):
        """Set reference to main window for button callbacks"""
        self.main_window = main_window
    
    def clear_current_annotations(self):
        """Clear annotations for current image - delegate to main window"""
        if hasattr(self, 'main_window') and self.main_window:
            self.main_window.clear_current_annotations()
    
    def clear_all_annotations(self):
        """Clear all annotations - delegate to main window"""
        if hasattr(self, 'main_window') and self.main_window:
            self.main_window.clear_all_annotations()
    
    def update_image_counter(self, current_index, total_images):
        """Update the image counter display"""
        if hasattr(self, 'image_counter_label'):
            self.image_counter_label.config(text=f"Image: {current_index + 1}/{total_images}")
    
    def update_filtered_counter(self, filtered_count, total_count=None):
        """Update the filtered predictions counter display"""
        if hasattr(self, 'filtered_counter_label'):
            if total_count is not None:
                self.filtered_counter_label.config(text=f"Predictions: {filtered_count}/{total_count}")
            else:
                self.filtered_counter_label.config(text=f"Predictions: {filtered_count}/0")
    
    def toggle_predictions_button(self):
        """Toggle prediction display button"""
        self.predictions_enabled = not self.predictions_enabled
        
        if self.predictions_enabled:
            self.predictions_button.config(text="Hide Predictions")
            if hasattr(self, 'main_window'):
                self.main_window.show_predictions = True
                self.main_window.toggle_predictions()
        else:
            self.predictions_button.config(text="Show Predictions")
            if hasattr(self, 'main_window'):
                self.main_window.show_predictions = False
            self.clear_predictions()
    
    def toggle_predictions(self):
        """Toggle prediction display (for compatibility)"""
        if hasattr(self, 'main_window'):
            self.main_window.show_predictions = self.predictions_enabled
            
            if self.predictions_enabled:
                if hasattr(self, 'main_window'):
                    self.main_window.toggle_predictions()
            else:
                self.clear_predictions()
    
    def on_filter_change(self, *args):
        """Handle filter parameter changes with debouncing"""
        if not self._update_pending and self.predictions_enabled and hasattr(self, 'main_window'):
            self._update_pending = True
            # Debounce updates - wait 200ms before updating
            self.canvas.after(200, self.delayed_update_predictions)
    
    def delayed_update_predictions(self):
        """Update predictions after debounce delay"""
        self._update_pending = False
        if self.predictions_enabled and hasattr(self, 'main_window'):
            self.main_window.update_predictions()
    
    def on_canvas_resize(self, event):
        """Handle canvas resize events"""
        if not self.image_loaded or self.original_image is None:
            return
        
        # Only handle resize events for the canvas itself, not child widgets
        if event.widget != self.canvas:
            return
        
        # Recalculate zoom and display parameters
        if hasattr(self, 'display_image_cache') and self.image_path in self.display_image_cache:
            # For cached images, recalculate the display
            cache_path = self.display_image_cache[self.image_path]
            if os.path.exists(cache_path):
                try:
                    img = cv2.imread(cache_path)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        self.display_cached_image(img_rgb, self.image_path)
                except Exception as e:
                    print(f"Error reloading cached image on resize: {e}")
        else:
            # For original images, update display
            self.fit_to_window_immediate()
            self.update_display()