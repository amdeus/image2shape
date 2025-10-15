#!/usr/bin/env python3
"""
Simple plot widget for displaying drone image positions
Shows clean XY plot of lat/lon coordinates with flight path
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import os

class MapWidget:
    """
    Simple matplotlib-based widget showing drone image positions
    
    Features:
    - Clean XY plot of lat/lon coordinates
    - Color-coded dots for RTK quality (green/orange/red)
    - Flight path line connecting positions
    - Red cross marker for currently selected image
    - Minimal design with no labels or clutter
    """
    def __init__(self, parent):
        self.parent = parent
        self.image_locations = []
        self.current_image_index = 0
        self.figure = None
        self.canvas = None
        self.ax = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the simple plot widget UI"""
        self.map_frame = self.parent
        
        # Create matplotlib figure with no border
        self.figure = Figure(figsize=(6, 4), dpi=80, facecolor='white')
        self.figure.patch.set_linewidth(0)  # Remove border
        self.ax = self.figure.add_subplot(111)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, self.map_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Set up click event handling
        self.canvas.mpl_connect('button_press_event', self.on_map_click)
        self.click_callback = None  # Will be set by main window
        
        # Initial empty plot
        self.create_empty_plot()
        
    def create_empty_plot(self):
        """Create empty plot"""
        self.ax.clear()
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.text(0.5, 0.5, 'No GPS data', 
                    transform=self.ax.transAxes, ha='center', va='center',
                    fontsize=10, alpha=0.5)
        self.canvas.draw()
        
    def update_locations(self, image_metadata):
        """Update map with image locations"""
        self.image_locations = []
        
        for image_path, metadata in image_metadata.items():
            # Check for GPS data using the correct keys from metadata processor
            if metadata.get('latitude') and metadata.get('longitude'):
                try:
                    lat = self._parse_coordinate(metadata.get('latitude'))
                    lon = self._parse_coordinate(metadata.get('longitude'))
                    
                    if lat is not None and lon is not None:
                        location_info = {
                            'image_path': image_path,
                            'image_name': os.path.basename(image_path),
                            'latitude': lat,
                            'longitude': lon,
                            'altitude': metadata.get('altitude'),
                            'altitude_above_ground': metadata.get('altitude_above_ground'),
                            'rtk_quality': self._assess_rtk_quality(metadata),
                            'datetime': metadata.get('datetime_original', 'Unknown')
                        }
                        self.image_locations.append(location_info)
                except Exception as e:
                    print(f"Error parsing coordinates for {image_path}: {e}")
                    
        if self.image_locations:
            self.generate_plot()
        else:
            print("No valid GPS locations found for plot")
            self.create_empty_plot()
            
    def _parse_coordinate(self, coord_str):
        """Parse coordinate string to decimal degrees"""
        if coord_str is None:
            return None
            
        try:
            # With -n flag, coordinates are now decimal numbers
            if isinstance(coord_str, (int, float)):
                return float(coord_str)
            
            coord_str = str(coord_str).strip()
            
            # Try direct float conversion first (expected with -n flag)
            try:
                return float(coord_str)
            except ValueError:
                pass
            
            # Fallback: Handle DMS format: "53 deg 49' 47.30" N" (for backward compatibility)
            if 'deg' in coord_str:
                # Remove deg, ', " and extra spaces
                cleaned = coord_str.replace('deg', '').replace("'", '').replace('"', '').strip()
                
                # Split by spaces and filter out empty strings
                parts = [p for p in cleaned.split() if p]
                
                if len(parts) >= 3:
                    degrees = float(parts[0])
                    minutes = float(parts[1])
                    seconds = float(parts[2])
                    
                    decimal = degrees + minutes/60 + seconds/3600
                    
                    # Check for hemisphere (N/S/E/W)
                    hemisphere = None
                    if len(parts) > 3:
                        hemisphere = parts[3].upper()
                    elif coord_str.endswith(('N', 'S', 'E', 'W')):
                        hemisphere = coord_str[-1].upper()
                    
                    if hemisphere in ['S', 'W']:
                        decimal = -decimal
                    
                    return decimal
            
            # Try direct float conversion
            return float(coord_str)
            
        except Exception as e:
            return None
            
    def _assess_rtk_quality(self, metadata):
        """Assess RTK quality based on standard deviations"""
        # Check for RTK status using correct metadata keys
        rtk_status = metadata.get('rtk_status') or metadata.get('gps_status')
        if rtk_status != 'RTK' and not metadata.get('has_rtk'):
            return 'No RTK'
            
        lat_std = metadata.get('rtk_std_lat')
        lon_std = metadata.get('rtk_std_lon')
        
        if lat_std is not None and lon_std is not None:
            max_std = max(float(lat_std), float(lon_std))
            if max_std < 0.02:
                return 'High (< 2cm)'
            elif max_std < 0.05:
                return 'Fair (2-5cm)'
            else:
                return 'Poor (> 5cm)'
        
        return 'RTK Available'
        
    def generate_plot(self):
        """Generate simple XY plot with image locations"""
        if not self.image_locations:
            return
            
        
        # Clear previous plot
        self.ax.clear()
        
        # Extract coordinates
        lats = [loc['latitude'] for loc in self.image_locations]
        lons = [loc['longitude'] for loc in self.image_locations]
        
        # Create colors based on RTK quality
        colors = []
        for loc in self.image_locations:
            if 'High' in loc['rtk_quality']:
                colors.append('green')
            elif 'Fair' in loc['rtk_quality']:
                colors.append('orange')
            elif 'Poor' in loc['rtk_quality']:
                colors.append('red')
            else:
                colors.append('blue')
        
        # Plot points (make them clickable)
        scatter = self.ax.scatter(lons, lats, c=colors, s=50, alpha=0.8, picker=True)
        
        # Store coordinates for click detection
        self.lats = lats
        self.lons = lons
        
        # Sort locations by datetime for proper flight path
        sorted_locations = sorted(self.image_locations, key=lambda x: x.get('datetime', ''))
        if len(sorted_locations) > 1:
            sorted_lons = [loc['longitude'] for loc in sorted_locations]
            sorted_lats = [loc['latitude'] for loc in sorted_locations]
            self.ax.plot(sorted_lons, sorted_lats, 'b-', alpha=0.5, linewidth=1, label='Flight Path')
        
        # Highlight current image with red cross (find by image path)
        if (hasattr(self, 'current_image_path') and self.current_image_path and 
            0 <= self.current_image_index < len(self.image_locations)):
            # Find the location that matches the current image path
            current_loc = None
            for loc in self.image_locations:
                if loc['image_path'] == self.current_image_path:
                    current_loc = loc
                    break
            
            if current_loc:
                self.ax.scatter([current_loc['longitude']], [current_loc['latitude']], 
                              c='red', s=80, marker='x', linewidth=2)
        
        # Remove all labels and ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Add grid
        self.ax.grid(True, alpha=0.3)
        
        # Set equal aspect ratio for proper geographic representation
        self.ax.set_aspect('equal', adjustable='box')
        
        # Adjust layout and refresh
        self.figure.tight_layout()
        self.canvas.draw()
        
        
    def highlight_current_image(self, image_index, image_path=None):
        """Highlight the current image on the plot"""
        self.current_image_index = image_index
        self.current_image_path = image_path
        if self.image_locations:
            self.generate_plot()
                
    def set_click_callback(self, callback):
        """Set callback function for when map dots are clicked"""
        self.click_callback = callback
    
    def on_map_click(self, event):
        """Handle clicks on the map to select images"""
        if event.inaxes != self.ax or not hasattr(self, 'image_locations'):
            return
        
        if not self.image_locations or not hasattr(self, 'lats') or not hasattr(self, 'lons'):
            return
        
        # Find the closest image to the click point
        click_lon = event.xdata
        click_lat = event.ydata
        
        if click_lon is None or click_lat is None:
            return
        
        # Calculate distances to all image locations
        min_distance = float('inf')
        closest_index = -1
        
        for i, (lat, lon) in enumerate(zip(self.lats, self.lons)):
            # Simple Euclidean distance (good enough for small areas)
            distance = ((click_lon - lon) ** 2 + (click_lat - lat) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        
        # Only trigger if click is reasonably close to a dot
        # Convert distance threshold based on current axis limits
        x_range = self.ax.get_xlim()[1] - self.ax.get_xlim()[0]
        y_range = self.ax.get_ylim()[1] - self.ax.get_ylim()[0]
        threshold = min(x_range, y_range) * 0.05  # 5% of the smaller axis range
        
        if min_distance < threshold and closest_index >= 0 and self.click_callback:
            # Get the original image index from the location data
            image_path = self.image_locations[closest_index]['image_path']
            self.click_callback(image_path)
    
    def cleanup(self):
        """Clean up resources"""
        if self.figure:
            plt.close(self.figure)