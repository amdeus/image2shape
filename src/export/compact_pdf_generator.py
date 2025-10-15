#!/usr/bin/env python3
"""
Compact PDF Report Generator for Image2Shape

This module generates minimal 2-page PDF reports with coordinate plots
and essential technical information.

Author: Image2Shape Development Team
Version: 2.0 - Compact Design
"""

import os
import tempfile
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import black, blue, red
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.enums import TA_LEFT


class CompactPDFGenerator:
    """Generate compact 2-page PDF reports for Image2Shape exports"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CompactHeader',
            parent=self.styles['Heading1'],
            fontSize=14,
            spaceAfter=6,
            textColor=black
        ))
        
        self.styles.add(ParagraphStyle(
            name='CompactSection',
            parent=self.styles['Heading2'],
            fontSize=12,
            spaceAfter=4,
            spaceBefore=8,
            textColor=black
        ))
    
    def generate_report(self, export_stats, georeferenced_coords, output_dir, mode, 
                       processing_summary=None, image_files=None, metadata_cache=None):
        """
        Generate compact 2-page PDF report
        
        Args:
            export_stats: Dictionary with export statistics
            georeferenced_coords: List of georeferenced detection results
            output_dir: Output directory for the report
            mode: Export mode ('single' or 'batch')
            processing_summary: Optional processing performance data
            image_files: Optional list of processed image files
            metadata_cache: Optional metadata cache for drone positions
        """
        try:
            # Create report filename
            report_filename = "Image2Shape_Report.pdf"
            report_path = os.path.join(output_dir, report_filename)
            
            # Create PDF document with minimal margins
            doc = SimpleDocTemplate(
                report_path,
                pagesize=A4,
                rightMargin=36,
                leftMargin=36,
                topMargin=36,
                bottomMargin=36
            )
            
            # Build compact report content
            story = []
            
            # Page 1: Overview & Coordinates
            story.extend(self._create_overview_page(export_stats, georeferenced_coords, mode, processing_summary, metadata_cache))
            story.append(PageBreak())
            
            # Page 2: Technical Details
            story.extend(self._create_technical_page(export_stats, mode, processing_summary, georeferenced_coords))
            
            # Build PDF
            doc.build(story)
            
            return {
                'success': True,
                'report_path': report_path,
                'file_size_mb': round(os.path.getsize(report_path) / (1024 * 1024), 2)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'report_path': None
            }
    
    def _create_overview_page(self, export_stats, georeferenced_coords, mode, processing_summary, metadata_cache=None):
        """Create compact overview page with coordinates plots"""
        story = []
        
        # Header with title and date (using table for layout instead of float)
        from reportlab.platypus import Table, TableStyle
        from reportlab.lib import colors
        
        header_data = [['Image2Shape Export Report', datetime.now().strftime('%Y-%m-%d %H:%M')]]
        header_table = Table(header_data, colWidths=[4*inch, 2*inch])
        header_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (0, 0), 14),
            ('FONTNAME', (1, 0), (1, 0), 'Helvetica'),
            ('FONTSIZE', (1, 0), (1, 0), 12),
            ('ALIGN', (0, 0), (0, 0), 'LEFT'),
            ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Summary section
        story.append(Paragraph("📊 SUMMARY", self.styles['CompactSection']))
        
        # Calculate summary stats
        total_features = export_stats.get('total_points', 0)
        confidence_range = self._get_confidence_range(georeferenced_coords)
        gps_quality = self._get_gps_quality_summary(georeferenced_coords)
        
        summary_text = f"""• Features Detected: {total_features:,}<br/>• Processing Mode: {mode.title()}<br/>• Confidence Range: {confidence_range}<br/>• GPS Quality: {gps_quality}"""
        story.append(Paragraph(summary_text, self.styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Coordinates section
        story.append(Paragraph("📍 COORDINATES", self.styles['CompactSection']))
        story.append(Spacer(1, 0.1*inch))
        
        # Generate stacked coordinate plots (larger size, centered)
        plots_path = self._generate_coordinate_plots(georeferenced_coords, processing_summary, metadata_cache)
        if plots_path and os.path.exists(plots_path):
            # Center the image by using a table with proper aspect ratio
            from reportlab.platypus import Table, TableStyle
            img = Image(plots_path, width=6.5*inch, height=6.5*inch)  # Square aspect ratio
            img_table = Table([[img]], colWidths=[6.5*inch])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            story.append(img_table)
            story.append(Spacer(1, 0.1*inch))
        
        return story
    
    def _create_technical_page(self, export_stats, mode, processing_summary, georeferenced_coords):
        """Create technical details page"""
        story = []
        
        story.append(Paragraph("🔧 TECHNICAL DETAILS", self.styles['CompactHeader']))
        story.append(Spacer(1, 0.2*inch))
        
        # Export Information
        story.append(Paragraph("Export Information:", self.styles['CompactSection']))
        export_filename = os.path.basename(export_stats.get('export_path', 'N/A'))
        export_text = f"""• Output: {export_filename}<br/>• Coordinate System: WGS84 (EPSG:4326)<br/>• Processing Time: {self._get_processing_time(processing_summary)}"""
        story.append(Paragraph(export_text, self.styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Quality Metrics
        story.append(Paragraph("Quality Metrics:", self.styles['CompactSection']))
        avg_confidence = self._get_average_confidence(georeferenced_coords)
        quality_text = f"""• Average Confidence: {avg_confidence}<br/>• RTK Accuracy: ±2-5cm<br/>• Standard GPS: ±3-5m"""
        story.append(Paragraph(quality_text, self.styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Processing Times (if available)
        if processing_summary:
            story.append(Paragraph("Processing Performance:", self.styles['CompactSection']))
            perf_text = self._get_performance_summary(processing_summary)
            story.append(Paragraph(perf_text, self.styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
        
        # Spatial overview (moved from page 1)
        story.append(Paragraph("🗺️ SPATIAL OVERVIEW", self.styles['CompactSection']))
        spatial_summary = self._get_spatial_summary(georeferenced_coords)
        story.append(Paragraph(spatial_summary, self.styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Software info
        story.append(Spacer(1, 1*inch))
        story.append(Paragraph("Software: Image2Shape v2.0", self.styles['Normal']))
        
        return story
    
    def _generate_coordinate_plots(self, georeferenced_coords, processing_summary, metadata_cache=None):
        """Generate stacked coordinate plots (one above the other) with larger size"""
        if not georeferenced_coords:
            return None
            
        try:
            # Simple approach: get drone positions from provided metadata cache or fallback
            # This is the same data source used by the map widget, so it's reliable
            drone_positions = self._get_simple_drone_positions(metadata_cache)
            
            # Create figure with two subplots stacked vertically for larger maps
            # Use square aspect ratio to prevent stretching
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            # Set minimal style
            plt.style.use('default')
            
            # Plot 1: Drone GPS Locations (where images were taken)
            if drone_positions:
                lats = [pos['latitude'] for pos in drone_positions]
                lons = [pos['longitude'] for pos in drone_positions]
                rtk_qualities = [pos['rtk_quality'] for pos in drone_positions]
                
                # Color by RTK quality (same as map widget)
                colors = []
                for quality in rtk_qualities:
                    if 'High' in quality:
                        colors.append('green')
                    elif 'Fair' in quality:
                        colors.append('orange')
                    elif 'Poor' in quality:
                        colors.append('red')
                    else:
                        colors.append('blue')
                
                # Plot drone positions
                ax1.scatter(lons, lats, c=colors, s=40, alpha=0.8, edgecolors='none')
                
                # Add flight path if we have multiple positions
                if len(drone_positions) > 1:
                    # Sort by datetime for proper flight path
                    sorted_positions = sorted(drone_positions, key=lambda x: x.get('datetime', ''))
                    sorted_lons = [pos['longitude'] for pos in sorted_positions]
                    sorted_lats = [pos['latitude'] for pos in sorted_positions]
                    ax1.plot(sorted_lons, sorted_lats, 'b-', alpha=0.5, linewidth=1)
                
                ax1.set_title('Drone GPS Locations', fontsize=14, pad=15)
                ax1.grid(True, alpha=0.3)
                ax1.set_aspect('equal', adjustable='box')
                
                # No legend - cleaner look
            else:
                ax1.text(0.5, 0.5, 'No drone GPS data available', 
                        transform=ax1.transAxes, ha='center', va='center',
                        fontsize=12, alpha=0.5)
                ax1.set_title('Drone GPS Locations', fontsize=14, pad=15)
            
            # Plot 2: Feature Locations
            df = pd.DataFrame(georeferenced_coords)
            if 'world_latitude' in df.columns and 'world_longitude' in df.columns:
                # Simple green dots for all features - no confidence coloring
                ax2.scatter(df['world_longitude'], df['world_latitude'], 
                          c='green', s=30, alpha=0.8, edgecolors='none')
                
                ax2.set_title('Feature Locations', fontsize=14, pad=15)
                ax2.grid(True, alpha=0.3)
                ax2.set_aspect('equal', adjustable='box')
            else:
                ax2.text(0.5, 0.5, 'No feature data available', 
                        transform=ax2.transAxes, ha='center', va='center',
                        fontsize=12, alpha=0.5)
                ax2.set_title('Feature Locations', fontsize=14, pad=15)
            
            # Set same axis limits for both plots if we have data for both
            if drone_positions and 'world_latitude' in df.columns:
                drone_lats = [pos['latitude'] for pos in drone_positions]
                drone_lons = [pos['longitude'] for pos in drone_positions]
                feature_lats = df['world_latitude'].tolist()
                feature_lons = df['world_longitude'].tolist()
                
                all_lats = drone_lats + feature_lats
                all_lons = drone_lons + feature_lons
                
                lat_margin = (max(all_lats) - min(all_lats)) * 0.1
                lon_margin = (max(all_lons) - min(all_lons)) * 0.1
                
                ax1.set_xlim(min(all_lons) - lon_margin, max(all_lons) + lon_margin)
                ax1.set_ylim(min(all_lats) - lat_margin, max(all_lats) + lat_margin)
                ax2.set_xlim(min(all_lons) - lon_margin, max(all_lons) + lon_margin)
                ax2.set_ylim(min(all_lats) - lat_margin, max(all_lats) + lat_margin)
            
            # Remove tick labels for cleaner look but keep the grid
            ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            
            # Ensure tight layout but preserve aspect ratios
            plt.tight_layout(pad=2.0)
            
            # Save to temporary file with white background and preserve aspect ratio
            chart_path = tempfile.mktemp(suffix='.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', 
                       pad_inches=0.2)  # Small padding to prevent clipping
            plt.close()
            
            return chart_path
            
        except Exception as e:
            print(f"Error generating coordinate plots: {e}")
            return None
    
    def _get_simple_drone_positions(self, metadata_cache=None):
        """Get drone positions using a simple and reliable approach"""
        try:
            # First priority: use provided metadata cache
            if metadata_cache:
                return self._extract_positions_from_metadata(metadata_cache)
            
            # Second priority: try to find the main app instance
            import sys
            main_app = None
            
            # Look for the app in sys.modules more carefully
            for module_name, module in sys.modules.items():
                if hasattr(module, '__dict__'):
                    for attr_name, attr_value in module.__dict__.items():
                        if (hasattr(attr_value, 'metadata_cache') and 
                            hasattr(attr_value, 'image_files') and
                            hasattr(attr_value, 'map_widget') and
                            len(getattr(attr_value, 'image_files', [])) > 0):
                            main_app = attr_value
                            break
                if main_app:
                    break
            
            if main_app and hasattr(main_app, 'metadata_cache'):
                return self._extract_positions_from_metadata(main_app.metadata_cache)
            
            # Final fallback: return empty list
            return []
            
        except Exception as e:
            print(f"Error getting drone positions: {e}")
            return []
    
    def _extract_positions_from_metadata(self, metadata_cache):
        """Extract drone positions from metadata cache (same logic as map widget)"""
        positions = []
        
        for image_path, metadata in metadata_cache.items():
            # Check for GPS data using the same keys as map widget
            if metadata.get('latitude') and metadata.get('longitude'):
                try:
                    lat = self._parse_coordinate(metadata.get('latitude'))
                    lon = self._parse_coordinate(metadata.get('longitude'))
                    
                    if lat is not None and lon is not None:
                        position = {
                            'image_path': image_path,
                            'image_name': os.path.basename(image_path),
                            'latitude': lat,
                            'longitude': lon,
                            'altitude': metadata.get('altitude'),
                            'rtk_quality': self._assess_rtk_quality(metadata),
                            'datetime': metadata.get('datetime_original', 'Unknown')
                        }
                        positions.append(position)
                except Exception as e:
                    print(f"Error parsing coordinates for {image_path}: {e}")
        
        return positions
    
    def _parse_coordinate(self, coord_str):
        """Parse coordinate string to decimal degrees (same as map widget)"""
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
            
            # Fallback: Handle DMS format for backward compatibility
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
        """Assess RTK quality based on standard deviations (same as map widget)"""
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
    
    def _get_confidence_range(self, georeferenced_coords):
        """Get confidence range string"""
        if not georeferenced_coords:
            return "N/A"
        
        df = pd.DataFrame(georeferenced_coords)
        if 'confidence_score' in df.columns:
            min_conf = df['confidence_score'].min()
            max_conf = df['confidence_score'].max()
            return f"{min_conf:.2f} - {max_conf:.2f}"
        return "N/A"
    
    def _get_gps_quality_summary(self, georeferenced_coords):
        """Get GPS quality summary string"""
        if not georeferenced_coords:
            return "N/A"
        
        df = pd.DataFrame(georeferenced_coords)
        if 'rtk_status' in df.columns:
            # Handle both string and integer RTK status formats
            rtk_count = ((df['rtk_status'] == 'RTK') | (df['rtk_status'] == 1)).sum()
            total = len(df)
            rtk_pct = (rtk_count / total * 100) if total > 0 else 0
            std_pct = 100 - rtk_pct
            return f"{rtk_pct:.0f}% RTK, {std_pct:.0f}% Standard"
        return "Standard GPS"
    
    def _get_spatial_summary(self, georeferenced_coords):
        """Get spatial overview summary with ranges in meters"""
        if not georeferenced_coords:
            return "No spatial data available."
        
        df = pd.DataFrame(georeferenced_coords)
        if 'world_latitude' in df.columns and 'world_longitude' in df.columns:
            lat_range = df['world_latitude'].max() - df['world_latitude'].min()
            lon_range = df['world_longitude'].max() - df['world_longitude'].min()
            center_lat = df['world_latitude'].mean()
            center_lon = df['world_longitude'].mean()
            
            # Convert ranges to meters
            lat_range_m = lat_range * 111000  # 1 degree latitude ≈ 111km
            lon_range_m = lon_range * 111000 * np.cos(np.radians(center_lat))  # Adjust for latitude
            area_km2 = lat_range_m * lon_range_m / 1000000  # Convert to km²
            
            return f"""
            • Area: {area_km2:.2f} km² • Center: {center_lat:.6f}°N, {center_lon:.6f}°E<br/>
            • Range: {lat_range_m:.0f}m N-S × {lon_range_m:.0f}m E-W
            """
        return "Spatial data not available."
    
    def _get_processing_time(self, processing_summary):
        """Get total processing time"""
        if not processing_summary:
            return "N/A"
        
        # Try different possible keys for total time
        total_time = (processing_summary.get('total_time') or 
                     processing_summary.get('export') or 
                     processing_summary.get('processing_time'))
        
        if total_time:
            return f"{total_time:.1f}s"
        return "N/A"
    
    def _get_average_confidence(self, georeferenced_coords):
        """Get average confidence score"""
        if not georeferenced_coords:
            return "N/A"
        
        df = pd.DataFrame(georeferenced_coords)
        if 'confidence_score' in df.columns:
            return f"{df['confidence_score'].mean():.2f}"
        return "N/A"
    
    def _get_performance_summary(self, processing_summary):
        """Get performance summary text"""
        if not processing_summary:
            return "Performance data not available."
        
        # Extract timing information from our tracked times
        load_time = processing_summary.get('load_folder', 0)
        training_time = processing_summary.get('train_model', 0)
        export_time = processing_summary.get('export', 0)
        
        # Calculate total time
        total_time = load_time + training_time + export_time
        
        summary_parts = []
        if load_time > 0:
            summary_parts.append(f"• Load Folder: {load_time:.1f}s")
        if training_time > 0:
            summary_parts.append(f"• ML Training: {training_time:.1f}s")
        if export_time > 0:
            summary_parts.append(f"• Export: {export_time:.1f}s")
        if total_time > 0:
            summary_parts.append(f"• Total Time: {total_time:.1f}s")
        
        if summary_parts:
            return "<br/>".join(summary_parts)
        else:
            return "Performance data not available."