#!/usr/bin/env python3
"""
Export Configuration Dialog for Image2Shape

This module provides an enhanced export interface with multiple processing modes,
output configuration, and optional PDF report generation.

Features:
- Output folder selection
- Processing mode selection (Batch vs Single Image)
- PDF report generation option
- Professional UI with clear workflow

Author: Image2Shape Development Team
Version: 1.0 - Enhanced Export Interface
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from typing import Optional, Callable
from pathlib import Path


class ExportConfigurationDialog:
    """
    Enhanced export configuration dialog with multiple processing modes
    """
    
    def __init__(self, parent, default_output_dir: Optional[str] = None):
        """
        Initialize export configuration dialog
        
        Args:
            parent: Parent window
            default_output_dir: Default output directory
        """
        self.parent = parent
        self.result = None
        self.cancelled = False
        
        # Configuration values
        self.output_folder = default_output_dir or str(Path.home() / "Desktop")
        self.include_report = True
        self.current_image_only = False
        self.export_type = "feature_polygons"  # Default export type
        self.buffer_size_m = 0.5  # Default buffer size in meters
        
        # Create modal dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Export Configuration")
        self.dialog.geometry("450x350")
        self.dialog.minsize(430, 320)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
        self.setup_ui()
        
        # Center dialog after UI is fully set up
        self.dialog.after(10, self.center_dialog)
        
    def setup_ui(self):
        """Create the minimal export configuration UI"""
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header_label = ttk.Label(
            main_frame, 
            text="Export Configuration", 
            font=('Arial', 12, 'bold')
        )
        header_label.pack(pady=(0, 15))
        
        # Output folder selection
        self.create_output_folder_section(main_frame)
        
        # Export type selection
        self.create_export_type_section(main_frame)
        
        # Export options
        self.create_export_options_section(main_frame)
        
        # Buttons
        self.create_button_section(main_frame)
        
    def create_output_folder_section(self, parent):
        """Create output folder selection section"""
        # Output folder label
        ttk.Label(parent, text="Output Folder:").pack(anchor=tk.W, pady=(0, 5))
        
        # Folder path display and browse button
        path_frame = ttk.Frame(parent)
        path_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.folder_var = tk.StringVar(value=self.output_folder)
        folder_entry = ttk.Entry(path_frame, textvariable=self.folder_var, state='readonly')
        folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        browse_button = ttk.Button(
            path_frame, 
            text="Browse...", 
            command=self.browse_output_folder,
            width=12
        )
        browse_button.pack(side=tk.RIGHT)
        
    def create_export_type_section(self, parent):
        """Create export type selection section"""
        # Export type label
        ttk.Label(parent, text="Export Type:").pack(anchor=tk.W, pady=(0, 5))
        
        # Export type dropdown
        export_type_frame = ttk.Frame(parent)
        export_type_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.export_type_var = tk.StringVar(value="Feature Polygons (Default)")
        self.export_type_combo = ttk.Combobox(
            export_type_frame,
            textvariable=self.export_type_var,
            values=[
                "Feature Polygons (Default)",
                "Feature Points Only", 
                "Combined Export"
            ],
            state="readonly",
            width=30
        )
        self.export_type_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.export_type_combo.bind('<<ComboboxSelected>>', self.on_export_type_change)
        
        # Buffer size section (initially visible)
        self.buffer_frame = ttk.Frame(parent)
        self.buffer_frame.pack(fill=tk.X, pady=(5, 15))
        
        buffer_label = ttk.Label(self.buffer_frame, text="Buffer Size:")
        buffer_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.buffer_var = tk.DoubleVar(value=self.buffer_size_m)
        buffer_entry = ttk.Entry(self.buffer_frame, textvariable=self.buffer_var, width=8)
        buffer_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Label(self.buffer_frame, text="meters").pack(side=tk.LEFT, padx=(0, 5))
        
        # Help text for buffer size
        buffer_help = ttk.Label(
            self.buffer_frame, 
            text="(Points will be buffered and merged into polygons)",
            font=('Arial', 8),
            foreground='gray'
        )
        buffer_help.pack(side=tk.LEFT, padx=(10, 0))
        
    def create_processing_mode_section(self, parent):
        """Create processing mode selection section"""
        mode_frame = ttk.LabelFrame(parent, text="Processing Mode", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.mode_var = tk.StringVar(value=self.processing_mode)
        
        # Batch processing option
        batch_radio = ttk.Radiobutton(
            mode_frame,
            text="Batch Processing",
            variable=self.mode_var,
            value="batch",
            command=self.on_mode_change
        )
        batch_radio.pack(anchor=tk.W, pady=(0, 5))
        
        batch_info = ttk.Label(
            mode_frame,
            text="• Process all loaded images with ML detection and georeferencing",
            font=('Arial', 9),
            foreground='gray'
        )
        batch_info.pack(anchor=tk.W, padx=(20, 0), pady=(0, 10))
        
        # Single image option
        single_radio = ttk.Radiobutton(
            mode_frame,
            text="Active Image Only",
            variable=self.mode_var,
            value="single",
            command=self.on_mode_change
        )
        single_radio.pack(anchor=tk.W, pady=(0, 5))
        
        single_info = ttk.Label(
            mode_frame,
            text="• Export predictions from currently displayed image only",
            font=('Arial', 9),
            foreground='gray'
        )
        single_info.pack(anchor=tk.W, padx=(20, 0))
        
    def create_export_options_section(self, parent):
        """Create export options section"""
        # PDF Report option
        self.report_var = tk.BooleanVar(value=self.include_report)
        report_check = ttk.Checkbutton(
            parent,
            text="☑ Include PDF Report",
            variable=self.report_var,
            command=self.on_report_change
        )
        report_check.pack(anchor=tk.W, pady=(0, 5))
        
        # Current image only option
        self.current_image_var = tk.BooleanVar(value=self.current_image_only)
        current_image_check = ttk.Checkbutton(
            parent,
            text="☑ Only Export Current Image",
            variable=self.current_image_var,
            command=self.on_current_image_change
        )
        current_image_check.pack(anchor=tk.W, pady=(0, 15))
        
    def create_info_section(self, parent):
        """Create information section"""
        info_frame = ttk.Frame(parent)
        info_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.info_label = ttk.Label(
            info_frame,
            text="Ready to export. Click 'Start Export' to begin processing.",
            font=('Arial', 10),
            foreground='blue'
        )
        self.info_label.pack(anchor=tk.W)
        
    def create_button_section(self, parent):
        """Create button section"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Cancel button
        cancel_button = ttk.Button(
            button_frame,
            text="Cancel",
            command=self.on_cancel,
            width=12
        )
        cancel_button.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Start export button
        self.export_button = ttk.Button(
            button_frame,
            text="Start Export",
            command=self.on_start_export,
            width=12
        )
        self.export_button.pack(side=tk.RIGHT)
        
    def center_dialog(self):
        """Center dialog on parent window with proper sizing"""
        # Force update to get actual sizes
        self.dialog.update_idletasks()
        self.parent.update_idletasks()
        
        # Get parent window position and size
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        # Get dialog's actual current dimensions
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        # If dialog hasn't been rendered yet, use requested size as fallback
        if dialog_width <= 1 or dialog_height <= 1:
            dialog_width = 450
            dialog_height = 350
        
        # Calculate center position
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        # Ensure dialog doesn't go off screen
        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()
        
        x = max(0, min(x, screen_width - dialog_width))
        y = max(0, min(y, screen_height - dialog_height))
        
        # Set the position (keep current size)
        self.dialog.geometry(f"+{x}+{y}")
        self.dialog.lift()
        self.dialog.focus_force()
        
    def browse_output_folder(self):
        """Browse for output folder"""
        folder = filedialog.askdirectory(
            title="Select Output Folder",
            initialdir=self.output_folder
        )
        
        if folder:
            self.output_folder = folder
            self.folder_var.set(folder)
            self.update_info_display()
            
    def on_mode_change(self):
        """Handle processing mode change"""
        self.processing_mode = self.mode_var.get()
        self.update_info_display()
        
    def on_report_change(self):
        """Handle report option change"""
        self.include_report = self.report_var.get()
        self.update_info_display()
    
    def on_current_image_change(self):
        """Handle current image option change"""
        self.current_image_only = self.current_image_var.get()
        self.update_info_display()
    
    def on_export_type_change(self, event=None):
        """Handle export type change"""
        selected = self.export_type_var.get()
        
        # Map display names to internal values
        type_mapping = {
            "Feature Polygons (Default)": "feature_polygons",
            "Feature Points Only": "feature_points",
            "Combined Export": "combined"
        }
        
        self.export_type = type_mapping.get(selected, "feature_polygons")
        
        # Show/hide buffer size controls based on selection
        if self.export_type == "feature_polygons":
            self.buffer_frame.pack(fill=tk.X, pady=(5, 15))
        else:
            self.buffer_frame.pack_forget()
        
        self.update_info_display()
        
    def update_info_display(self):
        """Update information display based on current settings"""
        mode_text = "current image only" if self.current_image_only else "all loaded images"
        report_text = " with PDF report" if self.include_report else ""
        
        # Add export type description
        type_descriptions = {
            "feature_polygons": "as buffered polygons",
            "feature_points": "as point features",
            "combined": "with drone positions, features, and image footprints"
        }
        type_text = type_descriptions.get(self.export_type, "")
        
        info_text = f"Ready to export {mode_text} {type_text}{report_text} to:\n{self.output_folder}"
        if hasattr(self, 'info_label'):
            self.info_label.config(text=info_text)
        
    def on_start_export(self):
        """Handle start export button click"""
        # Validate settings
        if not os.path.exists(self.output_folder):
            try:
                os.makedirs(self.output_folder, exist_ok=True)
            except Exception as e:
                messagebox.showerror(
                    "Error", 
                    f"Cannot create output folder:\n{self.output_folder}\n\nError: {e}"
                )
                return
        
        # Check if folder is writable
        test_file = os.path.join(self.output_folder, "test_write.tmp")
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            messagebox.showerror(
                "Error", 
                f"Cannot write to output folder:\n{self.output_folder}\n\nError: {e}"
            )
            return
        
        # Get buffer size value
        try:
            self.buffer_size_m = self.buffer_var.get()
            if self.buffer_size_m <= 0:
                messagebox.showerror("Error", "Buffer size must be greater than 0")
                return
        except tk.TclError:
            messagebox.showerror("Error", "Please enter a valid buffer size")
            return
        
        # Store result and close
        self.result = {
            'output_folder': self.output_folder,
            'include_report': self.include_report,
            'current_image_only': self.current_image_only,
            'export_type': self.export_type,
            'buffer_size_m': self.buffer_size_m
        }
        
        self.dialog.destroy()
        
    def on_cancel(self):
        """Handle cancel button or window close"""
        self.cancelled = True
        self.dialog.destroy()
        
    def get_result(self):
        """
        Get the export configuration result
        
        Returns:
            Dict or None: Export configuration if not cancelled
        """
        return None if self.cancelled else self.result


def show_export_dialog(parent, default_output_dir: Optional[str] = None) -> Optional[dict]:
    """
    Show export configuration dialog and return user selection
    
    Args:
        parent: Parent window
        default_output_dir: Default output directory
        
    Returns:
        Dict or None: Export configuration if not cancelled
    """
    dialog = ExportConfigurationDialog(parent, default_output_dir)
    parent.wait_window(dialog.dialog)
    return dialog.get_result()