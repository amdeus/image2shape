#!/usr/bin/env python3
"""
ML Algorithm Selection Dialog for Image2Shape

This module provides a clean, professional interface for selecting and configuring
ML algorithms. Designed with minimal user input and maximum clarity in mind.

Key Features:
- Clean, intuitive algorithm selection
- Visual algorithm comparison cards
- Smart parameter configuration
- Pretrained model support (future)
- Real-time capability assessment
- Professional UX design

Author: Image2Shape Development Team
Version: 2.1 - Multi-Algorithm Foundation
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Any, Optional, Callable
import os
from pathlib import Path

class MLAlgorithmDialog:
    """
    Professional ML Algorithm Selection Dialog
    
    This dialog provides a clean, user-friendly interface for selecting and
    configuring ML algorithms. It follows modern UX principles with minimal
    user input requirements and clear visual feedback.
    
    Design Principles:
    - Minimal clicks to get started
    - Clear visual hierarchy
    - Intelligent defaults
    - Progressive disclosure of advanced options
    - Immediate feedback on choices
    """
    
    def __init__(self, parent, current_algorithm=None, annotation_count=0):
        """
        Initialize the ML Algorithm Selection Dialog
        
        Args:
            parent: Parent window
            current_algorithm: Currently selected algorithm (if any)
            annotation_count: Number of available annotations for training
        """
        self.parent = parent
        self.current_algorithm = current_algorithm
        self.annotation_count = annotation_count
        self.result = None
        self.dialog = None
        
        # Import algorithm factory
        try:
            from ml.algorithm_factory import MLAlgorithmFactory
            self.factory = MLAlgorithmFactory
            self.available_algorithms = self.factory.get_available_algorithms()
        except ImportError as e:
            messagebox.showerror("Error", f"Could not load ML algorithms: {e}")
            return
        
        self.create_dialog()
    
    def create_dialog(self):
        """Create the main dialog window"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Select ML Algorithm")
        self.dialog.geometry("450x200")
        self.dialog.resizable(False, False)
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # Center the dialog relative to parent window
        self.dialog.update_idletasks()
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        # Calculate center position relative to parent
        x = parent_x + (parent_width // 2) - (450 // 2)
        y = parent_y + (parent_height // 2) - (200 // 2)
        
        self.dialog.geometry(f"450x200+{x}+{y}")
        
        # Configure styles
        self.setup_styles()
        
        # Initialize state
        self.parameters_visible = False
        self.selected_algorithm = None
        
        # Create minimalistic content
        self.create_minimalistic_content()
        
        # Set initial selection
        self.select_default_algorithm()
        
        # Bind events
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
        self.dialog.bind('<Return>', lambda e: self.on_confirm())
        self.dialog.bind('<Escape>', lambda e: self.on_cancel())
    
    def setup_styles(self):
        """Configure custom styles for the dialog"""
        style = ttk.Style()
        
        # Keep default dialog background (gray) - don't override
        # Algorithm card styles that work well on gray background
        style.configure('AlgorithmCard.TFrame', 
                       relief='solid', 
                       borderwidth=1)
        
        style.configure('SelectedCard.TFrame', 
                       relief='raised', 
                       borderwidth=3)
        
        # Header styles - use darker colors for better contrast on gray
        style.configure('Header.TLabel', 
                       font=('Arial', 16, 'bold'),
                       foreground='#0d47a1')  # Darker blue for gray background
        
        style.configure('Subheader.TLabel', 
                       font=('Arial', 11),
                       foreground='#212121')  # Darker gray for better contrast
        
        # Algorithm title style
        style.configure('AlgorithmTitle.TLabel', 
                       font=('Arial', 12, 'bold'),
                       foreground='#1565c0')
        
        # Status styles with better visibility on gray
        style.configure('Success.TLabel', 
                       foreground='#1b5e20',  # Darker green
                       font=('Arial', 10, 'bold'))
        
        style.configure('Warning.TLabel', 
                       foreground='#e65100',  # Darker orange
                       font=('Arial', 10, 'bold'))
        
        style.configure('Error.TLabel', 
                       foreground='#b71c1c',  # Darker red
                       font=('Arial', 10, 'bold'))
        
        # Remove special button styles - use default gray theme
        
        # Don't override frame backgrounds - let them use default gray theme
    
    def create_minimalistic_content(self):
        """Create the minimalistic dialog content"""
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # No title needed - window title is sufficient
        
        # Algorithm selection row
        selection_frame = ttk.Frame(main_frame)
        selection_frame.pack(fill=tk.X, pady=(10, 10))
        
        ttk.Label(selection_frame, text="Algorithm:").pack(side=tk.LEFT)
        
        # Algorithm dropdown
        self.algorithm_var = tk.StringVar()
        algorithm_names = [info['name'] for info in self.available_algorithms.values()]
        self.algorithm_combo = ttk.Combobox(selection_frame, 
                                           textvariable=self.algorithm_var,
                                           values=algorithm_names,
                                           state='readonly',
                                           width=25)
        self.algorithm_combo.pack(side=tk.LEFT, padx=(10, 5))
        self.algorithm_combo.bind('<<ComboboxSelected>>', self.on_algorithm_change)
        
        # Parameters toggle button
        self.params_button = ttk.Button(selection_frame, 
                                       text="...", 
                                       width=3,
                                       command=self.toggle_parameters)
        self.params_button.pack(side=tk.LEFT, padx=(5, 0))
        
        # Parameters frame (initially hidden)
        self.params_frame = ttk.LabelFrame(main_frame, text="Parameters")
        self.params_content = ttk.Frame(self.params_frame)
        self.params_content.pack(fill=tk.X, padx=10, pady=10)
        
        # Status line - simple text without colors
        if self.annotation_count >= 4:
            status_text = f"Ready to train with {self.annotation_count:,} annotations"
        elif self.annotation_count > 0:
            status_text = f"Only {self.annotation_count} annotations - Need at least 4"
        else:
            status_text = "No annotations - Add annotations first"
        
        self.status_label = ttk.Label(main_frame, text=status_text)
        self.status_label.pack(anchor=tk.W, pady=(10, 15))
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, 
                  text="Cancel", 
                  command=self.on_cancel).pack(side=tk.RIGHT, padx=(10, 0))
        
        self.train_button = ttk.Button(button_frame, 
                                      text="Train Model", 
                                      command=self.on_confirm)
        self.train_button.pack(side=tk.RIGHT)
        
        # Update button state
        self.update_train_button()
    
    def on_algorithm_change(self, event=None):
        """Handle algorithm selection change"""
        selected_name = self.algorithm_var.get()
        
        # Find algorithm ID by name
        for algo_id, algo_info in self.available_algorithms.items():
            if algo_info['name'] == selected_name:
                self.selected_algorithm = algo_id
                break
        
        # Update parameters
        self.update_parameters()
        self.update_train_button()
    
    def toggle_parameters(self):
        """Toggle parameter visibility"""
        if self.parameters_visible:
            self.params_frame.pack_forget()
            self.dialog.geometry("450x200")
            self.parameters_visible = False
        else:
            self.params_frame.pack(fill=tk.X, pady=(10, 10))
            self.dialog.geometry("450x300")
            self.parameters_visible = True
    
    def update_parameters(self):
        """Update parameter controls based on selected algorithm"""
        # Clear existing parameters
        for widget in self.params_content.winfo_children():
            widget.destroy()
        
        if not self.selected_algorithm:
            return
        
        if self.selected_algorithm == 'random_forest':
            self.create_rf_parameters()
        elif self.selected_algorithm == 'knn':
            self.create_knn_parameters()
    
    def create_rf_parameters(self):
        """Create Random Forest parameters"""
        # Trees parameter
        trees_frame = ttk.Frame(self.params_content)
        trees_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(trees_frame, text="Trees:").pack(side=tk.LEFT)
        self.trees_var = tk.IntVar(value=100)
        ttk.Spinbox(trees_frame, 
                   from_=10, to=500, 
                   textvariable=self.trees_var, 
                   width=10).pack(side=tk.RIGHT)
        
        # Random seed parameter
        seed_frame = ttk.Frame(self.params_content)
        seed_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(seed_frame, text="Random Seed:").pack(side=tk.LEFT)
        self.seed_var = tk.IntVar(value=42)
        ttk.Spinbox(seed_frame, 
                   from_=1, to=1000, 
                   textvariable=self.seed_var, 
                   width=10).pack(side=tk.RIGHT)
    
    def create_knn_parameters(self):
        """Create KNN parameters"""
        # Neighbors parameter
        neighbors_frame = ttk.Frame(self.params_content)
        neighbors_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(neighbors_frame, text="Neighbors (k):").pack(side=tk.LEFT)
        self.neighbors_var = tk.IntVar(value=5)
        ttk.Spinbox(neighbors_frame, 
                   from_=1, to=20, 
                   textvariable=self.neighbors_var, 
                   width=10).pack(side=tk.RIGHT)
        
        # Distance metric parameter
        distance_frame = ttk.Frame(self.params_content)
        distance_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(distance_frame, text="Distance:").pack(side=tk.LEFT)
        self.metric_var = tk.StringVar(value='euclidean')
        ttk.Combobox(distance_frame, 
                    textvariable=self.metric_var,
                    values=['euclidean', 'manhattan'],
                    state='readonly',
                    width=12).pack(side=tk.RIGHT)
    
    def select_default_algorithm(self):
        """Select the default algorithm"""
        if 'random_forest' in self.available_algorithms:
            self.algorithm_var.set(self.available_algorithms['random_forest']['name'])
            self.selected_algorithm = 'random_forest'
            self.update_parameters()
            self.update_train_button()
    
    def update_train_button(self):
        """Update the state of train button"""
        if self.selected_algorithm and self.annotation_count >= 4:
            self.train_button.configure(state='normal')
        else:
            self.train_button.configure(state='disabled')
    
    
    
    
    
    def get_algorithm_parameters(self) -> Dict[str, Any]:
        """Get the configured parameters for the selected algorithm"""
        if self.selected_algorithm == 'random_forest':
            params = {}
            if hasattr(self, 'trees_var'):
                params['n_estimators'] = self.trees_var.get()
            if hasattr(self, 'seed_var'):
                params['random_state'] = self.seed_var.get()
            return params
        
        elif self.selected_algorithm == 'knn':
            params = {}
            if hasattr(self, 'neighbors_var'):
                params['n_neighbors'] = self.neighbors_var.get()
            if hasattr(self, 'weights_var'):
                params['weights'] = self.weights_var.get()
            if hasattr(self, 'metric_var'):
                params['metric'] = self.metric_var.get()
            return params
        
        return {}
    
    def on_confirm(self):
        """Handle confirm button click"""
        if not self.selected_algorithm:
            messagebox.showwarning("No Selection", "Please select an algorithm first.")
            return
        
        if self.annotation_count < 4:
            messagebox.showwarning("Insufficient Data", 
                                 f"Need at least 4 annotations for training.\n"
                                 f"Current: {self.annotation_count} annotations")
            return
        
        # Prepare result
        self.result = {
            'algorithm_type': self.selected_algorithm,
            'parameters': self.get_algorithm_parameters(),
            'confirmed': True,
            'show_progress': True  # Flag to show progress dialog
        }
        
        self.dialog.destroy()
    
    def on_cancel(self):
        """Handle cancel button click"""
        self.result = {'confirmed': False}
        self.dialog.destroy()
    
    def get_result(self) -> Optional[Dict[str, Any]]:
        """Get the dialog result"""
        return self.result


class MLTrainingProgressDialog:
    """
    Minimalistic training progress dialog
    """
    
    def __init__(self, parent, algorithm_name):
        """Initialize training progress dialog"""
        self.parent = parent
        self.algorithm_name = algorithm_name
        self.dialog = None
        self.progress_var = None
        self.status_var = None
        self.cancelled = False
        
        self.create_dialog()
    
    def create_dialog(self):
        """Create the progress dialog"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Training Model")
        self.dialog.geometry("400x120")
        self.dialog.resizable(False, False)
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # Center relative to parent
        self.dialog.update_idletasks()
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        x = parent_x + (parent_width // 2) - (400 // 2)
        y = parent_y + (parent_height // 2) - (120 // 2)
        self.dialog.geometry(f"400x120+{x}+{y}")
        
        # Prevent closing during training
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel_request)
        
        # Create content
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # No title needed - window title is sufficient
        
        # Status text
        self.status_var = tk.StringVar(value="Preparing training data...")
        ttk.Label(main_frame, textvariable=self.status_var).pack(anchor=tk.W, pady=(0, 10))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, 
                                           variable=self.progress_var,
                                           maximum=100,
                                           length=360)
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        # Cancel button (initially disabled)
        self.cancel_button = ttk.Button(main_frame, 
                                       text="Cancel", 
                                       command=self.on_cancel_request,
                                       state='disabled')
        self.cancel_button.pack(side=tk.RIGHT)
    
    def update_progress(self, progress: float, status: str):
        """Update progress and status"""
        if self.dialog and self.dialog.winfo_exists():
            self.progress_var.set(progress)
            self.status_var.set(status)
            self.dialog.update()
    
    def enable_cancel(self):
        """Enable the cancel button"""
        if self.cancel_button:
            self.cancel_button.configure(state='normal')
    
    def on_cancel_request(self):
        """Handle cancel request"""
        self.cancelled = True
        # Don't close dialog immediately - let training handle it
    
    def close(self):
        """Close the dialog"""
        if self.dialog:
            self.dialog.destroy()
    
    def is_cancelled(self):
        """Check if training was cancelled"""
        return self.cancelled


def show_ml_algorithm_dialog(parent, current_algorithm=None, annotation_count=0) -> Optional[Dict[str, Any]]:
    """
    Show the ML Algorithm Selection Dialog
    
    Args:
        parent: Parent window
        current_algorithm: Currently selected algorithm
        annotation_count: Number of available annotations
        
    Returns:
        Dictionary with selection result or None if cancelled
    """
    dialog = MLAlgorithmDialog(parent, current_algorithm, annotation_count)
    parent.wait_window(dialog.dialog)
    return dialog.get_result()


def show_ml_training_progress(parent, algorithm_name) -> 'MLTrainingProgressDialog':
    """
    Show the ML Training Progress Dialog
    
    Args:
        parent: Parent window
        algorithm_name: Name of the algorithm being trained
        
    Returns:
        MLTrainingProgressDialog instance for progress updates
    """
    return MLTrainingProgressDialog(parent, algorithm_name)