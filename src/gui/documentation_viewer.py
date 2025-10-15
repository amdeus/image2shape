#!/usr/bin/env python3
"""
Documentation Viewer for Image2Shape

This module provides a window for displaying documentation files (README and User Guide)
within the application interface.

Author: Image2Shape Development Team
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os
from pathlib import Path


class DocumentationViewer:
    """
    Documentation viewer window for displaying README and User Guide
    """
    
    def __init__(self, parent, title, content):
        """
        Initialize documentation viewer
        
        Args:
            parent: Parent window
            title: Window title
            content: Markdown content to display
        """
        self.parent = parent
        self.title = title
        self.content = content
        
        # Create window
        self.window = tk.Toplevel(parent)
        self.window.title(f"Image2Shape - {title}")
        self.window.geometry("900x700")
        
        # Center the window relative to parent
        self.center_window()
        
        # Make window modal
        self.window.transient(parent)
        self.window.grab_set()
        
        # Setup UI
        self.setup_ui()
        
        # Load content
        self.display_content()
        
    def center_window(self):
        """Center the window relative to parent"""
        self.window.update_idletasks()
        
        # Get parent window position and size
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        # Calculate center position
        window_width = 900
        window_height = 700
        x = parent_x + (parent_width // 2) - (window_width // 2)
        y = parent_y + (parent_height // 2) - (window_height // 2)
        
        # Ensure window doesn't go off screen
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = max(0, min(x, screen_width - window_width))
        y = max(0, min(y, screen_height - window_height))
        
        self.window.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title label
        title_label = ttk.Label(main_frame, text=self.title, font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Text widget with scrollbar
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create text widget
        self.text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            font=('Consolas', 10),
            bg='#f8f9fa',
            fg='#212529',
            selectbackground='#007acc',
            relief=tk.FLAT,
            borderwidth=1,
            padx=10,
            pady=10
        )
        
        # Create scrollbar
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.text_widget.yview)
        self.text_widget.config(yscrollcommand=scrollbar.set)
        
        # Pack text widget and scrollbar
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Close button
        close_button = ttk.Button(button_frame, text="Close", command=self.close_window)
        close_button.pack(side=tk.RIGHT)
        
        # Configure text widget tags for formatting
        self.setup_text_formatting()
        
    def setup_text_formatting(self):
        """Setup text formatting tags"""
        # Headers
        self.text_widget.tag_configure("h1", font=('Arial', 16, 'bold'), spacing1=10, spacing3=5)
        self.text_widget.tag_configure("h2", font=('Arial', 14, 'bold'), spacing1=8, spacing3=4)
        self.text_widget.tag_configure("h3", font=('Arial', 12, 'bold'), spacing1=6, spacing3=3)
        
        # Code blocks
        self.text_widget.tag_configure("code", font=('Consolas', 9), background='#e9ecef', relief=tk.SOLID, borderwidth=1)
        
        # Bold text
        self.text_widget.tag_configure("bold", font=('Arial', 10, 'bold'))
        
        # Lists
        self.text_widget.tag_configure("list", lmargin1=20, lmargin2=20)
        
    def display_content(self):
        """Display the markdown content with basic formatting"""
        lines = self.content.split('\n')
        
        for line in lines:
            line = line.rstrip()
            
            # Headers
            if line.startswith('# '):
                self.text_widget.insert(tk.END, line[2:] + '\n', "h1")
            elif line.startswith('## '):
                self.text_widget.insert(tk.END, line[3:] + '\n', "h2")
            elif line.startswith('### '):
                self.text_widget.insert(tk.END, line[4:] + '\n', "h3")
            
            # Code blocks
            elif line.startswith('```'):
                if hasattr(self, '_in_code_block'):
                    delattr(self, '_in_code_block')
                    self.text_widget.insert(tk.END, '\n')
                else:
                    self._in_code_block = True
                    self.text_widget.insert(tk.END, '\n')
                continue
            elif hasattr(self, '_in_code_block'):
                self.text_widget.insert(tk.END, line + '\n', "code")
            
            # Lists
            elif line.startswith('- ') or line.startswith('* '):
                self.text_widget.insert(tk.END, '• ' + line[2:] + '\n', "list")
            elif line.strip() and line[0].isdigit() and '. ' in line:
                self.text_widget.insert(tk.END, line + '\n', "list")
            
            # Bold text (simple **text** detection)
            elif '**' in line:
                parts = line.split('**')
                for i, part in enumerate(parts):
                    if i % 2 == 0:
                        self.text_widget.insert(tk.END, part)
                    else:
                        self.text_widget.insert(tk.END, part, "bold")
                self.text_widget.insert(tk.END, '\n')
            
            # Regular text
            else:
                self.text_widget.insert(tk.END, line + '\n')
        
        # Make text widget read-only
        self.text_widget.config(state=tk.DISABLED)
        
        # Scroll to top
        self.text_widget.see("1.0")
        
    def close_window(self):
        """Close the documentation window"""
        self.window.destroy()


def show_readme(parent):
    """Show README documentation"""
    readme_path = Path(__file__).parent.parent.parent / "doc" / "README.md"
    
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        DocumentationViewer(parent, "README", content)
    except FileNotFoundError:
        tk.messagebox.showerror("Error", f"README file not found at: {readme_path}")
    except Exception as e:
        tk.messagebox.showerror("Error", f"Failed to load README: {str(e)}")


def show_user_guide(parent):
    """Show User Guide documentation"""
    user_guide_path = Path(__file__).parent.parent.parent / "doc" / "USER_GUIDE.md"
    
    try:
        with open(user_guide_path, 'r', encoding='utf-8') as f:
            content = f.read()
        DocumentationViewer(parent, "User Guide", content)
    except FileNotFoundError:
        tk.messagebox.showerror("Error", f"User Guide file not found at: {user_guide_path}")
    except Exception as e:
        tk.messagebox.showerror("Error", f"Failed to load User Guide: {str(e)}")