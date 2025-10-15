#!/usr/bin/env python3
"""
Image2Shape - Drone Image Feature Detection and Georeferencing Tool
Entry point for the application
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from gui.main_window import Image2ShapeApp

def main():
    """Main entry point for Image2Shape application"""
    app = Image2ShapeApp()
    app.run()

if __name__ == "__main__":
    main()