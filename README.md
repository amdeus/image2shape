# Image2Shape 

> Direct Georeferencing for DJI P1 Drone Images - On-Site Processing Without Cloud Dependencies

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DJI P1](https://img.shields.io/badge/Camera-DJI%20P1-orange.svg)](https://www.dji.com/zenmuse-p1)

##  Overview

Image2Shape is a specialized tool for on-site drone image processing that transforms DJI P1 aerial imagery into accurate geospatial data through direct georeferencing. Designed for field operations where fast, local processing is essential without relying on cloud services or large data transfers.

###  Key Advantages
- Fast On-Site Processing: Get results in minutes, not hours
- No Cloud Dependency: Process everything locally on your device
- Minimal Data Transfer: Keep sensitive data secure and local
- Direct Georeferencing: Leverage DJI P1's high-precision GNSS capabilities
- Low Resource Requirements: Optimized for field laptops and mobile workstations

##  DJI P1 Integration

This tool is specifically optimized for DJI P1 camera systems and automatically:

- Extracts EXIF Metadata: GPS coordinates, altitude, camera parameters, and flight telemetry
- Utilizes RTK/PPK Data: High-precision positioning for centimeter-level accuracy
- Applies Camera Calibration: Automatic lens distortion correction using P1 specifications
- Processes IMU Data: Pitch, roll, yaw information for precise image orientation


##  Workflow Pipeline

```
DJI P1 Images → Metadata Extraction → Feature Detection → ML Processing → Shapefile Export
     ↓               ↓                    ↓               ↓              ↓
   📸 JPEG/RAW    🗺️ GPS+IMU Data    🎯 Annotations   🧠 Training    📁 GIS Ready
```

### 1. Input Processing
- Load DJI P1 images with embedded GNSS data
- Extract precise georeferencing parameters from EXIF metadata
- Apply camera calibration and distortion correction

### 2. Feature Annotation & Detection
- Interactive annotation tools for training data creation
- Automated feature detection using trained ML models
- Support for multiple feature types (buildings, vegetation, infrastructure)

### 3. Machine Learning Pipeline
- Training: Random Forest and KNN algorithms optimized for drone imagery
- Inference: Fast, lightweight models for real-time processing
- Validation: Built-in accuracy assessment tools

### 4. Direct Georeferencing
- Transform image coordinates to real-world coordinates using DJI P1 metadata
- Apply WGS84 geographic projection with longitude compression correction
- Generate accurate spatial references without ground control points

### 5. Shapefile Export
- Export detected features as GIS-ready shapefiles
- Include attribute data and confidence scores
- Compatible with QGIS, ArcGIS, and other GIS software

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8+ required
# exiftool for metadata extraction
sudo apt install exiftool  # Ubuntu/Debian
brew install exiftool      # macOS
```

### Installation
```bash
git clone https://github.com/amdeus/image2shape.git
cd image2shape
pip install -r doc/requirements.txt
```

### Launch Application
```bash
python main.py
```

### Basic Workflow

1. Load DJI P1 Images
   - Click "Load Images" and select your DJI P1 image folder
   - Metadata will be automatically extracted and validated

2. Create Training Annotations
   - Draw rectangles on features (red) and background (cyan)
   - Build training dataset for your specific use case

3. Train ML Model
   - Choose Random Forest or KNN algorithm
   - Train on your annotated samples (~30 seconds)

4. Process & Review
   - Run feature detection on all images
   - Adjust confidence and clustering parameters

5. Export Results
   - Generate shapefiles with precise coordinates
   - Import directly into your GIS workflow

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

