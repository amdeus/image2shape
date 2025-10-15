# Image2Shape 🚁📍

> **Direct Georeferencing for DJI P1 Drone Images - On-Site Processing Without Cloud Dependencies**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DJI P1](https://img.shields.io/badge/Camera-DJI%20P1-orange.svg)](https://www.dji.com/zenmuse-p1)

## 🎯 Overview

Image2Shape is a specialized tool for **on-site drone image processing** that transforms DJI P1 aerial imagery into accurate geospatial data through direct georeferencing. Designed for field operations where fast, local processing is essential without relying on cloud services or large data transfers.

### 🔥 Key Advantages
- **🏃‍♂️ Fast On-Site Processing**: Get results in minutes, not hours
- **📡 No Cloud Dependency**: Process everything locally on your device
- **💾 Minimal Data Transfer**: Keep sensitive data secure and local
- **🎯 Direct Georeferencing**: Leverage DJI P1's high-precision GNSS capabilities
- **⚡ Low Resource Requirements**: Optimized for field laptops and mobile workstations

## 🛸 DJI P1 Integration

This tool is **specifically optimized for DJI P1 camera systems** and automatically:

- **📋 Extracts EXIF Metadata**: GPS coordinates, altitude, camera parameters, and flight telemetry
- **🎯 Utilizes RTK/PPK Data**: High-precision positioning for centimeter-level accuracy
- **📐 Applies Camera Calibration**: Automatic lens distortion correction using P1 specifications
- **🧭 Processes IMU Data**: Pitch, roll, yaw information for precise image orientation

### Supported P1 Configurations
- ✅ DJI P1 with RTK base station
- ✅ DJI P1 with PPK post-processing
- ✅ Standard GPS mode (meter-level accuracy)

## 🔄 Workflow Pipeline

```
DJI P1 Images → Metadata Extraction → Feature Detection → ML Processing → Shapefile Export
     ↓               ↓                    ↓               ↓              ↓
   📸 JPEG/RAW    🗺️ GPS+IMU Data    🎯 Annotations   🧠 Training    📁 GIS Ready
```

### 1. **Input Processing**
- Load DJI P1 images with embedded GNSS data
- Extract precise georeferencing parameters from EXIF metadata
- Apply camera calibration and distortion correction

### 2. **Feature Annotation & Detection**
- Interactive annotation tools for training data creation
- Automated feature detection using trained ML models
- Support for multiple feature types (buildings, vegetation, infrastructure)

### 3. **Machine Learning Pipeline**
- **Training**: Random Forest and KNN algorithms optimized for drone imagery
- **Inference**: Fast, lightweight models for real-time processing
- **Validation**: Built-in accuracy assessment tools

### 4. **Direct Georeferencing**
- Transform image coordinates to real-world coordinates using DJI P1 metadata
- Apply WGS84 geographic projection with longitude compression correction
- Generate accurate spatial references without ground control points

### 5. **Shapefile Export**
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

1. **📁 Load DJI P1 Images**
   - Click "Load Images" and select your DJI P1 image folder
   - Metadata will be automatically extracted and validated

2. **🎯 Create Training Annotations**
   - Draw rectangles on features (red) and background (cyan)
   - Build training dataset for your specific use case

3. **🧠 Train ML Model**
   - Choose Random Forest or KNN algorithm
   - Train on your annotated samples (~30 seconds)

4. **🔍 Process & Review**
   - Run feature detection on all images
   - Adjust confidence and clustering parameters

5. **📊 Export Results**
   - Generate shapefiles with precise coordinates
   - Import directly into your GIS workflow

## 🎯 Use Cases

### **Survey & Mapping**
- Rapid site surveys without extensive post-processing
- Infrastructure inspection and monitoring
- Environmental monitoring and change detection

### **Emergency Response**
- Disaster assessment with immediate results
- Search and rescue operations
- Damage assessment in remote areas

### **Construction & Agriculture**
- Progress monitoring on construction sites
- Crop health assessment
- Precision agriculture applications

## 📊 Performance Specifications

- **Processing Speed**: ~2-5 seconds per image on standard laptop
- **Accuracy**: Sub-meter precision with RTK-enabled DJI P1
- **Memory Usage**: < 2GB RAM for typical survey missions
- **Training Time**: < 30 seconds for standard datasets
- **Storage**: Minimal - only shapefiles and metadata retained

## 🛠️ Technical Features

### Input Requirements
- **Camera**: DJI P1 with EXIF metadata
- **Formats**: JPEG, DNG/RAW with embedded GPS
- **Metadata**: GPS coordinates, altitude, yaw, focal length

### Output Formats
- **Primary**: ESRI Shapefile (.shp, .shx, .dbf, .prj)
- **Optional**: GeoJSON, CSV with coordinates
- **Reports**: PDF processing summaries

### Coordinate Systems
- **Primary**: WGS84 Geographic (EPSG:4326)
- **Support**: UTM projections (auto-detected)
- **Custom**: User-defined CRS support

## 📁 Project Structure

```
image2shape/
├── main.py                 # Main application entry point
├── modules/
│   ├── image_processing.py # DJI P1 metadata extraction
│   ├── ml_pipeline.py      # ML training and inference
│   ├── georeferencing.py   # Direct georeferencing algorithms
│   └── shapefile_export.py # GIS export functionality
├── doc/
│   ├── README.md          # Technical documentation
│   ├── requirements.txt   # Python dependencies
│   └── USER_GUIDE.md      # Detailed user guide
└── data/                  # Sample data and models
```

## 🔧 Architecture Highlights

- **MetadataProcessor**: Extracts DJI P1 EXIF data using exiftool
- **MLAlgorithmFactory**: Multi-algorithm framework (Random Forest, KNN)
- **SelfContainedGeoreferencer**: Direct coordinate transformation
- **BatchGeoreferencer**: Vectorized operations for performance
- **ShapefileExporter**: Multi-format GIS export capabilities

## 📖 Documentation

- **[Technical Guide](doc/README.md)**: Detailed technical specifications
- **[User Guide](doc/USER_GUIDE.md)**: Step-by-step usage instructions
- **[Requirements](doc/requirements.txt)**: Complete dependency list

## 🤝 Contributing

We welcome contributions! Areas of interest:
- Additional ML algorithms for feature detection
- Support for other drone camera systems
- Performance optimizations
- New export formats

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Why Image2Shape?

**Built for professionals who need fast, accurate, and local geospatial processing**

- 🚫 **No Cloud Lock-in**: Keep your data secure and processing local
- ⚡ **Field-Ready**: Get actionable results while still on-site
- 🎯 **DJI P1 Optimized**: Leverage the full potential of your high-end drone camera
- 💰 **Cost-Effective**: No per-image processing fees or subscription costs
- 🔒 **Data Security**: Sensitive survey data never leaves your device

---

Made with ❤️ for field professionals who demand precision and speed 🌍