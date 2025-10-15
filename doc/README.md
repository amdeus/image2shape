# Image2Shape - Drone Image Feature Detection

Image2Shape is a Python application for automated feature detection in drone imagery using machine learning and photogrammetric georeferencing. The system processes drone images to detect features and exports results as georeferenced shapefiles for GIS analysis.

## Features

### Machine Learning
- **Multi-algorithm framework**: Random Forest and K-Nearest Neighbors with factory pattern
- **Feature extraction**: RGB+HSV values (6 features per pixel)
- **Training data**: Extracted from manual annotation rectangles
- **Performance**: Training typically under 30 seconds, prediction 1-2 seconds per image

### Georeferencing
- **Coordinate system**: WGS84 (EPSG:4326)
- **Input data**: GPS coordinates, flight altitude, drone yaw, focal length from EXIF
- **Sensor model**: DJI P1 specifications (4.4μm pixel size)
- **Corrections**: Yaw rotation, longitude compression correction

### Interface
- **GUI framework**: Tkinter three-panel layout
- **Image management**: Multi-folder loading with metadata caching
- **Annotation system**: Rectangle-based training data creation
- **Real-time feedback**: Live prediction display with filtering controls
- **Built-in documentation**: README and User Guide accessible from Help menu

### Export Options
- **Point features**: Individual detection coordinates with confidence scores
- **Buffered polygons**: Feature areas with configurable buffer size
- **Combined export**: Drone positions, features, and image footprints
- **File formats**: ESRI Shapefile with QGIS compatibility
- **Reports**: Optional PDF reports with processing statistics

## Technical Specifications

### Accuracy
- **Positioning**: Sub-meter precision with longitude compression correction
- **ML classification**: Performance depends on training data quality and filtering settings
- **Feature detection**: Pixel-level annotation accuracy

### Performance
- **ML training**: Typically under 30 seconds for standard datasets
- **Prediction**: 1-2 seconds per image depending on size and sampling density
- **Batch processing**: Parallel processing with ThreadPoolExecutor
- **Memory usage**: Moderate memory requirements for typical operations
- **Export speed**: Efficient shapefile generation with optimized polygon processing

### Requirements
- **Python**: 3.8+ with conda environment recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: SSD recommended for large datasets
- **External dependency**: exiftool for metadata extraction
- **Operating systems**: Linux (primary), Windows (supported)

## Installation

### Prerequisites
```bash
# Install exiftool (required for metadata extraction)
sudo apt install exiftool  # Ubuntu/Debian
brew install exiftool      # macOS
```

### Setup
```bash
# Clone repository
git clone https://github.com/your-org/image2shape.git
cd image2shape

# Create conda environment
conda create -n image2shape python=3.8
conda activate image2shape

# Install dependencies
pip install -r doc/requirements.txt
```

## Usage

### Basic Workflow
1. **Load images**: Select folder(s) containing drone images with GPS metadata
2. **Create annotations**: Draw rectangles on features (red) and background (cyan)
3. **Train model**: Select algorithm and train classifier from pixel samples
4. **Review predictions**: Adjust confidence and clustering parameters
5. **Export results**: Generate shapefiles for GIS analysis

### Command Line
```bash
# Run application
python main.py
```

## Input Requirements

### Drone Images
- **Formats**: JPG, PNG, TIFF with EXIF metadata
- **GPS data**: Latitude, longitude, altitude in EXIF
- **Flight parameters**: Yaw angle, focal length required for georeferencing
- **Quality**: RTK GPS preferred, standard GPS acceptable

### Metadata Fields
- **GPS coordinates**: GPSLatitude, GPSLongitude (decimal degrees)
- **Altitude**: RelativeAltitude (height above ground)
- **Orientation**: FlightYawDegree (drone heading)
- **Camera**: FocalLength, ImageWidth, ImageHeight
- **Optional**: RTK quality indicators

## Architecture

### Core Components
- **MetadataProcessor**: EXIF extraction using exiftool
- **MLAlgorithmFactory**: Multi-algorithm framework with factory pattern
- **FeatureExtractor**: Pixel-level feature extraction (RGB+HSV)
- **SelfContainedGeoreferencer**: Photogrammetric coordinate transformation
- **BatchGeoreferencer**: Batch processing with vectorized operations
- **ShapefileExporter**: ESRI Shapefile generation with multiple export types

### Processing Pipeline
1. **Image loading**: Parallel metadata extraction with caching
2. **Annotation**: Rectangle-based training data creation
3. **Feature extraction**: Pixel sampling from annotation areas
4. **ML training**: Multi-algorithm training with cross-validation
5. **Prediction**: Grid-based pixel classification
6. **Georeferencing**: Coordinate transformation to WGS84
7. **Export**: Shapefile generation with attributes

## Dependencies

```python
# Core processing
opencv-python>=4.8.0    # Image processing
scikit-learn>=1.3.0     # Machine learning algorithms
numpy>=1.24.0           # Numerical operations
pillow>=10.0.0          # Image I/O
matplotlib>=3.7.0       # Plotting and visualization

# Geospatial
geopandas>=0.13.0       # Shapefile operations
shapely>=2.0.0          # Geometric operations
fiona>=1.9.0            # Geospatial file I/O
rtree>=1.0.0            # Spatial indexing

# Additional components
psutil>=5.8.0           # System monitoring
reportlab>=4.0.0        # PDF report generation
folium>=0.14.0          # Interactive maps
pandas>=1.5.0           # Data manipulation

# GUI (built-in)
tkinter                 # Python standard library
```

## Documentation

- **User Guide**: `doc/USER_GUIDE.md` - Detailed usage instructions
- **Architecture**: `.agent.md` - Technical implementation details
- **Requirements**: `doc/requirements.txt` - Python dependencies

## License

MIT License - see LICENSE file for details.