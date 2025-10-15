# Image2Shape - User Guide

Instructions for drone image feature detection and georeferencing workflows.

## Workflow Overview

### System Architecture

Image2Shape provides a three-panel interface for drone image processing:

- **Left Panel**: Multi-folder dataset management, GPS map visualization, and metadata summary
- **Right Panel**: Annotation tools with ML prediction overlay
- **Integrated Controls**: Filtering and batch processing controls

### 1. System Initialization

```bash
conda activate image2shape
python main.py
```

**System Requirements:**
- Available memory (4GB minimum, 8GB recommended)
- SSD storage recommended for large datasets

**Getting Help:**
- Access this User Guide anytime via **Help → User Guide**
- View README documentation via **Help → README**
- Both documents open in a built-in viewer window

### 2. Dataset Loading and Management

#### Single Folder Loading
1. **File → Load Folder** or click "Load Folder"
2. Select folder containing drone images with GPS metadata
3. Monitor preprocessing progress with progress dialog
4. Review metadata summary and validation results

#### Multi-Folder Workflow
1. **Load additional folders** using the same process
2. **Additive dataset management** - new images are added to existing project
3. **Combined metadata analysis** across all loaded folders
4. **Unified processing** of entire dataset

**Requirements:**
- **GPS Metadata**: RTK preferred, standard GPS acceptable
- **Image Formats**: JPG (recommended), PNG, TIFF supported
- **Dataset Size**: Supports large datasets
- **Folder Structure**: Organized by flight session or area

## Creating Annotations

### Annotation Types

- **Background** (cyan): Areas without features of interest
- **Features** (red): Areas containing features you want to detect

### How to Annotate

1. **Navigate images**: Use arrow keys (← →) or click image list
2. **Draw background**: Left-click + drag to create cyan rectangles
3. **Draw features**: Right-click + drag to create red rectangles
4. **Size matters**: Make rectangles 200x200 pixels or larger
5. **Multiple per image**: Add several annotations per image for better training

### Annotation Tips

- **Balance**: Create roughly equal amounts of feature and background annotations
- **Variety**: Annotate different lighting conditions and image areas
- **Quality over quantity**: 10-20 good annotations better than 50 poor ones
- **Clear examples**: Choose obvious feature and background areas

## ML Training

### Training Prerequisites

**Minimum Requirements:**
- 4 annotations total (2 feature + 2 background minimum)
- Balanced annotation distribution across image areas

**Recommended:**
- **10-20 annotations** for better model performance
- **Multiple lighting conditions** represented
- **Diverse image areas** annotated (center, edges, various altitudes)
- **Quality over quantity** - precise annotation boundaries

### Training Process

1. **Initiate Training**: Click "Train Model" button
2. **Algorithm Selection**: Choose between Random Forest or K-Nearest Neighbors
3. **Parameter Configuration**: Adjust algorithm-specific parameters if needed
4. **Pixel Extraction**: System extracts pixel samples from annotations automatically
5. **Feature Engineering**: 6-dimensional feature vectors (R,G,B,H,S,V) per pixel
6. **Model Training**: Selected algorithm trains with cross-validation
7. **Model Persistence**: Model and metadata saved to `data/models/`

### Training Results

**Results Dialog Shows:**
- **Sample Statistics**: Number of pixels extracted from annotations
- **Feature Engineering**: 6 features per pixel (RGB + HSV color space)
- **Model Performance**: Training accuracy and cross-validation scores
- **Algorithm Info**: Selected algorithm and parameters used
- **Model Metadata**: Training timestamp, sample distribution

**Quality Indicators:**
- **Training Accuracy**: Higher values indicate good annotation quality
- **Cross-Validation**: Indicates model generalization capability
- **Sample Balance**: Even distribution between feature/background classes

## Using Predictions

### Enable Predictions

1. Check **"Show Predictions"** checkbox
2. Model loads automatically if available
3. Predictions appear as colored dots on current image

### Prediction Display

- **Red dots**: Detected features
- **Dot size**: Indicates confidence level (larger = higher confidence)
- **Real-time**: Updates automatically when changing images

### Filter Controls

**Confidence Slider (0.5-0.95)**:
- Higher values = fewer, more confident predictions
- Lower values = more predictions, some false positives
- Start with 0.7, adjust based on results

**Min Size Slider (1-10)**:
- Minimum cluster size for grouped detections
- Higher values = remove isolated single pixels
- Recommended: 3-5 for most cases

### Recommended Settings

**High Precision** (fewer false positives):
- Confidence: 0.8-0.9
- Min Size: 4-6

**High Recall** (catch more features):
- Confidence: 0.6-0.7
- Min Size: 2-3

## Navigation and Workflow

### Keyboard Shortcuts

- **← →**: Previous/next image
- **Ctrl+O**: Open folder

### Efficient Workflow

1. **Load folder** and review metadata summary
2. **Quick annotation**: Use arrow keys to rapidly switch between images
3. **Annotate 5-10 images** with clear examples
4. **Train model** and check accuracy
5. **Test predictions** on different images
6. **Add more annotations** if accuracy is low
7. **Retrain** and repeat until satisfied

### Quality Control

- **Check predictions** on images you didn't annotate
- **Look for patterns** in false positives/negatives
- **Add corrections** by annotating problem areas
- **Retrain** to improve model

## Troubleshooting

### Common Issues

**"No GPS metadata found"**:
- Ensure images are from GPS-enabled drone
- Check EXIF data with photo viewer
- Try different image format (JPG usually best)

**"Training failed"**:
- Need minimum 4 annotations
- Check annotations are visible (colored rectangles)
- Ensure both feature and background annotations exist

**"No predictions visible"**:
- Check "Show Predictions" is enabled
- Lower confidence threshold
- Verify model trained successfully

**"Too many false positives"**:
- Increase confidence threshold (0.8+)
- Increase minimum cluster size (4+)
- Add more background annotations and retrain

**"Missing real features"**:
- Decrease confidence threshold (0.6-0.7)
- Add more feature annotations and retrain
- Check feature annotations cover variety of examples

### Performance Tips

- **Close other applications** to free memory
- **Use smaller image batches** if memory limited
- **Restart application** if it becomes slow
- **Clear annotations** and start fresh if needed

## File Management

### Automatic Saves

- **Annotations**: Saved automatically as JSON files in `data/annotations/`
- **Models**: Saved automatically in `data/models/`
- **Cache**: Display images cached for fast navigation

### Manual Management

- **Clear current image**: Remove annotations from current image only
- **Clear all**: Remove all annotations (use carefully!)
- **Model files**: Can be deleted to force retraining

## Export and Batch Processing

### Export Options

Once you have a trained model with satisfactory predictions:

#### 1. Export Process
1. **Verify Settings**: Confirm confidence threshold and clustering parameters
2. **Initiate Export**: Click "Export Results" button
3. **Select Export Type**: Choose from point features, buffered polygons, or combined data
4. **Select Output Location**: Choose shapefile save location
5. **Monitor Progress**: Processing status with progress tracking
6. **Review Results**: Check processing statistics and output files

#### 2. Export Types
- **Point Features**: Individual detection coordinates with confidence scores
- **Buffered Polygons**: Feature areas with configurable buffer size
- **Combined Export**: Drone positions, features, and image footprints
- **PDF Reports**: Optional processing statistics and metadata

#### 3. Output Integration
1. **QGIS Import**: Direct import of generated shapefiles
2. **Coordinate System**: WGS84 with longitude compression correction
3. **Attribute Analysis**: Confidence scores, image sources, detection metadata
4. **Spatial Analysis**: Use standard GIS tools for further analysis

### Quality Considerations

**Accuracy Characteristics:**
- **Positioning**: Sub-meter accuracy with longitude compression correction
- **Classification**: Performance depends on training data quality and filtering settings
- **Detection**: Pixel-level annotation accuracy

**Validation Workflow:**
- **Ground Truth Comparison**: Compare results with known reference points
- **Quality Control**: Review confidence score distributions and spatial patterns
- **Parameter Adjustment**: Modify filtering settings based on results