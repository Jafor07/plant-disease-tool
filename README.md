# Plant Disease YOLO Dataset Preparation Tool

A Streamlit-based application for preparing plant disease detection datasets in YOLO format. This tool enables manual annotation of diseased plant regions with a free-draw canvas and exports processed images, binary masks, YOLO labels, and metadata.

## Features

### 1. **Dataset Loader Mode**
- Load all images from `raw_images/` directory with visual indicators (✅ processed / ⬜ unprocessed)
- Navigate between images using:
  - Radio button selection
  - Previous/Next buttons
  - Progress counter showing annotated images
- Real-time progress tracking

### 2. **Drawing Canvas & Annotation**
- Free-draw canvas overlay on the selected image
- White brush strokes mark diseased regions
- Brush size adjustment (1-100 pixels)
- Toolbar for undo, redo, and reset operations
- Background image remains separate from drawing data

### 3. **Background Effects**
Choose between background processing methods while preserving the annotated disease region:
- **Gaussian Blur**: Adjustable kernel size (3-51, odd values only)
- **Grayscale**: Convert background to monochrome
- Real-time preview of processed images

### 4. **Dataset Export**
On "Save & Next", the application writes:
- **Processed Image**: `processed_images/{filename}` - Original resolution preserved, background effect applied
- **Binary Mask**: `masks/{stem}.png` - White (255) for disease regions, black (0) for background
- **YOLO Labels**: `labels/{stem}.txt` - Normalized bounding boxes derived from connected components
  - Format: `class x_center y_center width height` (class 0 for disease)
- **Metadata**: `metadata/{stem}.json` - Effect type, blur strength, resolution, and timestamp
- Automatic advance to next image

### 5. **Upload & Annotate Mode**
Alternative workflow for single-image annotation:
- File uploader for PNG or JPEG images
- Same canvas, effect, and brush controls as Dataset Loader
- Save annotations with automatic file processing
- Download buttons for:
  - Processed image
  - Binary mask
  - YOLO labels file

## Installation

### Requirements
```
Python 3.8+
Streamlit >= 1.27
streamlit-drawable-canvas-fix >= 0.8.0
OpenCV (headless) >= 4.5
NumPy >= 1.20
Pillow >= 8.0
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Jafor07/plant-disease-tool.git
cd plant-disease-tool
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the Application

```bash
streamlit run app.py
```

The app will launch in your default browser at `http://localhost:8501`.

### Workflow: Dataset Loader Mode

1. **Select Mode**: Choose "Dataset Loader" in the sidebar
2. **Select Image**: Pick an image from the list or use Previous/Next buttons
3. **Configure Settings**:
   - Choose background effect (Gaussian Blur or Grayscale)
   - Set blur strength (for blur mode)
   - Adjust brush size
4. **Annotate**: Draw white strokes over diseased regions
5. **Review**: Check the processed preview in real-time
6. **Save**: Click "Save & Next" to:
   - Write all outputs to disk
   - Advance to next image
   - Automatically refresh UI

### Workflow: Upload & Annotate Mode

1. **Select Mode**: Choose "Upload & Annotate" in the sidebar
2. **Upload Image**: Select a PNG or JPEG file from your computer
3. **Configure Settings**: Same as Dataset Loader mode
4. **Annotate**: Draw white strokes over diseased regions
5. **Save**: Click "Save & Generate Downloads" to:
   - Write outputs to disk
   - Generate download buttons for processed image, mask, and labels
6. **Download**: Retrieve individual files as needed

## Directory Structure

```
project-root/
├── app.py                      # Main application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── raw_images/                 # Input images (user-provided)
├── processed_images/           # Output processed images
├── masks/                      # Binary PNG masks
├── labels/                     # YOLO format label files
└── metadata/                   # JSON metadata for each image
```

### Automatic Directory Creation
The application automatically creates all required directories on first run.

## Output Formats

### Binary Mask
- Format: PNG (8-bit grayscale)
- White (255): Disease regions
- Black (0): Background regions
- Dimensions: Same as original image (no resizing)

### YOLO Labels
- Format: Text file with one annotation per line
- Format: `class x_center y_center width height`
- Coordinates: Normalized (0.0-1.0) by image width/height
- Class: Always 0 for disease regions
- Bounding boxes derived from connected components of mask

Example:
```
0 0.5123 0.6234 0.1234 0.2345
0 0.7123 0.4234 0.1534 0.1945
```

### Metadata JSON
Stores processing information for reproducibility:
```json
{
    "blur_type": "blur",
    "blur_strength": 15,
    "image_resolution": {
        "width": 1920,
        "height": 1080
    },
    "timestamp": "2026-02-10T12:34:56.789123"
}
```

## Technical Details

### Image Processing
- **Library**: OpenCV (cv2) from headless package
- **Color Space**: Internal BGR, RGB for display
- **Resolution**: Original resolution preserved throughout processing
- **Deterministic Output**: No automatic resizing or lossy transformations

### Canvas Data
- Uses `streamlit-drawable-canvas-fix` component
- Drawing data returned as RGBA array (H, W, 4)
- Background image not included in drawing data
- Alpha channel indicates user-annotated pixels

### Mask Generation
- Alpha channel > 0 indicates drawn pixels
- Binary conversion: alpha > 0 → 255, otherwise 0
- No anti-aliasing or interpolation to maintain annotation integrity

## Known Limitations

- Maximum image size depends on available system memory
- Very large images may have slower canvas rendering
- Brush size applies uniform width across all strokes
- Bounding boxes are axis-aligned (AABB) from contour bounds

## Troubleshooting

### Issue: "No images found in 'raw_images/'"
- **Solution**: Add image files (PNG, JPG, JPEG, BMP, TIF, TIFF) to the `raw_images/` directory

### Issue: Canvas not responding
- **Solution**: Reload the page or clear browser cache; ensure JavaScript is enabled

### Issue: Very blurry output
- **Solution**: Reduce blur strength slider value (try 5 instead of 15)

### Issue: Missing bounding boxes in labels
- **Solution**: Ensure disease regions are drawn clearly; very small marks may not generate detectable contours

## Example Workflow

1. **Prepare Data**:
   ```bash
   mkdir raw_images
   cp plant_images/*.jpg raw_images/
   ```

2. **Run Application**:
   ```bash
   streamlit run app.py
   ```

3. **Annotate Images**:
   - Select Dataset Loader mode
   - Configure blur strength (e.g., 15)
   - Draw disease regions
   - Click Save & Next

4. **Export Dataset**:
   - Annotated images in `processed_images/`
   - Masks in `masks/`
   - Labels in `labels/`
   - Metadata in `metadata/`

5. **Use with YOLO**:
   ```python
   from ultralytics import YOLO
   
   # Train with annotated dataset
   model = YOLO('yolov8n.pt')
   results = model.train(data='dataset.yaml', epochs=100)
   ```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | ≥1.27 | Web UI framework |
| streamlit-drawable-canvas-fix | ≥0.8.0 | Drawing canvas component |
| opencv-python-headless | Latest | Image processing |
| numpy | ≥1.20 | Numerical operations |
| pillow | ≥8.0 | Image I/O |

## Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Commit changes with clear messages
4. Submit a pull request

## License

[Specify your license here]

## Author

Created as a plant disease detection dataset preparation tool.

## Support

For issues, feature requests, or questions:
1. Check existing GitHub issues
2. Create a new issue with detailed description
3. Include screenshots or error messages if applicable
