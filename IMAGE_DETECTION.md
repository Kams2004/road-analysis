# Image Detection Guide

## Your Trained Model
- Location: `/home/kamsu-perold/pothole-detection/runs/detect/runs/train/pothole_detector/weights/best.pt`
- Accuracy: 74.2% mAP50
- Precision: 73.4%
- Recall: 67.0%

## Usage

### Basic Detection:
```bash
python detect_image.py --model runs/detect/runs/train/pothole_detector/weights/best.pt --image path/to/your/image.jpg
```

### With Custom Confidence Threshold:
```bash
python detect_image.py --model runs/detect/runs/train/pothole_detector/weights/best.pt --image path/to/your/image.jpg --conf 0.5
```

### Specify Output Path:
```bash
python detect_image.py --model runs/detect/runs/train/pothole_detector/weights/best.pt --image path/to/your/image.jpg --output result.jpg
```

## Examples

### Example 1: Test on validation image
```bash
python detect_image.py --model runs/detect/runs/train/pothole_detector/weights/best.pt --image valid/images/image_001.jpg
```

### Example 2: Higher confidence (fewer false positives)
```bash
python detect_image.py --model runs/detect/runs/train/pothole_detector/weights/best.pt --image myimage.jpg --conf 0.6
```

### Example 3: Lower confidence (catch more potholes)
```bash
python detect_image.py --model runs/detect/runs/train/pothole_detector/weights/best.pt --image myimage.jpg --conf 0.15
```

## Output

The script will:
1. **Print text results** showing:
   - Total number of potholes detected
   - Confidence score for each detection
   - Location coordinates (x1, y1, x2, y2)
   - Size of each pothole in pixels

2. **Save visualized image** with:
   - Bounding boxes around detected potholes
   - Confidence scores displayed
   - Default name: `detected_[original_name].jpg`

## Confidence Threshold Guide
- **0.15-0.25**: High recall (finds more potholes, more false positives)
- **0.25-0.40**: Balanced (default)
- **0.40-0.60**: High precision (fewer false positives, might miss some)
- **0.60+**: Very strict (only very confident detections)
