# Pothole Detection System - Training & Deployment Guide

## Overview
This system uses YOLOv8 for real-time pothole detection from video streams.

## Step-by-Step Procedure

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
python train.py
```

**Training Tips for Maximum Accuracy:**
- **Model Selection**: The script uses YOLOv8x (largest, most accurate). For faster training, change to:
  - `yolov8n.pt` - Nano (fastest, least accurate)
  - `yolov8s.pt` - Small
  - `yolov8m.pt` - Medium
  - `yolov8l.pt` - Large
  - `yolov8x.pt` - Extra Large (best accuracy)

- **Batch Size**: Adjust based on GPU memory:
  - 16GB GPU: batch=16
  - 8GB GPU: batch=8
  - 4GB GPU: batch=4

- **Epochs**: Default is 100. Monitor validation metrics and adjust:
  - If overfitting: reduce epochs or increase patience
  - If underfitting: increase epochs

- **Training Output**: 
  - Best model: `runs/train/pothole_detector/weights/best.pt`
  - Last model: `runs/train/pothole_detector/weights/last.pt`
  - Metrics & plots: `runs/train/pothole_detector/`

### Step 3: Evaluate Model
```bash
python evaluate.py --model runs/train/pothole_detector/weights/best.pt
```

### Step 4: Test on Video Stream

**Option A: Video File**
```bash
python detect_video.py --model runs/train/pothole_detector/weights/best.pt --source path/to/video.mp4
```

**Option B: Webcam**
```bash
python detect_video.py --model runs/train/pothole_detector/weights/best.pt --source 0
```

**Option C: RTSP Stream**
```bash
python detect_video.py --model runs/train/pothole_detector/weights/best.pt --source rtsp://username:password@ip:port/stream
```

**Additional Options:**
- `--conf 0.5`: Set confidence threshold (default: 0.25)
- `--no-save`: Don't save output video

## Configuration for Maximum Accuracy

### 1. Data Quality
- Ensure diverse lighting conditions in dataset
- Include various pothole sizes and types
- Balance dataset if needed

### 2. Hyperparameter Tuning
Edit `train.py` to adjust:
- `epochs`: More epochs = better learning (watch for overfitting)
- `imgsz`: Larger images = better small object detection (640, 800, 1024)
- `batch`: Larger batch = more stable training
- `lr0`: Learning rate (0.001 is good starting point)

### 3. Data Augmentation
Already configured in `train.py`:
- Mosaic augmentation
- Mixup augmentation
- HSV color augmentation
- Geometric transformations

### 4. Post-Training Optimization
- Use `best.pt` for highest accuracy
- Adjust confidence threshold based on use case:
  - High precision needed: conf=0.5-0.7
  - High recall needed: conf=0.2-0.3
  - Balanced: conf=0.25-0.4

## Monitoring Training

Watch these metrics in real-time:
- **mAP50**: Mean Average Precision at IoU 0.5 (higher is better)
- **mAP50-95**: mAP averaged over IoU 0.5-0.95 (higher is better)
- **Precision**: Accuracy of positive predictions
- **Recall**: Ability to find all potholes
- **Loss**: Should decrease over time

## Troubleshooting

**Out of Memory Error:**
- Reduce batch size
- Use smaller model (yolov8m or yolov8s)
- Reduce image size

**Low Accuracy:**
- Train longer (more epochs)
- Use larger model (yolov8x)
- Check dataset quality
- Increase data augmentation

**Slow Inference:**
- Use smaller model
- Reduce image size
- Use GPU if available

## Output Files

After training:
- `runs/train/pothole_detector/weights/best.pt` - Best model weights
- `runs/train/pothole_detector/results.png` - Training metrics
- `runs/train/pothole_detector/confusion_matrix.png` - Confusion matrix
- `output_detection.mp4` - Annotated video output

## Performance Expectations

| Model | mAP50 | Speed (FPS) | GPU Memory |
|-------|-------|-------------|------------|
| YOLOv8n | ~0.85 | 100+ | 2GB |
| YOLOv8s | ~0.88 | 80+ | 3GB |
| YOLOv8m | ~0.90 | 60+ | 5GB |
| YOLOv8l | ~0.92 | 40+ | 8GB |
| YOLOv8x | ~0.94 | 30+ | 12GB |

*Actual results depend on dataset quality and training configuration*
