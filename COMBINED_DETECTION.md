# Combined Pothole & Traffic Sign Detection

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Roboflow API Key
1. Go to https://roboflow.com
2. Sign up/login
3. Go to Settings → API Keys
4. Copy your API key

### 3. Download & Merge Datasets
```bash
# Edit download_and_merge_datasets.py and add your API key
python download_and_merge_datasets.py
```

This will:
- Download traffic sign dataset
- Merge with existing pothole dataset
- Create `data_combined.yaml` config

### 4. Train Combined Model
```bash
python train_combined.py
```

### 5. Run Detection
```bash
# On image
python detect_combined.py path/to/image.jpg

# On video
python detect_combined.py path/to/video.mp4

# On webcam
python detect_combined.py 0
```

## Model Output
The model will detect:
- **Class 0**: Pothole
- **Class 1+**: Traffic signs (stop, yield, speed limit, etc.)

## Files Created
- `download_and_merge_datasets.py` - Dataset download & merge
- `train_combined.py` - Training script
- `detect_combined.py` - Detection script
- `data_combined.yaml` - Combined dataset config
