# Combined Detection: Potholes + Traffic Signs + Dos d'âne

## What Gets Detected

| Class              | Model              | Colour  |
|--------------------|--------------------|---------|
| Pothole            | pothole_detector   | 🔴 Red   |
| Traffic signs      | traffic_signs      | 🟢 Green |
| Dos d'âne (sign)   | traffic_signs      | 🟡 Yellow|
| Dos d'âne (road)   | speed_bump_detector| 🟠 Orange|

## Setup Steps

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download datasets (add your Roboflow API key first)
```bash
python download_traffic_signs.py   # traffic signs
python download_speed_bump.py      # physical speed bumps
```

### 3. Train all models
```bash
python train.py                    # pothole model
python train_traffic_signs.py      # traffic sign model
python train_speed_bump.py         # speed bump model
```

### 4. Run detection
```bash
python detect_dual.py image.jpg    # image
python detect_dual.py video.mp4    # video
python detect_dual.py 0            # webcam
```

## Notes
- Models are independent — you can retrain one without affecting others
- If a model is not yet trained, it is automatically skipped
- "Dos d'âne" is detected TWICE: once via its road sign, once visually on the road
