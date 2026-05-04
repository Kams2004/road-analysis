# CPU Training Guide for Your Laptop

## Your System
- CPU: Intel Core i5-1120U (12 cores)
- RAM: 16GB
- GPU: None (integrated graphics)

## Optimized Configuration

### Changes Made:b
1. **Model**: YOLOv8n (Nano) - smallest, fastest models
2. **Batch Size**: 4 (prevents memory overflow)
3. **Image Size**: 416 (reduced from 640)
4. **Workers**: 2 (prevents CPU overload)
5. **Epochs**: 50 (reasonable for CPU)
6. **Cache**: Disabled (saves RAM)

### Training Command:
```bash
python train.py
```

### Expected Training Time:
- **~6-10 hours** for 50 epochs (depends on dataset size)
- Monitor with: `htop` or Task Manager

### Tips to Prevent Crashes:

1. **Close other applications** before training
2. **Run from terminal** instead of VS Code:
   ```bash
   cd /home/kamsu-perold/pothole-detection
   python train.py
   ```
3. **Monitor memory usage**:
   ```bash
   watch -n 1 free -h
   ```

4. **If still crashing**, reduce further:
   - Change `batch=4` to `batch=2`
   - Change `imgsz=416` to `imgsz=320`
   - Change `workers=2` to `workers=1`

### Performance Expectations:
- **Accuracy**: ~85-88% mAP50 (good for most use cases)
- **Speed**: ~30-50 FPS on CPU inference
- **Model Size**: ~6MB

### For Better Results:
Use Google Colab (free GPU):
1. Upload dataset to Google Drive
2. Use `train_gpu.py` in Colab
3. Train with YOLOv8m or YOLOv8l
4. Download trained model

## Alternative: Cloud Training
- **Google Colab**: Free GPU (T4)
- **Kaggle**: Free GPU/TPU
- **AWS EC2**: g4dn.xlarge (~$0.50/hour)
