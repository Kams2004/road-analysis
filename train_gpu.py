from ultralytics import YOLO
import torch

def train_pothole_detector():
    """Train YOLOv8 model - GPU optimized version"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")
    
    if device == 'cpu':
        print("WARNING: No GPU detected. Use train.py instead for CPU-optimized training.")
        return
    
    model = YOLO('yolov8m.pt')  # Medium model for GPU
    
    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        device=device,
        optimizer='AdamW',
        lr0=0.001,
        workers=4,
        project='runs/train',
        name='pothole_detector_gpu',
        exist_ok=True,
        cache=True
    )
    
    print(f"\nBest model: {model.trainer.best}")
    return results

if __name__ == "__main__":
    train_pothole_detector()
