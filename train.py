from ultralytics import YOLO
import torch

def train_pothole_detector():
    """Train YOLOv8 model for pothole detection with optimized settings"""
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")
    
    # Load YOLOv8 model (optimized for CPU training)
    model = YOLO('yolov8n.pt')  # Nano model for CPU training
    
    # Train with CPU-optimized hyperparameters
    results = model.train(
        data='data.yaml',
        epochs=50,               # Reduced for CPU
        imgsz=416,               # Smaller image size for CPU
        batch=4,                 # Small batch for 16GB RAM
        patience=15,
        save=True,
        device='cpu',
        
        # Optimization parameters
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=2,
        warmup_momentum=0.8,
        
        # Reduced augmentation for faster training
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        mosaic=0.5,              # Reduced mosaic
        mixup=0.0,               # Disabled mixup
        
        # CPU-optimized settings
        workers=2,               # Reduced workers
        project='runs/train',
        name='pothole_detector',
        exist_ok=True,
        pretrained=True,
        verbose=True,
        cache=False,             # Don't cache to save RAM
        
        # Validation settings
        val=True,
        plots=True,
        save_period=10
    )
    
    print("\n=== Training Complete ===")
    print(f"Best model saved at: {model.trainer.best}")
    print(f"Results saved at: {model.trainer.save_dir}")
    
    return results

if __name__ == "__main__":
    train_pothole_detector()
