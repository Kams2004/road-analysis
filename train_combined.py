from ultralytics import YOLO
import torch

def train_combined_detector():
    """Train YOLOv8 for pothole + traffic sign detection"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")
    
    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data='data_combined.yaml',
        epochs=100,
        imgsz=640,
        batch=8 if device == 'cuda' else 4,
        patience=20,
        device=device,
        
        optimizer='AdamW',
        lr0=0.001,
        
        workers=4 if device == 'cuda' else 2,
        project='runs/train',
        name='combined_detector',
        exist_ok=True,
        pretrained=True,
        cache=False,
        
        val=True,
        plots=True,
        save_period=10
    )
    
    print(f"\n✓ Training complete! Best model: {model.trainer.best}")
    return results

if __name__ == "__main__":
    train_combined_detector()
