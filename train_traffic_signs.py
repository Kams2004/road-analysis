from ultralytics import YOLO
import torch

def train_traffic_signs():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")

    model = YOLO('yolov8n.pt')

    results = model.train(
        data='traffic_signs_augmented/data.yaml',
        epochs=80,               # More epochs for better generalization
        imgsz=640,               # Larger size — signs are small objects
        batch=4,
        patience=20,
        device=device,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Stronger augmentation to handle sign variations
        hsv_h=0.03,
        hsv_s=0.7,
        hsv_v=0.5,
        degrees=15,              # Signs can be at angles
        translate=0.1,
        scale=0.5,               # Signs appear at various distances
        fliplr=0.5,
        mosaic=1.0,              # Full mosaic for more context
        mixup=0.1,
        copy_paste=0.1,          # Helps with rare classes like crosswalk
        workers=2,
        cache=False,
        project='runs/train',
        name='traffic_sign_detector',
        exist_ok=True,
        pretrained=True,
        val=True,
        plots=True,
        save_period=10
    )

    print(f"\nBest model: {model.trainer.best}")
    return results

if __name__ == "__main__":
    train_traffic_signs()
