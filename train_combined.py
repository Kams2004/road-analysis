from ultralytics import YOLO
import torch
import os
from email_notifier import TrainingNotifier

def train_combined():
    """
    Combined training: pothole + traffic signs + speed bump
    Optimized for: 24 CPU cores, 117GB RAM, CPU-only
    Dataset: ~33,000 train images across 6 classes
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")
    print(f"CPU cores available: {os.cpu_count()}")

    total_epochs = 150
    notifier = TrainingNotifier("Combined Road Detector (all classes)", total_epochs)
    notifier.on_train_start()

    # Use YOLOv8s (small) — better accuracy than nano, still trainable on CPU
    model = YOLO('yolov8s.pt')

    model.add_callback("on_train_epoch_end",
        lambda trainer: notifier.on_epoch_end(trainer.epoch + 1))

    results = model.train(
        data='data_combined.yaml',
        epochs=total_epochs,
        imgsz=640,

        # With 117GB RAM and 33k images, cache in RAM for much faster training
        cache='ram',

        # Large batch — server has plenty of RAM
        batch=32,

        # Use all 24 cores for data loading
        workers=16,

        device=device,

        # Optimizer
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        # Warmup
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Augmentation — strong, dataset is large enough to benefit
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,

        # Early stopping — generous patience for large dataset
        patience=30,

        # Save
        save=True,
        save_period=10,
        project='runs/train',
        name='combined_road_detector',
        exist_ok=True,

        pretrained=True,
        val=True,
        plots=True,
        verbose=True,
    )

    print(f"\n✅ Training complete!")
    print(f"Best model: {model.trainer.best}")
    print(f"Results: {model.trainer.save_dir}")
    notifier.on_train_end(str(model.trainer.best))
    return results

if __name__ == "__main__":
    train_combined()
