from ultralytics import YOLO
import os
from email_notifier import TrainingNotifier

def train():
    total_epochs = 100
    notifier = TrainingNotifier("Pothole Model", total_epochs)
    notifier.on_train_start()

    model = YOLO('yolov8s.pt')
    model.add_callback("on_train_epoch_end",
        lambda trainer: notifier.on_epoch_end(trainer.epoch + 1))

    results = model.train(
        data='/home/deploy/road-analysis/data.yaml',
        epochs=total_epochs,
        imgsz=640,
        batch=16,
        workers=6,
        cache='ram',
        device='cpu',
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5,
        translate=0.1,
        scale=0.4,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        patience=30,
        save=True,
        save_period=10,
        project='runs/train',
        name='model_pothole',
        exist_ok=True,
        pretrained=True,
        val=True,
        plots=True,
        verbose=True,
    )

    print(f"\n✅ Pothole model done! Best: {model.trainer.best}")
    notifier.on_train_end(str(model.trainer.best))
    return results

if __name__ == "__main__":
    train()
