"""
Train YOLOv8 on the new traffic signs dataset.
Detects and classifies 78 road sign types including speed limits,
pedestrian crossings, stop signs, traffic lights, etc.
"""
from ultralytics import YOLO
import yaml, os

DATA_YAML = "/home/kamsu-perold/pothole-detection/traffic_signs_clean.yaml"
MODEL     = "yolov8s.pt"   # small model — good balance of speed & accuracy
PROJECT   = "/home/kamsu-perold/pothole-detection/runs/traffic_signs"
NAME      = "train_v1"

def verify_dataset():
    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)
    for split, path in [("train", cfg["train"]), ("val", cfg["val"])]:
        n = len(os.listdir(path)) if os.path.isdir(path) else 0
        print(f"  {split}: {n} images at {path}")
    print(f"  classes: {cfg['nc']}")

def train():
    print("=== Dataset ===")
    verify_dataset()

    model = YOLO(MODEL)
    results = model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=320,            # reduced from 512 — saves RAM & speeds up CPU
        batch=4,              # small batch — only ~2.2 GB RAM available
        workers=4,            # 4 of 12 threads for data loading (leave headroom)
        patience=10,          # early stopping
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        augment=True,
        mosaic=0.5,           # reduced mosaic — less memory
        mixup=0.0,            # disabled — saves memory
        degrees=10,
        fliplr=0.5,
        cache=False,           # do NOT cache — not enough RAM
        device="cpu",
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True,
    )
    print(f"\n=== Training complete ===")
    print(f"Best model: {PROJECT}/{NAME}/weights/best.pt")
    return results

if __name__ == "__main__":
    train()
