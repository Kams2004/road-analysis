from ultralytics import YOLO
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training speed bump model on: {device}")

model = YOLO('yolov8n.pt')

model.train(
    data='speed_bump.yaml',
    epochs=50,
    imgsz=416,
    batch=2,
    patience=10,
    device=device,
    optimizer='AdamW',
    lr0=0.001,
    workers=4,
    project='runs/train',
    name='speed_bump_detector',
    exist_ok=True,
    pretrained=True,
    cache=False,
    val=True,
    plots=True
)

print("✓ Speed bump model saved: runs/train/speed_bump_detector/weights/best.pt")
