from ultralytics import YOLO
import cv2
import sys
import os

# ── Model paths ──────────────────────────────────────────────────────────────
POTHOLE_MODEL    = 'runs/train/pothole_detector/weights/best.pt'
TRAFFIC_MODEL    = 'runs/train/traffic_signs/weights/best.pt'
SPEED_BUMP_MODEL = 'runs/train/speed_bump_detector/weights/best.pt'

# ── Colours (BGR) ────────────────────────────────────────────────────────────
RED    = (0,   0,   255)   # potholes
GREEN  = (0,   255, 0  )   # traffic signs
ORANGE = (0,   165, 255)   # speed bumps (visual)
YELLOW = (0,   255, 255)   # speed bump sign detected by traffic model

FONT = cv2.FONT_HERSHEY_SIMPLEX

def draw_box(img, x1, y1, x2, y2, label, color, conf):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
    cv2.putText(img, text, (x1, y1 - 4), FONT, 0.5, (255, 255, 255), 1)

def is_speed_bump_sign(name: str) -> bool:
    """Check if a traffic sign class name refers to a dos d'âne / speed bump sign"""
    keywords = ['speed bump', 'dos d', 'dos-d', 'ralentisseur', 'hump', 'bump']
    return any(k in name.lower() for k in keywords)

def run_detection(source):
    # ── Load models ──────────────────────────────────────────────────────────
    models = {}
    for key, path in [('pothole', POTHOLE_MODEL),
                      ('traffic', TRAFFIC_MODEL),
                      ('speed_bump', SPEED_BUMP_MODEL)]:
        if os.path.exists(path):
            models[key] = YOLO(path)
            print(f"✓ Loaded {key} model")
        else:
            print(f"⚠ {key} model not found at {path}, skipping")

    if not models:
        print("✗ No models found. Train them first.")
        return

    # ── Video / image handling ────────────────────────────────────────────────
    is_image = isinstance(source, str) and source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    is_video = isinstance(source, str) and source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    is_webcam = source == 0 or source == '0'

    if is_image:
        img = cv2.imread(source)
        img = process_frame(img, models)
        out = 'output_combined.jpg'
        cv2.imwrite(out, img)
        print(f"✓ Saved: {out}")
        cv2.imshow('Detection', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif is_video or is_webcam:
        cap = cv2.VideoCapture(0 if is_webcam else source)
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        out_path = 'output_combined.mp4'
        writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        print("Running… press Q to quit")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = process_frame(frame, models)
            writer.write(frame)
            cv2.imshow('Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f"✓ Saved: {out_path}")

def process_frame(img, models):
    """Run all loaded models on a single frame and draw results"""

    # 1. Pothole detection
    if 'pothole' in models:
        for r in models['pothole'].predict(img, conf=0.25, verbose=False):
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                draw_box(img, x1, y1, x2, y2, 'Pothole', RED, float(box.conf[0]))

    # 2. Traffic sign detection (includes dos d'âne sign if in dataset)
    if 'traffic' in models:
        for r in models['traffic'].predict(img, conf=0.25, verbose=False):
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls  = int(box.cls[0])
                name = models['traffic'].names[cls]
                conf = float(box.conf[0])
                # Highlight speed bump signs differently
                color = YELLOW if is_speed_bump_sign(name) else GREEN
                label = f"Sign: {name}" if not is_speed_bump_sign(name) else f"⚠ Dos d'âne sign: {name}"
                draw_box(img, x1, y1, x2, y2, label, color, conf)

    # 3. Physical speed bump (dos d'âne) visual detection
    if 'speed_bump' in models:
        for r in models['speed_bump'].predict(img, conf=0.25, verbose=False):
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                draw_box(img, x1, y1, x2, y2, "Dos d'âne", ORANGE, float(box.conf[0]))

    # Legend
    legend = [
        ("Pothole",          RED),
        ("Traffic Sign",     GREEN),
        ("Dos d'ane sign",   YELLOW),
        ("Dos d'ane (road)", ORANGE),
    ]
    for i, (label, color) in enumerate(legend):
        cv2.rectangle(img, (10, 10 + i*22), (20, 20 + i*22), color, -1)
        cv2.putText(img, label, (25, 20 + i*22), FONT, 0.5, color, 1)

    return img

if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    if source == '0':
        source = 0
    run_detection(source)
