"""
Detect and classify road signs in an image or video.
Prints the sign name (e.g. 'speed_limit_80', 'give_way', 'pedestrian_crossing')
for every detection.
"""
import argparse, cv2
from ultralytics import YOLO

MODEL_PATH = "/home/kamsu-perold/pothole-detection/runs/traffic_signs/train_v1/weights/best.pt"
CONF       = 0.4


def detect(source: str, model_path: str = MODEL_PATH, conf: float = CONF, show: bool = True):
    model = YOLO(model_path)
    results = model.predict(source=source, conf=conf, save=True, stream=True)

    for r in results:
        frame = r.orig_img.copy()
        for box in r.boxes:
            cls_id    = int(box.cls[0])
            cls_name  = model.names[cls_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = f"{cls_name} {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            print(f"  Detected: {cls_name:30s}  conf={confidence:.2f}  box=[{x1},{y1},{x2},{y2}]")

        if show:
            cv2.imshow("Road Sign Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Image/video path or 0 for webcam")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--conf",  default=CONF, type=float)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    detect(args.source, args.model, args.conf, show=not args.no_show)
