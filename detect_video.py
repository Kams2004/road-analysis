from ultralytics import YOLO
import cv2
import argparse

# Color per class (BGR)
CLASS_COLORS = {
    'pothole':      (0,   0,   255),  # Red
    'speedlimit':   (255, 165,  0),   # Orange
    'crosswalk':    (0,   255,  0),   # Green
    'trafficlight': (255, 0,   255),  # Magenta
    'stop':         (0,   0,   180),  # Dark red
    'speed_bump':   (255, 255,  0),   # Cyan
}

# Road zone — only apply geometric filter to pothole & speed_bump
ROAD_CLASSES   = {'pothole', 'speed_bump'}
ROAD_ZONE_TOP  = 0.40   # ignore top 40% for road classes


def is_valid_detection(label, x1, y1, x2, y2, frame_w, frame_h):
    if label in ROAD_CLASSES:
        center_y = (y1 + y2) / 2
        if center_y < frame_h * ROAD_ZONE_TOP:
            return False
        area_ratio = ((x2 - x1) * (y2 - y1)) / (frame_w * frame_h)
        if area_ratio < 0.002 or area_ratio > 0.25:
            return False
    return True


def detect_video(model_path, source, conf=0.45, save_output=True):
    model  = YOLO(model_path)
    names  = model.names  # {0: 'pothole', 1: 'speedlimit', ...}

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: cannot open {source}")
        return

    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height} @ {fps}fps")

    out = None
    if save_output:
        out = cv2.VideoWriter("output_combined.mp4",
                              cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_count = 0
    print("Processing... Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame, conf=conf, verbose=False)
        annotated = frame.copy()

        counts = {}
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id  = int(box.cls[0].item())
                conf_v  = box.conf[0].item()
                label   = names[cls_id]

                if not is_valid_detection(label, x1, y1, x2, y2, width, height):
                    continue

                color = CLASS_COLORS.get(label, (255, 255, 255))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{label} {conf_v:.0%}",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                counts[label] = counts.get(label, 0) + 1

        # HUD
        summary = "  ".join(f"{k}:{v}" for k, v in counts.items()) or "no detections"
        cv2.putText(annotated, f"Frame {frame_count} | {summary}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow("Road Analysis", annotated)
        if out:
            out.write(annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
        print("Output saved to: output_combined.mp4")
    cv2.destroyAllWindows()
    print(f"Frames processed: {frame_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   type=str, required=True)
    parser.add_argument('--source',  type=str, required=True)
    parser.add_argument('--conf',    type=float, default=0.45)
    parser.add_argument('--no-save', action='store_true')
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    detect_video(args.model, source, args.conf, save_output=not args.no_save)


if __name__ == "__main__":
    main()
