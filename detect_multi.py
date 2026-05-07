from ultralytics import YOLO
import cv2
import argparse

# Color per class (BGR)
CLASS_COLORS = {
    'pothole':      (0,   0,   255),   # Red
    'speedlimit':   (0,   165, 255),   # Orange
    'crosswalk':    (0,   255,  0),    # Green
    'trafficlight': (255,  0,  255),   # Magenta
    'stop':         (0,   0,   180),   # Dark red
    'speed_bump':   (255, 255,  0),    # Cyan
}

ROAD_CLASSES  = {'pothole', 'speed_bump'}
ROAD_ZONE_TOP = 0.40


def is_valid(label, x1, y1, x2, y2, fw, fh):
    if label in ROAD_CLASSES:
        if (y1 + y2) / 2 < fh * ROAD_ZONE_TOP:
            return False
        ratio = ((x2 - x1) * (y2 - y1)) / (fw * fh)
        if ratio < 0.002 or ratio > 0.25:
            return False
    return True


def load_models(pothole_pt, signs_pt, speedbump_pt):
    models = []
    if pothole_pt:
        models.append(YOLO(pothole_pt))
        print(f"✅ Loaded pothole model: {pothole_pt}")
    if signs_pt:
        models.append(YOLO(signs_pt))
        print(f"✅ Loaded signs model:   {signs_pt}")
    if speedbump_pt:
        models.append(YOLO(speedbump_pt))
        print(f"✅ Loaded speedbump model: {speedbump_pt}")
    return models


def detect_video(models, source, conf=0.45, save_output=True):
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
        out = cv2.VideoWriter("output_detection.mp4",
                              cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_count = 0
    print("Processing... Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        annotated = frame.copy()
        counts = {}

        for model in models:
            results = model(frame, conf=conf, verbose=False)
            names   = model.names
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    label  = names[int(box.cls[0].item())]
                    conf_v = box.conf[0].item()

                    if not is_valid(label, x1, y1, x2, y2, width, height):
                        continue

                    color = CLASS_COLORS.get(label, (255, 255, 255))
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, f"{label} {conf_v:.0%}",
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    counts[label] = counts.get(label, 0) + 1

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
        print("Output saved to: output_detection.mp4")
    cv2.destroyAllWindows()
    print(f"Frames processed: {frame_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',    type=str, required=True)
    parser.add_argument('--pothole',   type=str, default=None, help='path to pothole model .pt')
    parser.add_argument('--signs',     type=str, default=None, help='path to signs model .pt')
    parser.add_argument('--speedbump', type=str, default=None, help='path to speedbump model .pt')
    parser.add_argument('--conf',      type=float, default=0.45)
    parser.add_argument('--no-save',   action='store_true')
    args = parser.parse_args()

    models = load_models(args.pothole, args.signs, args.speedbump)
    if not models:
        print("Error: provide at least one model (--pothole, --signs, --speedbump)")
        return

    source = int(args.source) if args.source.isdigit() else args.source
    detect_video(models, source, args.conf, save_output=not args.no_save)


if __name__ == "__main__":
    main()
