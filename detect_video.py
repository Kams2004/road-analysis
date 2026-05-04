from ultralytics import YOLO
import cv2
import argparse
import numpy as np

# Minimum box area as % of frame area — filters tiny false positives
MIN_BOX_AREA_RATIO = 0.002   # 0.2% of frame
# Maximum box area — filters huge detections (houses, walls)
MAX_BOX_AREA_RATIO = 0.25    # 25% of frame
# Max aspect ratio (width/height or height/width) — potholes are roughly square
MAX_ASPECT_RATIO = 4.0
# Only detect in the ROAD ZONE: bottom X% of the frame
ROAD_ZONE_TOP = 0.40         # ignore top 40% of frame (sky, buildings, signs)

def is_valid_pothole(x1, y1, x2, y2, frame_w, frame_h, conf):
    """Filter out false positives using geometric and positional rules"""

    # 1. Must be in the road zone (lower part of frame)
    box_center_y = (y1 + y2) / 2
    if box_center_y < frame_h * ROAD_ZONE_TOP:
        return False, "above road zone"

    # 2. Box size filter
    box_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_w * frame_h
    area_ratio = box_area / frame_area

    if area_ratio < MIN_BOX_AREA_RATIO:
        return False, "too small"
    if area_ratio > MAX_BOX_AREA_RATIO:
        return False, "too large (likely building/wall)"

    # 3. Aspect ratio filter — potholes are not very elongated
    w = x2 - x1
    h = y2 - y1
    aspect = max(w, h) / (min(w, h) + 1e-6)
    if aspect > MAX_ASPECT_RATIO:
        return False, "wrong shape (too elongated)"

    return True, "ok"

def draw_road_zone(frame, frame_h):
    """Draw a semi-transparent overlay showing the active detection zone"""
    overlay = frame.copy()
    top_y = int(frame_h * ROAD_ZONE_TOP)
    cv2.rectangle(overlay, (0, top_y), (frame.shape[1], frame_h), (0, 255, 0), -1)
    cv2.addWeighted(overlay, 0.05, frame, 0.95, 0, frame)
    cv2.line(frame, (0, top_y), (frame.shape[1], top_y), (0, 255, 0), 2)
    cv2.putText(frame, "Road Detection Zone", (10, top_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

def detect_potholes_video(model_path, source, conf_threshold=0.45, save_output=True, show_zone=True):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open video source {source}")
        return

    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height} @ {fps}fps")
    print(f"Road zone: bottom {int((1 - ROAD_ZONE_TOP) * 100)}% of frame")
    print(f"Confidence threshold: {conf_threshold}")

    out = None
    if save_output:
        out = cv2.VideoWriter("output_detection.mp4",
                              cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_count = 0
    total_valid = 0
    total_filtered = 0

    print("Processing... Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame, conf=conf_threshold, verbose=False)

        annotated = frame.copy()
        if show_zone:
            annotated = draw_road_zone(annotated, height)

        valid_count = 0
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0].item()

                valid, reason = is_valid_pothole(x1, y1, x2, y2, width, height, conf)

                if valid:
                    valid_count += 1
                    total_valid += 1
                    # Draw valid detection in RED
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(annotated, f"Pothole {conf:.0%}",
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    total_filtered += 1
                    # Draw filtered detection in GREY (optional, for debugging)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    cv2.putText(annotated, f"Ignored:{reason}",
                                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

        # Info overlay
        cv2.putText(annotated, f"Frame:{frame_count}  Potholes:{valid_count}  Filtered:{total_filtered}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Pothole Detection", annotated)
        if out:
            out.write(annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
        print("\nOutput saved to: output_detection.mp4")
    cv2.destroyAllWindows()

    print(f"\n=== Detection Summary ===")
    print(f"Frames processed : {frame_count}")
    print(f"Valid potholes   : {total_valid}")
    print(f"Filtered out     : {total_filtered}  (houses, gutters, sky, etc.)")

def main():
    parser = argparse.ArgumentParser(description='Pothole Detection - Road Zone Only')
    parser.add_argument('--model',    type=str,   required=True)
    parser.add_argument('--source',   type=str,   required=True)
    parser.add_argument('--conf',     type=float, default=0.45,
                        help='Confidence threshold (default: 0.45)')
    parser.add_argument('--road-zone', type=float, default=0.40,
                        help='Top boundary of road zone as fraction of frame height (default: 0.40)')
    parser.add_argument('--no-save',  action='store_true')
    parser.add_argument('--no-zone',  action='store_true',
                        help='Hide the road zone overlay')
    args = parser.parse_args()

    global ROAD_ZONE_TOP
    ROAD_ZONE_TOP = args.road_zone

    source = int(args.source) if args.source.isdigit() else args.source
    detect_potholes_video(args.model, source, args.conf,
                          save_output=not args.no_save,
                          show_zone=not args.no_zone)

if __name__ == "__main__":
    main()
