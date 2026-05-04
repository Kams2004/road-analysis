from ultralytics import YOLO
import cv2
import argparse
import torch
import torchvision.ops as ops
from pathlib import Path
from datetime import datetime

SIGN_COLOR_DEFAULT = (255, 255, 255)
SIGN_COLORS = {
    'traffic_light': (0, 255, 0),
    'stop':          (0, 0, 255),
    'give_way':      (0, 165, 255),
    'speed_hump':    (255, 0, 255),
}
POTHOLE_COLOR    = (0, 0, 255)
SPEED_BUMP_COLOR = (255, 0, 255)   # Magenta

ROAD_ZONE_TOP    = 0.50
MIN_BOX_AREA     = 0.002
MAX_BOX_AREA     = 0.10
MAX_ASPECT_RATIO = 3.0

def is_valid_pothole(x1, y1, x2, y2, fw, fh):
    if (y1 + y2) / 2 < fh * ROAD_ZONE_TOP:
        return False
    area_ratio = ((x2 - x1) * (y2 - y1)) / (fw * fh)
    if not (MIN_BOX_AREA < area_ratio < MAX_BOX_AREA):
        return False
    aspect = max(x2-x1, y2-y1) / (min(x2-x1, y2-y1) + 1e-6)
    return aspect <= MAX_ASPECT_RATIO

def agnostic_nms(boxes_result, iou_thresh=0.4):
    """NMS within same-type groups: suppress duplicate speed limits, and duplicate other signs separately."""
    boxes = boxes_result[0].boxes
    if len(boxes) == 0:
        return boxes_result
    names = boxes_result[0].names

    speed_limit_idx = [i for i, b in enumerate(boxes) if names[int(b.cls[0].item())].startswith('speed_limit')]
    other_idx       = [i for i, b in enumerate(boxes) if not names[int(b.cls[0].item())].startswith('speed_limit')]

    keep = []
    for group in [speed_limit_idx, other_idx]:
        if not group:
            continue
        idx = torch.tensor(group)
        g_boxes = boxes[idx]
        kept = ops.nms(g_boxes.xyxy, g_boxes.conf, iou_thresh)
        keep.extend(idx[kept].tolist())

    keep = sorted(keep)
    boxes_result[0].boxes = boxes[torch.tensor(keep)]
    return boxes_result

def save_crops(frame, detections, label, crops_dir, source_id):
    if not detections:
        return
    out_dir = Path(crops_dir) / label
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (x1, y1, x2, y2, conf) in enumerate(detections):
        cv2.imwrite(str(out_dir / f"{source_id}_{label}_{i+1}_{conf:.0f}.jpg"), frame)

def _save_all_crops(frame, annotated, pothole_results, traffic_results, speed_bump_results, crops_dir, source_id):
    fh, fw = frame.shape[:2]
    pothole_boxes = [
        (max(0,x1), max(0,y1), min(fw,x2), min(fh,y2), box.conf[0].item()*100)
        for result in pothole_results for box in result.boxes
        for x1,y1,x2,y2 in [list(map(int, box.xyxy[0].tolist()))]
        if is_valid_pothole(x1, y1, x2, y2, fw, fh)
    ]
    save_crops(annotated, pothole_boxes, 'pothole', crops_dir, source_id)
    traffic_names = traffic_results[0].names
    sign_boxes = {}
    for result in traffic_results:
        for box in result.boxes:
            x1,y1,x2,y2 = list(map(int, box.xyxy[0].tolist()))
            cls_name = traffic_names[int(box.cls[0].item())]
            sign_boxes.setdefault(cls_name, []).append(
                (max(0,x1), max(0,y1), min(fw,x2), min(fh,y2), box.conf[0].item()*100))
    for cls_name, boxes in sign_boxes.items():
        save_crops(annotated, boxes, cls_name, crops_dir, source_id)
    if speed_bump_results:
        sb_boxes = [
            (max(0,x1), max(0,y1), min(fw,x2), min(fh,y2), box.conf[0].item()*100)
            for result in speed_bump_results for box in result.boxes
            for x1,y1,x2,y2 in [list(map(int, box.xyxy[0].tolist()))]
        ]
        save_crops(annotated, sb_boxes, 'speed_bump', crops_dir, source_id)

def load_models(pothole_path, traffic_path, speed_bump_path):
    print("Loading pothole model...")
    pothole_model = YOLO(pothole_path)
    print("Loading traffic sign model...")
    traffic_model = YOLO(traffic_path)
    speed_bump_model = None
    if speed_bump_path:
        print("Loading speed bump model...")
        speed_bump_model = YOLO(speed_bump_path)
    return pothole_model, traffic_model, speed_bump_model

def draw_detections(frame, pothole_results, traffic_results, speed_bump_results):
    annotated = frame.copy()
    fw, fh = frame.shape[1], frame.shape[0]

    # Potholes
    for result in pothole_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            if not is_valid_pothole(x1, y1, x2, y2, fw, fh):
                continue
            cv2.rectangle(annotated, (x1, y1), (x2, y2), POTHOLE_COLOR, 2)
            cv2.putText(annotated, f"Pothole {conf:.0%}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, POTHOLE_COLOR, 2)

    # Traffic signs
    traffic_names = traffic_results[0].names
    for result in traffic_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            cls_name = traffic_names[int(box.cls[0].item())]
            color = SIGN_COLORS.get(cls_name, SIGN_COLOR_DEFAULT)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name.replace('_', ' ').title()} {conf:.0%}"
            cv2.putText(annotated, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Speed bumps
    for result in (speed_bump_results or []):
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), SPEED_BUMP_COLOR, 2)
            cv2.putText(annotated, f"Speed Bump {conf:.0%}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, SPEED_BUMP_COLOR, 2)

    return annotated

def print_results(pothole_results, traffic_results, speed_bump_results, frame_id=None):
    prefix = f"Frame {frame_id} | " if frame_id is not None else ""
    traffic_names = traffic_results[0].names

    print(f"\n{prefix}{'='*50}")
    print(f"Potholes detected: {len(pothole_results[0].boxes)}")
    for i, box in enumerate(pothole_results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        print(f"  Pothole {i+1}: conf={box.conf[0]:.2%}  loc=({x1},{y1})-({x2},{y2})")

    print(f"Traffic signs detected: {len(traffic_results[0].boxes)}")
    for i, box in enumerate(traffic_results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_name = traffic_names[int(box.cls[0].item())]
        print(f"  Sign {i+1}: {cls_name.replace('_', ' ').title()}  conf={box.conf[0]:.2%}  loc=({x1},{y1})-({x2},{y2})")

    if speed_bump_results:
        print(f"Speed bumps detected: {len(speed_bump_results[0].boxes)}")
        for i, box in enumerate(speed_bump_results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            print(f"  Speed Bump {i+1}: conf={box.conf[0]:.2%}  loc=({x1},{y1})-({x2},{y2})")

def detect_image(pothole_model, traffic_model, speed_bump_model, image_path, conf, iou, crops_dir, output_path=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return

    pothole_results    = pothole_model(image, conf=conf, iou=iou, verbose=False)
    traffic_results    = traffic_model(image, conf=conf, iou=iou, verbose=False)
    traffic_results    = agnostic_nms(traffic_results, iou)
    speed_bump_results = speed_bump_model(image, conf=conf, iou=iou, verbose=False) if speed_bump_model else None

    print_results(pothole_results, traffic_results, speed_bump_results)
    annotated = draw_detections(image, pothole_results, traffic_results, speed_bump_results)
    _save_all_crops(frame=image, annotated=annotated, pothole_results=pothole_results,
                    traffic_results=traffic_results, speed_bump_results=speed_bump_results,
                    crops_dir=crops_dir, source_id=Path(image_path).stem)

    if output_path is None:
        output_path = f"detected_{Path(image_path).name}"

    cv2.imwrite(output_path, annotated)
    print(f"\nAnnotated image saved to: {output_path}")

def detect_video(pothole_model, traffic_model, speed_bump_model, source, conf, iou, crops_dir, save_output=True):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open source {source}")
        return

    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height} @ {fps}fps | Press 'q' to quit")

    out = None
    if save_output:
        out = cv2.VideoWriter("output_combined.mp4",
                              cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        pothole_results    = pothole_model(frame, conf=conf, iou=iou, verbose=False)
        traffic_results    = traffic_model(frame, conf=conf, iou=iou, verbose=False)
        traffic_results    = agnostic_nms(traffic_results, iou)
        speed_bump_results = speed_bump_model(frame, conf=conf, iou=iou, verbose=False) if speed_bump_model else None

        if frame_count % 30 == 0:
            print_results(pothole_results, traffic_results, speed_bump_results, frame_id=frame_count)
        annotated = draw_detections(frame, pothole_results, traffic_results, speed_bump_results)
        _save_all_crops(frame=frame, annotated=annotated, pothole_results=pothole_results,
                        traffic_results=traffic_results, speed_bump_results=speed_bump_results,
                        crops_dir=crops_dir, source_id=f"frame{frame_count:06d}")

        n_potholes   = len(pothole_results[0].boxes)
        n_signs      = len(traffic_results[0].boxes)
        n_speedbumps = len(speed_bump_results[0].boxes) if speed_bump_results else 0
        cv2.putText(annotated,
                    f"Frame:{frame_count}  Potholes:{n_potholes}  Signs:{n_signs}  SpeedBumps:{n_speedbumps}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Combined Detection", annotated)
        if out:
            out.write(annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
        print("\nOutput saved to: output_combined.mp4")
    cv2.destroyAllWindows()
    print(f"Total frames processed: {frame_count}")

def main():
    parser = argparse.ArgumentParser(description='Combined Pothole, Traffic Sign & Speed Bump Detection')
    parser.add_argument('--pothole-model',    type=str, required=True)
    parser.add_argument('--traffic-model',    type=str, required=True)
    parser.add_argument('--speed-bump-model', type=str, required=False, default=None)
    parser.add_argument('--source', type=str, required=True,
                        help='Image path, video path, or 0 for webcam')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou',  type=float, default=0.4)
    parser.add_argument('--crops-dir', type=str, default='detections_crops')
    parser.add_argument('--no-save', action='store_true')
    args = parser.parse_args()

    pothole_model, traffic_model, speed_bump_model = load_models(
        args.pothole_model, args.traffic_model, args.speed_bump_model
    )

    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    source = args.source
    is_image = Path(source).suffix.lower() in image_exts

    if is_image:
        detect_image(pothole_model, traffic_model, speed_bump_model, source, args.conf, args.iou, args.crops_dir,
                     output_path=None if not args.no_save else "/dev/null")
    else:
        src = int(source) if source.isdigit() else source
        detect_video(pothole_model, traffic_model, speed_bump_model, src, args.conf, args.iou, args.crops_dir,
                     save_output=not args.no_save)

if __name__ == "__main__":
    main()
