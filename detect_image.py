from ultralytics import YOLO
import cv2
import argparse
from pathlib import Path

def detect_potholes_image(model_path, image_path, conf_threshold=0.25, save_path=None):
    """
    Detect potholes in a single image
    
    Args:
        model_path: Path to trained model weights
        image_path: Path to input image
        conf_threshold: Confidence threshold for detections
        save_path: Path to save output image (optional)
    """
    
    # Load model
    model = YOLO(model_path)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image from {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Run detection
    results = model(image, conf=conf_threshold, verbose=False)
    
    # Process results
    result = results[0]
    boxes = result.boxes
    
    # Print detection results
    print(f"\n{'='*50}")
    print(f"DETECTION RESULTS")
    print(f"{'='*50}")
    print(f"Total potholes detected: {len(boxes)}")
    
    if len(boxes) > 0:
        print(f"\nDetailed detections:")
        for i, box in enumerate(boxes):
            conf = box.conf[0].item()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width = x2 - x1
            height = y2 - y1
            print(f"  Pothole {i+1}:")
            print(f"    - Confidence: {conf:.2%}")
            print(f"    - Location: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})")
            print(f"    - Size: {int(width)}x{int(height)} pixels")
    else:
        print("No potholes detected in the image.")
    
    # Create annotated image
    annotated_image = result.plot()
    
    # Determine save path
    if save_path is None:
        input_path = Path(image_path)
        save_path = f"detected_{input_path.name}"
    
    # Save annotated image
    cv2.imwrite(save_path, annotated_image)
    print(f"\n{'='*50}")
    print(f"Annotated image saved to: {save_path}")
    print(f"{'='*50}\n")
    
    return len(boxes), save_path

def main():
    parser = argparse.ArgumentParser(description='Pothole Detection on Single Image')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model weights (best.pt)')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output image')
    
    args = parser.parse_args()
    
    detect_potholes_image(
        model_path=args.model,
        image_path=args.image,
        conf_threshold=args.conf,
        save_path=args.output
    )

if __name__ == "__main__":
    main()
