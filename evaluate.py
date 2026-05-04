from ultralytics import YOLO
import argparse

def evaluate_model(model_path, data_yaml='data.yaml'):
    """Evaluate trained model on test dataset"""
    
    model = YOLO(model_path)
    
    print("Evaluating model on test dataset...")
    
    # Run validation
    metrics = model.val(
        data=data_yaml,
        split='test',
        imgsz=640,
        batch=16,
        conf=0.001,
        iou=0.6,
        plots=True,
        save_json=True,
        verbose=True
    )
    
    print("\n=== Evaluation Results ===")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Pothole Detection Model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model weights')
    
    args = parser.parse_args()
    evaluate_model(args.model)
