from roboflow import Roboflow
import os
import shutil
import yaml

def download_traffic_signs():
    """Download traffic sign dataset from Roboflow"""
    print("Downloading traffic sign dataset...")
    rf = Roboflow(api_key="YOUR_API_KEY")  # Get from roboflow.com
    project = rf.workspace("traffic-sign-detection-vngbh").project("traffic-sign-detection-jgti0")
    dataset = project.version(1).download("yolov8")
    return dataset.location

def merge_datasets():
    """Merge pothole and traffic sign datasets"""
    print("\nMerging datasets...")
    
    traffic_base = "Traffic-Sign-Detection-1"
    
    for split in ['train', 'valid', 'test']:
        traffic_img = f"{traffic_base}/{split}/images"
        traffic_lbl = f"{traffic_base}/{split}/labels"
        
        if not os.path.exists(traffic_img):
            continue
            
        # Copy traffic sign images and labels
        for img in os.listdir(traffic_img):
            shutil.copy(f"{traffic_img}/{img}", f"{split}/images/{img}")
        
        for lbl in os.listdir(traffic_lbl):
            # Update class IDs in labels (traffic signs start from class 1)
            with open(f"{traffic_lbl}/{lbl}", 'r') as f:
                lines = f.readlines()
            
            with open(f"{split}/labels/{lbl}", 'w') as f:
                for line in lines:
                    parts = line.strip().split()
                    class_id = int(parts[0]) + 1  # Shift traffic sign classes
                    f.write(f"{class_id} {' '.join(parts[1:])}\n")
    
    print("Dataset merge complete!")

def create_combined_config():
    """Create combined data.yaml"""
    # Read traffic sign classes
    with open("Traffic-Sign-Detection-1/data.yaml", 'r') as f:
        traffic_data = yaml.safe_load(f)
    
    combined_config = {
        'train': '../train/images',
        'val': '../valid/images',
        'test': '../test/images',
        'nc': 1 + len(traffic_data['names']),
        'names': ['pothole'] + traffic_data['names']
    }
    
    with open('data_combined.yaml', 'w') as f:
        yaml.dump(combined_config, f, default_flow_style=False)
    
    print(f"\nCombined config created with {combined_config['nc']} classes:")
    print(combined_config['names'])

if __name__ == "__main__":
    # Step 1: Download traffic signs
    download_traffic_signs()
    
    # Step 2: Merge datasets
    merge_datasets()
    
    # Step 3: Create combined config
    create_combined_config()
    
    print("\n✓ Setup complete! Use 'data_combined.yaml' for training.")
