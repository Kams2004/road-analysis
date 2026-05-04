"""
Convert Pascal VOC XML annotations to YOLO format
and split into train/val/test sets
"""
import os
import xml.etree.ElementTree as ET
import shutil
import random
from pathlib import Path

DATASET_PATH = "/home/kamsu-perold/.cache/kagglehub/datasets/andrewmvd/road-sign-detection/versions/1"
OUTPUT_PATH = "/home/kamsu-perold/pothole-detection/traffic_signs"

CLASSES = ['speedlimit', 'crosswalk', 'trafficlight', 'stop']

def convert_box(size, box):
    """Convert VOC bbox to YOLO format (normalized xywh)"""
    dw, dh = 1.0 / size[0], 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh

def convert_annotation(xml_path, label_path):
    """Convert single XML file to YOLO label file"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    lines = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in CLASSES:
            continue
        cls_id = CLASSES.index(cls)
        bb = obj.find('bndbox')
        box = (float(bb.find('xmin').text), float(bb.find('xmax').text),
               float(bb.find('ymin').text), float(bb.find('ymax').text))
        lines.append(f"{cls_id} " + " ".join(f"{v:.6f}" for v in convert_box((w, h), box)))

    if lines:
        with open(label_path, 'w') as f:
            f.write("\n".join(lines))
        return True
    return False

def prepare_dataset():
    # Get all image files that have annotations
    images_dir = Path(DATASET_PATH) / "images"
    annotations_dir = Path(DATASET_PATH) / "annotations"

    all_files = [f.stem for f in annotations_dir.glob("*.xml")]
    random.seed(42)
    random.shuffle(all_files)

    # Split: 80% train, 10% val, 10% test
    n = len(all_files)
    train = all_files[:int(n * 0.8)]
    val   = all_files[int(n * 0.8):int(n * 0.9)]
    test  = all_files[int(n * 0.9):]

    print(f"Total: {n} | Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    for split, files in [("train", train), ("valid", val), ("test", test)]:
        img_out = Path(OUTPUT_PATH) / split / "images"
        lbl_out = Path(OUTPUT_PATH) / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for stem in files:
            img_src = images_dir / f"{stem}.png"
            xml_src = annotations_dir / f"{stem}.xml"

            if not img_src.exists() or not xml_src.exists():
                continue

            shutil.copy(img_src, img_out / f"{stem}.png")
            convert_annotation(xml_src, lbl_out / f"{stem}.txt")

    # Write data.yaml
    yaml_content = f"""train: {OUTPUT_PATH}/train/images
val: {OUTPUT_PATH}/valid/images
test: {OUTPUT_PATH}/test/images

nc: {len(CLASSES)}
names: {CLASSES}
"""
    with open(f"{OUTPUT_PATH}/data.yaml", 'w') as f:
        f.write(yaml_content)

    print(f"Dataset prepared at: {OUTPUT_PATH}")
    print(f"data.yaml written at: {OUTPUT_PATH}/data.yaml")

if __name__ == "__main__":
    prepare_dataset()
