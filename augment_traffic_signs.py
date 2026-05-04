"""
Augment the existing traffic signs dataset to increase variety
and improve detection of signs in different conditions
"""
import cv2
import numpy as np
import os
import shutil
from pathlib import Path

SRC = Path("/home/kamsu-perold/pothole-detection/traffic_signs/train")
DST = Path("/home/kamsu-perold/pothole-detection/traffic_signs_augmented/train")

def augment_image(img):
    """Return list of augmented versions of the image"""
    augmented = []

    # Brightness variations
    for factor in [0.5, 0.7, 1.3, 1.6]:
        bright = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        augmented.append(bright)

    # Blur (simulates motion/distance)
    augmented.append(cv2.GaussianBlur(img, (3, 3), 0))

    # Noise
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    augmented.append(noisy)

    # Horizontal flip (with label adjustment handled separately)
    augmented.append(cv2.flip(img, 1))

    return augmented

def flip_label(label_line):
    """Flip bounding box horizontally: x_center = 1 - x_center"""
    parts = label_line.strip().split()
    parts[1] = str(round(1.0 - float(parts[1]), 6))
    return " ".join(parts)

def augment_dataset():
    # Copy original first
    if DST.parent.exists():
        shutil.rmtree(DST.parent)
    shutil.copytree(SRC.parent, DST.parent)

    img_dir = DST / "images"
    lbl_dir = DST / "labels"

    images = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
    print(f"Augmenting {len(images)} images...")

    for img_path in images:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        with open(lbl_path) as f:
            labels = f.readlines()

        augmented_imgs = augment_image(img)
        suffixes = ['_b05', '_b07', '_b13', '_b16', '_blur', '_noise', '_flip']

        for suffix, aug_img in zip(suffixes, augmented_imgs):
            out_img = img_dir / f"{img_path.stem}{suffix}{img_path.suffix}"
            out_lbl = lbl_dir / f"{img_path.stem}{suffix}.txt"

            cv2.imwrite(str(out_img), aug_img)

            # Flip labels for horizontal flip
            if suffix == '_flip':
                flipped = [flip_label(l) for l in labels]
                with open(out_lbl, 'w') as f:
                    f.writelines(flipped)
            else:
                with open(out_lbl, 'w') as f:
                    f.writelines(labels)

    total = len(list(img_dir.glob("*.*")))
    print(f"Augmented dataset: {total} total training images")

    # Write new data.yaml
    yaml = f"""train: {DST.parent}/train/images
val: /home/kamsu-perold/pothole-detection/traffic_signs/valid/images
test: /home/kamsu-perold/pothole-detection/traffic_signs/test/images

nc: 4
names: ['speedlimit', 'crosswalk', 'trafficlight', 'stop']
"""
    with open(DST.parent / "data.yaml", 'w') as f:
        f.write(yaml)
    print(f"data.yaml written to: {DST.parent}/data.yaml")

if __name__ == "__main__":
    augment_dataset()
