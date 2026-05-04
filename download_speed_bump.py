import os, shutil, subprocess

print("Downloading speed bump dataset...")
subprocess.run(
    'curl -L "https://app.roboflow.com/ds/ZgCutq1D4p?key=sHTH8mlqAG" > roboflow.zip && unzip -o roboflow.zip -d speed_bump_raw && rm roboflow.zip',
    shell=True, check=True
)

for split in ['train', 'valid', 'test']:
    for sub in ['images', 'labels']:
        src = f"speed_bump_raw/{split}/{sub}"
        dst = f"speed_bump/{split}/{sub}"
        os.makedirs(dst, exist_ok=True)
        if os.path.exists(src):
            for f in os.listdir(src):
                shutil.copy(f"{src}/{f}", f"{dst}/{f}")

shutil.rmtree("speed_bump_raw", ignore_errors=True)

total = sum(
    len(os.listdir(f"speed_bump/{s}/images"))
    for s in ['train', 'valid', 'test']
    if os.path.exists(f"speed_bump/{s}/images")
)
print(f"✓ {total} images ready in speed_bump/")
print("  Now run: python train_speed_bump.py")
