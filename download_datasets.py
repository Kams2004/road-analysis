"""
HOW TO GET THE DOWNLOAD LINK (no account needed):
1. Go to https://universe.roboflow.com/traffic-sign-detection-vngbh/traffic-sign-detection-jgti0/dataset/1
2. Click  "Download Dataset"
3. Select format: YOLOv8
4. Click "Continue" → it shows a download link or a code snippet
5. Copy the URL that starts with https://universe.roboflow.com/ds/...
6. Paste it below as TRAFFIC_URL
"""

import os, zipfile, urllib.request, shutil

# ── Paste your URLs here ──────────────────────────────────────────────────────
TRAFFIC_URL   = "PASTE_TRAFFIC_SIGN_URL_HERE"
SPEEDBUMP_URL = "PASTE_SPEED_BUMP_URL_HERE"
# ─────────────────────────────────────────────────────────────────────────────

def download_and_extract(url, dest_folder, zip_name):
    if url.startswith("PASTE"):
        print(f"⚠  Please paste the download URL for {dest_folder} in this script first.")
        return False

    print(f"Downloading {dest_folder}...")
    urllib.request.urlretrieve(url, zip_name)

    print("Extracting...")
    os.makedirs(dest_folder, exist_ok=True)
    with zipfile.ZipFile(zip_name, 'r') as z:
        z.extractall(dest_folder)
    os.remove(zip_name)
    print(f"✓ {dest_folder} ready")
    return True

def fix_speed_bump_structure():
    """Move speed bump data into expected speed_bump/ folder"""
    raw = "speed_bump_raw"
    for split in ['train', 'valid', 'test']:
        for sub in ['images', 'labels']:
            src = f"{raw}/{split}/{sub}"
            dst = f"speed_bump/{split}/{sub}"
            os.makedirs(dst, exist_ok=True)
            if os.path.exists(src):
                for f in os.listdir(src):
                    shutil.copy(f"{src}/{f}", f"{dst}/{f}")
    print("✓ Speed bump data moved to speed_bump/")

if __name__ == "__main__":
    download_and_extract(TRAFFIC_URL,   "traffic_signs",  "traffic_signs.zip")
    ok = download_and_extract(SPEEDBUMP_URL, "speed_bump_raw", "speed_bump.zip")
    if ok:
        fix_speed_bump_structure()
