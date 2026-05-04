from roboflow import Roboflow
import os, shutil

rf = Roboflow(api_key="KlGixvbuJhl93JIvXhH4")

# Speed bump warning sign dataset (panneau dos d'âne)
project = rf.workspace("roboflow-universe-projects").project("speed-bump-sign-detection")
version = project.version(1)
dataset = version.download("yolov8", location="./speed_bump_sign_raw")

print("✓ Speed bump sign dataset downloaded to ./speed_bump_sign_raw")
print("  Now run: python merge_speed_bump_sign.py")
