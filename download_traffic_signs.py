from roboflow import Roboflow

rf = Roboflow(api_key="KlGixvbuJhl93JIvXhH4")
project = rf.workspace("traffic-sign-detection-vngbh").project("traffic-sign-detection-jgti0")
version = project.version(1)
dataset = version.download("yolov8", location="./traffic_signs")

print("✓ Traffic sign dataset downloaded to ./traffic_signs")
