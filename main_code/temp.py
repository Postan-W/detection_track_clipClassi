from ultralytics import YOLO

model = YOLO("./weights/yolov8m20240606.pt")

results = model(source="./videos/positive1.mp4",stream=True,classes=[1])