from ultralytics import YOLO

# Load a model
model = YOLO("../weights/yolov8x-pose.pt")#load an official model
# Export the model
model.export(format="engine")