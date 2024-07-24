from ultralytics import YOLO

# Load a model
model = YOLO("../weights/yolov8m-pose.pt")
# Export the model
model.export(format="engine",device=[0])