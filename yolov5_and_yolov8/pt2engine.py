from ultralytics import YOLO

# Load a model
model = YOLO("yolov5s.pt")
# Export the model
model.export(format="engine",device=[0])