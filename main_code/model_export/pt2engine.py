from ultralytics import YOLO

# Load a model
model = YOLO("../weights/yolov10l_climb_fall2.pt")
# Export the model
model.export(format="engine",device=[0])