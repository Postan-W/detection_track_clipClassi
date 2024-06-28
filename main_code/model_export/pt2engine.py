from ultralytics import YOLO

# Load a model
model = YOLO("../weights/yolov8l_climb_6classes_0628.pt")#load an official model
# Export the model
model.export(format="engine")