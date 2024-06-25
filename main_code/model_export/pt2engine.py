from ultralytics import YOLO

# Load a model
model = YOLO("../weights/climbing_80epoch_yolov8m20240625.pt")#load an official model
# Export the model
model.export(format="engine")