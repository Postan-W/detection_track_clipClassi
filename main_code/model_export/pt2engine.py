from ultralytics import YOLO

# Load a model
model = YOLO("../weights/climb_20240704.pt")#load an official model
# Export the model
model.export(format="engine")