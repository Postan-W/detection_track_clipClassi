from ultralytics import YOLO

# Load a model
model = YOLO("../weights/fall_2024712_last.pt")#load an official model
# Export the model
model.export(format="engine")