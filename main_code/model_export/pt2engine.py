from ultralytics import YOLO

# Load a model
model = YOLO("../weights/fall_2024712_last.pt")
# Export the model
model.export(format="engine",device=[0])