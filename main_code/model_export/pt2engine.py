from ultralytics import YOLO

# Load a model
model = YOLO("../weights/cimb_fall_1280.pt")
# Export the model
model.export(format="engine",device=[0])