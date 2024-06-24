from ultralytics import YOLO

# Load a model
model = YOLO("../weights/fall_yolov8m_20240624.pt")#load an official model
# Export the model
model.export(format="engine")