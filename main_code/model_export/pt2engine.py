from ultralytics import YOLO

# Load a model
model = YOLO("../weights/climb_yolov8l_80epoch_batch64_old_data_20240625.pt")#load an official model
# Export the model
model.export(format="engine")