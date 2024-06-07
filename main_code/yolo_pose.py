from ultralytics import YOLO

# Load a model
model = YOLO("weights/yolov8n-pose.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg",save=True)  # predict on an image