from ultralytics import YOLO

# Load a model
model = YOLO("./weights/yolov8m.pt")  # pretrained YOLOv8n model.type:nsmlx
print(model.names)

result = model.track(source="./bus.jpg",save=True)[0]
print(result.boxes.data.tolist())
print(result.boxes.id)