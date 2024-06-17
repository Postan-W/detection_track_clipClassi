from ultralytics import YOLO

# Load a model
model = YOLO("./weights/yolov8m.pt")  # pretrained YOLOv8n model.type:nsmlx
print(model.names)

result = model.track(source="./bus.jpg",save=True,classes=[0])[0]
print([i[-3] for i in result.boxes.data.tolist()])
print(result.boxes.id)