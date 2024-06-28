from ultralytics import YOLO
# Load a model
model = YOLO("../weights/yolov8m-pose.pt")  # load an official model
# model = YOLO("path/to/yolov8l_20240617.pt")  # load a custom model
print(model.names)#人体姿势估计，检测的对象只有人
source_path = "../clip_images/pj1_120.jpg"
# Predict with the model
results = model(source_path,save=True)  # predict on an image
result = results[0]
print(result.boxes.data.cpu().numpy())
print(result.keypoints.xy.cpu().numpy())
