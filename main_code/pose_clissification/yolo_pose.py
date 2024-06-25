from ultralytics import YOLO

# Load a model
model = YOLO("../weights/yolov8m-pose.pt")  # load an official model
# model = YOLO("path/to/yolov8l_20240617.pt")  # load a custom model

source_path = "C:/Users/wmingdru/Desktop/workspace/projects/detection_track_clipClassi/subway_images/train/images/allscenes4_150.jpg"
# Predict with the model
results = model(source_path,save=True,classes=[0])  # predict on an image
result = results[0]
print(len(result.boxes.data.cpu().numpy()))
print(len(result.keypoints.xy.cpu().numpy()))
