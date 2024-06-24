from ultralytics import YOLO
model = YOLO("../weights/fall_yolov8m_20240624.engine")

if __name__ == "__main__":
    input_video = "C:/Users/wmingdru/Desktop/workspace/data/shuaidao/test_video_20240624/merged/merged2.mp4"
    results = model(input_video, save=True, stream=True, classes=[0,1,2,3,4])
    for result in results:
        pass