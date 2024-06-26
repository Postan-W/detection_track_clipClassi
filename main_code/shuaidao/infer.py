from ultralytics import YOLO
model = YOLO("../weights/fall_yolov8l_20240626.engine")
print("类别有:{}".format(model.names))

if __name__ == "__main__":
    input_video = "../../videos/output/shuaidao_merged.mp4"
    results = model(input_video, save=True, stream=True, classes=[3],conf=0.75)
    for result in results:
        pass