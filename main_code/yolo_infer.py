from ultralytics import YOLO

model = YOLO("./weights/yolov8m20240606.pt")
print("模型检测的目标是:{}".format(model.names))

input = "../videos/output/merged_video.mp4"

def get_detected_output(model,input):
    results = model(input,save=True,stream=True)#return a generator
    for result in results:#仅仅是为了让迭代器返回值，所以下面不写任何逻辑
        pass

get_detected_output(model=model,input=input)