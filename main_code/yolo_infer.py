import numpy as np


def get_model(path):
    from ultralytics import YOLO
    model = YOLO(path)
    print("模型检测的目标是:{}".format(model.names))
    return model

model_path = "./weights/yolov8m.engine"
input = "../videos/output/merged_video.mp4"
model = get_model(model_path)
img = "../temp_images/img.png"
result = model.track(persist=True,source=img,tracker="./track_config/botsort.yaml",classes=[0],conf=0.3,iou=0.7)[0]
boxes = result.boxes.data.numpy()
print(boxes)
print(boxes.tolist())
print(np.array(boxes.tolist()))
def get_detected_output(model,input,classes,conf=0.65,imgsz=640,iou=0.7,device="cuda",save_frames=False):
    """
    :param model:
    :param input:
    :param classes: 要检测的对象的类别的id
    :return:
    """
    results = model(input,save=True,stream=True,classes=classes,conf=conf,iou=iou,imgsz=imgsz,save_frames=save_frames)#return a generator
    for result in results:#仅仅是为了让生成器返回值，所以下面不写任何逻辑
        pass

# get_detected_output(model=get_model(model_path),input=input,classes=[1],save_frames=True)
