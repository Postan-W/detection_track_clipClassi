

def get_model(path):
    from ultralytics import YOLO
    model = YOLO(path)
    print("模型检测的目标是:{}".format(model.names))
    return model

model_path = "./weights/yolov8m20240606.pt"
input = "../videos/output/merged_video.mp4"

def get_detected_output(model,input,classes,conf=0.65,imgsz=640,iou=0.7,device="cuda"):
    """
    :param model:
    :param input:
    :param classes: 要检测的对象的类别的id
    :return:
    """
    results = model(input,save=True,stream=True,classes=classes,conf=conf,iou=iou,imgsz=imgsz,device=device)#return a generator
    for result in results:#仅仅是为了让迭代器返回值，所以下面不写任何逻辑
        pass

get_detected_output(model=get_model(model_path),input=input,classes=[1])
