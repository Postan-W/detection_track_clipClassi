import torch
from ultralytics import YOLO
import numpy as np
from utils.tools import plot_one_box,letterbox,non_max_suppression,scale_coords,in_poly_area,draw_poly_area
import random
import cv2
class YOLOv5Detector:
    def __init__(self,model="yolov5s.pt",device=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model, map_location=self.device)['model'].float()
        self.model.to(self.device).eval()
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            self.model.half()  # to FP16

        self.confthre = 0.6
        self.nmsthre = 0.45
        self.img_size = 640

        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        self.label = "person"  # 只识别人

    def inference(self, image,area):
        img = letterbox(image,new_shape=self.img_size)[0]#接收的是单张图片
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR到RGB;hwc到chw
        img = np.ascontiguousarray(img)#转为元素内存连续数组，提高数据处理速度
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()#uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = img.unsqueeze(0)
        pred = self.model(img, augment=False)[0]
        pred = non_max_suppression(pred,self.confthre,self.nmsthre,classes=[self.names.index(self.label)],agnostic=False)[0]#NMS筛选的结果是(1,n,6)，并且按conf从大到小排列

        pred[:,:4] = scale_coords(img.shape[2:],pred[:,:4],image.shape)#将bbox缩放回原图
        draw_poly_area(image,area)
        for *xyxy, conf, cls in pred:
            if in_poly_area(xyxy,area):
                plot_one_box(xyxy, image, color=(0, 255, 0))
        return pred

class YOLOv8Detector:
    def __init__(self, model="./yolov8s.pt",device=None):
        self.device = torch.device(device)
        self.model = YOLO(model)


    def inference(self, image, area):
        result  = self.model(source=image,classes=[0],save=False,conf=0.3,iou=0.7,verbose=False,device=self.device)[0]
        boxes = result.boxes.data.cpu().numpy().tolist()
        draw_poly_area(image,area)
        for box in boxes:
            if in_poly_area(box, area):
                plot_one_box(box, image, color=(0, 255, 0))


if __name__ == "__main__":
    area_point_list = [[[315, 34], [619, 49], [585, 585], [335, 578]]]
    # yolov5 = YOLOv5Detector("./yolov5s.pt")
    yolov8 = YOLOv8Detector()
    image = cv2.imread("./p1.jpeg")
    # yolov5.inference(image,area_point_list)
    yolov8.inference(image, area_point_list)
    cv2.imshow("ds", image)
    cv2.waitKey(0)
    while True:
        pass