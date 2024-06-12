from ultralytics import YOLO
from get_input import VideoReader
from queue import Queue
from data_structure import Frame
from threading import Thread
from myutils.public_logger import logger

class ClimbingDetection:
    def __init__(self,input_queue:Queue,output_queue:Queue,yolo_model:str="./weights/yolov8m20240606.pt"):
        self.yolo_model = YOLO(yolo_model)
        self.thread = Thread(target=self.task)
        self.input_queue = input_queue
        self.output_queue = output_queue

    def task(self):
        while True:
            frame = self.input_queue.get()
            if not frame.stops:
                result = self.yolo_model(source=frame.data,classes=[1],conf=0.3,iou=0.7,stream=False,show_labels=False,show_conf=False,show_boxes=False,save=False,save_crop=False)[0]#因为只有一张图片

            else:
                self.output_queue.put(frame)

    def start(self):
        self.thread.start()

if __name__ == '__main__':
    input_queue = Queue(1000)
    output_queue = Queue(1000)
    video_path = "./videos/positive1.mp4"
    video_reader = VideoReader(video_path=video_path,image_queue=input_queue,timestep=1)
    video_reader.start()


