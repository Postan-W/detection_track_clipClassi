from ultralytics import YOLO
from get_input import VideoReader
from queue import Queue
from data_structure import Frame
from threading import Thread
from myutils.public_logger import logger
import cv2

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
                #注1：If object confidence score will be low, i.e lower than track_high_thresh, then there will be no tracks successfully returned and updated.
                #注2：Tracking configuration shares properties with Predict mode, such as conf, iou, and show. For further configurations, refer to the Predict model page.
                #注3：Ultralytics also allows you to use a modified tracker configuration file. To do this, simply make a copy of a tracker config file (for example, custom_tracker.yaml) from ultralytics/cfg/trackers and modify any configurations (except the tracker_type) as per your needs.
                result = self.yolo_model.track(source=frame.data,classes=[1],conf=0.3,iou=0.7,stream=False,show_labels=False,show_conf=False,show_boxes=False,save=False,save_crop=False)[0]#因为只有一张图片
                frame.boxes = result.boxes.data.tolist()#[[],[]..]] or []
                print(result.boxes.is_track)
                frame.data = result.plot()
                self.output_queue.put(frame)
            else:
                self.output_queue.put(frame)
                break

    def start(self):
        self.thread.start()

if __name__ == '__main__':
    input_queue = Queue(1000)
    output_queue = Queue(1000)
    video_path = "./videos/positive1.mp4"
    output_path = "./outputs/positive1_output.mp4"
    video_reader = VideoReader(video_path=video_path,image_queue=input_queue,timestep=1)
    video_reader.start()
    climbing_detection = ClimbingDetection(input_queue,output_queue)
    climbing_detection.start()
    first_image = output_queue.get().data
    height, width, _ = first_image.shape
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
    video.write(first_image)
    while True:
        frame = output_queue.get()
        if not frame.stops:
            video.write(frame.data)
        else:
            video.release()
            break






