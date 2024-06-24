import cv2
from threading import Thread
from queue import Queue
from data_structure import Frame
from myutils.public_logger import logger

class VideoReader:
    def __init__(self, video_path,image_queue:Queue,timestep=1):
        self.path = video_path
        self.thread = Thread(target=self.task)
        self.queue = image_queue
        self.timestep = timestep

    def task(self):
        cap = cv2.VideoCapture(self.path)
        ret, frame = cap.read()
        count = 1
        while ret:
            if count % self.timestep == 0 and ret:
                self.queue.put(Frame(data=frame))
            ret, frame = cap.read()
            count += 1

        logger.info("视频抽帧已完成")
        self.queue.put(Frame(stops=True))#终止指示帧


        cap.release()
        cv2.destroyAllWindows()

    def start(self):
        self.thread.start()

