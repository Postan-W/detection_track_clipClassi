from ultralytics import YOLO
from queue import Queue
from threading import Thread
from .myutils.public_logger import logger
import time

class ClimbingDetection:
    def __init__(self,input_queue:Queue,output_queue:Queue,yolo_model:str="./weights/yolov8m20240606.pt",track_config="./track_config/botsort.yaml"):
        self.yolo_model = YOLO(yolo_model)
        self.thread = Thread(target=self.task)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.id_record = {}#key:id,value:time.time()
        self.track_config = track_config

    def id_update(self,frame,threshhold:int=5):
        """
        :param frame: 自定义的frame
        :param threshhold: 行为间隔。该时间内则认为是同一个翻越动作，不重复报警。该时间外则认为是新的翻越动作
        :return:
        """
        #注：一张图片可能检测到多个翻越的box，这些box中有大于等于一个box是新出现的时候就报警
        #用本地视频抽帧测试时，这样以time.time()计时就是有问题的，因为几十秒的视频可能几秒就抽完了。但如果是实时流，则可以认为每帧的时间是和现实相近的。
        # 不过无论是每帧与现实时间比是提前了还是滞后了，使用time.time()计时确实起到了每大于等于设定时间才报警一次的直观效果
        if len(frame.boxes) == 0:
            return
        else:
            for box in frame.boxes:
                id = int(box[-1])
                if id in self.id_record.keys():
                    if int(time.time() - self.id_record[id]) < threshhold:
                        frame.alarm.append(False)
                    else:
                        frame.alarm.append(True)#如果超时了，则认为是同一个人的新的翻越行为，仍要报警
                        self.id_record[id] = time.time()
                else:
                    self.id_record[id] = time.time()
                    frame.alarm.append(True)
                    logger.info("有人翻越闸机,id是{}".format(id))

    def task(self):
        while True:
            frame = self.input_queue.get()
            #注1：If object confidence score will be low, i.e lower than track_high_thresh, then there will be no tracks successfully returned and updated.
            #注2：Tracking configuration shares properties with Predict mode, such as conf, iou, and show. For further configurations, refer to the Predict model page.
            #注3：Ultralytics also allows you to use a modified tracker configuration file. To do this, simply make a copy of a tracker config file (for example, custom_tracker.yaml) from ultralytics/cfg/trackers and modify any configurations (except the tracker_type) as per your needs.
            #注4:追踪的结果是ReID的
            result = self.yolo_model.track(source=frame.data,tracker=self.track_config,classes=[1],conf=0.3,iou=0.7,stream=False,show_labels=False,show_conf=False,show_boxes=False,save=False,save_crop=False)[0]#因为只有一张图片
            frame.boxes = result.boxes.data.tolist()#[[x1,y1,x2,y2,cls,conf,id],[]..]] or []
            # track_id = [int(i) for i in result.boxes.id.tolist()] if result.boxes.id != None else None
            # if track_id != None:
            #     print(track_id,frame.boxes)
            frame.data = result.plot()
            self.id_update(frame)
            self.output_queue.put(frame)


    def start(self):
        self.thread.start()










