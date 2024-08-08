from ultralytics import YOLO
from get_input import VideoReader
from queue import Queue
from threading import Thread
import cv2
import glob
from ultralytics import YOLO
from PIL import Image
import time
from myutils.cv2_utils import plot_boxes_with_text_for_yolotrack
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class ClimbingFallingDetection:
    def __init__(self,input_queue:Queue,output_queue:Queue,yolo_model:str="./weights/cimb_fall_1280.engine",track_config="./track_config/botsort.yaml",crop_count = 0):
        self.yolo_model = YOLO(yolo_model)
        self.thread = Thread(target=self.task)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.id_record = {}#key:id;value:time.time()
        self.track_config = track_config
        self.crop_count = crop_count
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
                if len(box) == 7:
                    id = int(box[-3])
                    if id in self.id_record.keys():
                        if int(time.time() - self.id_record[id]) < threshhold:
                            frame.alarm.append(False)
                            # logger.info("**********有人翻越，但无需重复报警********")
                        else:
                            frame.alarm.append(True)  #如果超时了，则认为是同一个人的新的翻越行为，仍要报警
                            self.id_record[id] = time.time()
                            # logger.info("======同一人翻越，但时间间隔超过阈值，需报警==========")
                    else:
                        self.id_record[id] = time.time()
                        frame.alarm.append(True)
                        #logger.info("###############有新目标翻越，id是{},需报警#################".format(id))
                else:
                    #logger.info("box的长度是:{},因为低于追踪设定的最小置信度，所以没有id"
                    frame.alarm.append(False)

    def task(self):
        while True:
            frame = self.input_queue.get()
            if not frame.stops:
                result = self.yolo_model.track(persist=True,verbose=False,imgsz=1280,source=frame.data,tracker=self.track_config,classes=[0,1],conf=0.6,iou=0.6,stream=False,show_labels=False,show_conf=False,show_boxes=False,save=False,save_crop=False)[0]#因为只有一张图片
                frame.boxes = result.boxes.data.tolist()#[[x1,y1,x2,y2,id,conf,cls],[]..]] or []
                for box in frame.boxes:
                    class_name = "climb" if box[-1] == 0 else "fall"
                    color = [255,0,0] if box[-1] == 0 else [0,0,255]
                    plot_boxes_with_text_for_yolotrack([box], frame.data, color=color,class_name=class_name)

                self.id_update(frame)
                self.output_queue.put(frame)
            else:
                self.output_queue.put(frame)
                break

    def start(self):
        self.thread.start()

if __name__ == '__main__':
    videos = glob.glob("C:/Users/wmingdru/Desktop/clear_videos/*")
    crop_count = 0
    all_start = time.perf_counter()
    video_num = len(videos)
    for video_index,video in enumerate(videos):
        current_start = time.perf_counter()
        input_queue = Queue(1000)
        output_queue = Queue(1000)
        video_path = video
        output_path = os.path.join("./outputs/",os.path.splitext(os.path.split(video)[1])[0]+".mp4")
        video_reader = VideoReader(video_path=video_path,image_queue=input_queue,timestep=1)
        total_frames = int(video_reader.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_reader.start()
        climbing_detection = ClimbingFallingDetection(input_queue,output_queue,crop_count=crop_count)
        climbing_detection.start()
        first_image = output_queue.get().data
        height, width, _ = first_image.shape
        video = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'X264'), 30, (width, height))
        # print((width, height))
        video.write(first_image)
        processed_count = 0
        while True:
            # time.sleep(1/30)#模拟下实时流，30fps
            frame = output_queue.get()
            if not frame.stops:
                # if len(frame.boxes) != 0:
                #     if True in frame.alarm:
                #         logger.info("有人翻越闸机")
                #     else:
                #         logger.info("有人翻越闸机，但无需重复报警")

                video.write(frame.data)
                processed_count += 1
                print("\r当前视频:{}=={}/{},{}%,当前耗时:{}==,总进度:{}/{},总耗时:{}".format(os.path.split(video_path)[1],processed_count, total_frames, round((processed_count / total_frames) * 100, 2),round(time.perf_counter()-current_start,1),video_index+1,video_num,round(time.perf_counter()-all_start,1)),end="")
            else:
                video.release()
                break






