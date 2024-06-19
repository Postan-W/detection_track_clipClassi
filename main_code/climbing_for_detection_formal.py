from ultralytics import YOLO
from get_input import VideoReader
from queue import Queue
from threading import Thread
from myutils.public_logger import logger
import cv2
import torch
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
from modelscope.preprocessors.image import load_image
from ultralytics import YOLO
from PIL import Image
import time
from myutils.cv2_utils import plot_boxes_with_text_for_yolotrack
class ClimbingDetection:
    def __init__(self,input_queue:Queue,output_queue:Queue,yolo_model:str="./weights/yolov8l20240618.pt",track_config="./track_config/botsort.yaml"):
        self.yolo_model = YOLO(yolo_model)
        self.thread = Thread(target=self.task)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.id_record = {}#key:id;value:time.time()
        self.track_config = track_config
        #clip模型暂时成固定的
        self.clip_model = pipeline(task=Tasks.multi_modal_embedding,model='damo/multi-modal_clip-vit-large-patch14_336_zh', model_revision='v1.0.1')
        #暂时写成固定代码，包括下游处理的时候也是根据input_texts的0和1个prompt做筛选的
        self.input_texts = ["有人在弯腰翻越障碍物", "有人双手支撑在障碍物上攀爬", "有人笔直地站着", "人笔直通过障碍物",
                       "有人从障碍物旁边走过", "画面里没有人","有人低头看手机"]
        self.text_embedding = self.clip_model.forward({'text': self.input_texts})['text_embedding']

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
                #注1：If object confidence score will be low, i.e lower than track_high_thresh, then there will be no tracks successfully returned and updated.
                #注2：Tracking configuration shares properties with Predict mode, such as conf, iou, and show. For further configurations, refer to the Predict model page.
                #注3：Ultralytics also allows you to use a modified tracker configuration file. To do this, simply make a copy of a tracker config file (for example, custom_tracker.yaml) from ultralytics/cfg/trackers and modify any configurations (except the tracker_type) as per your needs.
                #注4:追踪的结果是ReID的,需要persist=True
                result = self.yolo_model.track(persist=True,verbose=True,source=frame.data,tracker=self.track_config,classes=[1],conf=0.3,iou=0.7,stream=False,show_labels=False,show_conf=False,show_boxes=False,save=False,save_crop=False)[0]#因为只有一张图片
                frame.boxes = result.boxes.data.tolist()#[[x1,y1,x2,y2,id,conf,cls],[]..]] or []
                # track_id = [int(i) for i in result.boxes.id.tolist()] if result.boxes.id != None else None
                # if track_id != None:
                #     print(track_id,frame.boxes)
                # frame.data = result.plot()
                if not len(frame.boxes) == 0:
                    final_boxes = []

                    clip_input = []
                    for box in frame.boxes:
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        croped = frame.data[y1:y2, x1:x2, :]
                        rgb_image = cv2.cvtColor(croped, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(rgb_image)
                        clip_input.append(pil_image)
                    img_embedding = self.clip_model.forward({'img': clip_input})['img_embedding']
                    with torch.no_grad():
                        # 计算内积得到logit，考虑模型temperature
                        logits_per_image = (img_embedding / self.clip_model.model.temperature) @ self.text_embedding.t()
                        # 根据logit计算概率分布
                        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                    clean_result = []
                    for img_result in probs:
                        clean_result.append([round(i, 2) for i in img_result])

                    for i, prob in enumerate(clean_result):
                        # print(i, prob)
                        target_prob = prob[0] + prob[1]
                        residual_prob = 1 - target_prob
                        # print(target_prob, residual_prob)
                        if target_prob > residual_prob:
                            final_boxes.append(list(frame.boxes[i]))
                    frame.boxes = final_boxes#更新boxes
                    plot_boxes_with_text_for_yolotrack(frame.boxes, frame.data, class_name="climb")#画框

                self.id_update(frame)
                self.output_queue.put(frame)
            else:
                self.output_queue.put(frame)
                break

    def start(self):
        self.thread.start()

if __name__ == '__main__':
    input_queue = Queue(1000)
    output_queue = Queue(1000)
    video_path = "../videos/output/allscenes_merged.mp4"
    output_path = "outputs/allscenes_result_l618_with_clip.mp4"
    video_reader = VideoReader(video_path=video_path,image_queue=input_queue,timestep=1)
    video_reader.start()
    climbing_detection = ClimbingDetection(input_queue,output_queue)
    climbing_detection.start()
    first_image = output_queue.get().data
    height, width, _ = first_image.shape
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
    # print((width, height))
    video.write(first_image)
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
        else:
            video.release()
            break






