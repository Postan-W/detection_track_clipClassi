import joblib
import torch
from ultralytics import YOLO
import cv2
from pose_data_structure import action_list
from pose_utils import plot_boxes_with_text_single_box,keypoints_filter
import glob,os
import numpy as np
import lightgbm as lgb
device = torch.device("cuda")
pose_model = YOLO("../weights/yolov8x-pose.engine")
classi_model = lgb.Booster(model_file="./models/lgbm.txt")
# classi_model = joblib.load("./models/best_gbm.joblib")

def infer_on_video(test_video,output_path):
    with torch.no_grad():  #在forward时不会存储为计算梯度能用到的中间值
        cap = cv2.VideoCapture(test_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = cap.read()
        height, width, _ = frame.shape
        video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'X264'), 30,(width, height))
        processed_count = 0
        while ret:
            try:
                result = pose_model.track(frame, save=False, verbose=False, persist=True,tracker="../track_config/botsort.yaml")[0]
            except:
                break
            boxes = result.boxes.data.cpu().numpy()
            xyn = result.keypoints.xyn.cpu().numpy()  #normalized keypoints
            if not len(boxes) == 0:
                for i, box in enumerate(boxes):
                    keypoints = xyn[i]
                    if keypoints_filter(keypoints):
                        action_result = classi_model.predict(torch.tensor(keypoints[3:].flatten(),dtype=torch.float32).unsqueeze(0).tolist())
                        action_index = np.argmax(action_result, axis=1)[0]
                        action_probability = round(action_result[0][action_index],2)
                        action_name = action_list[action_index]
                        text_info = action_name + " p:{}".format(action_probability) + " conf:" + (str(round(box[4],2)) if len(box) == 6 else str(round(box[5],2)))
                        box_conf = round(box[4],2) if len(box) == 6 else round(box[5],2)#被遮挡的人体部位的关键点总是被误检，但conf应该是低的，所以用conf过滤到这种情况

                        #需要再加几何上的过滤规则，比如俯视的时候头在最上面?可以从这个入手？
                        if action_name in ["climb"]:
                            if action_probability > 0.93 and box_conf > 0.82:
                                plot_boxes_with_text_single_box(box, frame, color=[255, 0, 0],text_info=text_info)
                        elif action_name in ["fall"]:
                            if action_probability > 0.93 and box_conf > 0.82:
                                plot_boxes_with_text_single_box(box, frame, text_info=text_info)
                        else:
                            plot_boxes_with_text_single_box(box, frame, color=[0, 255, 0], text_info=text_info)


            video.write(frame)
            processed_count += 1
            print("{}/{},{}%".format(processed_count, total_frames, round((processed_count / total_frames) * 100, 2)))
            ret, frame = cap.read()

        cap.release()
        video.release()

videos_dir = glob.glob("../../videos/pose_infer/*")
output_dir = "./output/"

for video in videos_dir:
    temp = os.path.splitext(os.path.split(video)[1])
    infer_on_video(test_video=video, output_path=os.path.join(output_dir,temp[0]+"_infered_lgbm"+".mp4"))