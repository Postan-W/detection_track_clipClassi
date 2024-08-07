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
pose_model = YOLO("../../weights/yolov8x-pose.engine")
classi_model = lgb.Booster(model_file="./models/su_plus_jinan3_lgbm.txt")
# classi_model = joblib.load("./models/best_gbm.joblib")


def infer_on_video(test_video,output_path):
    fall_count = 0
    climb_count = 0
    with torch.no_grad():
        cap = cv2.VideoCapture(test_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = cap.read()
        height, width, _ = frame.shape
        video = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'X264'), 30,(width, height))
        processed_count = 0
        while ret:
            try:
                result = pose_model(frame, save=False, verbose=False)[0]
            except:
                break
            boxes = result.boxes.data.cpu().numpy()
            xyn = result.keypoints.xyn.cpu().numpy()  #normalized keypoints
            xy = result.keypoints.xy.cpu().numpy()
            if not len(boxes) == 0:
                for i, box in enumerate(boxes):
                    keypoints = xyn[i]
                    if keypoints_filter(keypoints):
                        action_result = classi_model.predict(torch.tensor(keypoints[3:].flatten(),dtype=torch.float32).unsqueeze(0).tolist())
                        action_index = np.argmax(action_result, axis=1)[0]
                        action_probability = round(action_result[0][action_index],2)
                        action_name = action_list[action_index]
                        text_info = action_name + " p:{}".format(action_probability) + " conf:" + (str(round(box[4],2)) if len(box) == 6 else str(round(box[5],2)))
                        text_info = "*"*30 + text_info + "*"*30
                        box_conf = round(box[4],2) if len(box) == 6 else round(box[5],2)#被遮挡的人体部位的关键点总是被误检，但conf应该是低的，所以用conf过滤到这种情况
                        for point in xy[i]:  # 画关键点
                            x, y = int(point[0]), int(point[1])
                            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                        if ((action_probability > 0.95) and (box_conf > 0.8)) or ((action_probability >= 0.92) and (box_conf >= 0.85)):
                            if action_name in ["climb"]:
                                plot_boxes_with_text_single_box(box, frame, color=[255, 0, 0], text_info=text_info)
                                # climb_count += 1
                                # cv2.imwrite("C:/Users/wmingdru/Desktop/workspace/projects/detection_track_clipClassi/main_code/pose_clissification/pose2/crops/climb_{}.jpg".format(climb_count),frame)
                            elif action_name in ["fall"]:
                                plot_boxes_with_text_single_box(box, frame, text_info=text_info)
                                # fall_count += 1
                                # cv2.imwrite("C:/Users/wmingdru/Desktop/workspace/projects/detection_track_clipClassi/main_code/pose_clissification/pose2/crops/fall_{}.jpg".format(fall_count), frame)


            video.write(frame)
            processed_count += 1
            print("{}/{},{}%".format(processed_count, total_frames, round((processed_count / total_frames) * 100, 2)))
            ret, frame = cap.read()

        cap.release()
        video.release()

# videos_dir = glob.glob("../../../videos/suzhou_train/*")
# videos_dir = glob.glob("C:/Users/wmingdru/Desktop/pose_train_videos/*")
videos_dir = glob.glob("../../../videos/pose/*")
output_dir = "./output/"

for video in videos_dir:
    temp = os.path.splitext(os.path.split(video)[1])
    infer_on_video(test_video=video, output_path=os.path.join(output_dir,temp[0]+"_infered_lgbm"+".mp4"))