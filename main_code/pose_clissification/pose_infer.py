import torch
from ultralytics import YOLO
import cv2
from pose_data_structure import action_list
from pose_utils import plot_boxes_with_text_single_box,keypoints_filter
import glob,os

device = torch.device("cuda")
pose_model = YOLO("../weights/yolov8l-pose.engine")
classi_model = torch.load("./models/best_epoch_in_val.pt").to(device)
classi_model.eval()#在评估模式下，所有特定于训练的层(dropout和batchnorm等)将被设置为不活动。但我的模型本来就只是全连接，所以这句话没起作用

def infer_on_video(test_video,output_path):
    with torch.no_grad():  # 在forward时不会存储为计算梯度能用到的中间值
        cap = cv2.VideoCapture(test_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = cap.read()
        height, width, _ = frame.shape
        video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
        processed_count = 0
        while ret:
            result = \
            pose_model.track(frame, save=False, verbose=False, persist=True, tracker="../track_config/botsort.yaml")[0]
            boxes = result.boxes.data.cpu().numpy()
            xyn = result.keypoints.xyn.cpu().numpy()  # normalized keypoints
            if not len(boxes) == 0:
                for i, box in enumerate(boxes):
                    keypoints = xyn[i]
                    if keypoints_filter(keypoints):
                        action_result = classi_model(
                            torch.tensor(keypoints[3:].flatten(), dtype=torch.float32).unsqueeze(0).to(device))
                        action_index = action_result.argmax(dim=1).item()
                        action_name = action_list[action_index]
                        text_info = action_name
                        plot_boxes_with_text_single_box(box, frame, text_info=text_info)

            video.write(frame)
            processed_count += 1
            print("{}/{},{}%".format(processed_count, total_frames, round((processed_count / total_frames) * 100, 2)))
            ret, frame = cap.read()

        cap.release()
        video.release()

videos_dir = glob.glob("../../videos/fanyue/*.mp4")
output_dir = "./output"

for video in videos_dir:
    temp = os.path.splitext(os.path.split(video)[1])
    infer_on_video(test_video=video, output_path=os.path.join(output_dir,temp[0]+"_infered"+temp[1]))