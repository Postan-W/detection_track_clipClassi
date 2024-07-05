#Author wmingzhu
#Date 20240701
from ultralytics import YOLO
import cv2
from pose_utils import plot_boxes_with_text_single_box,keypoints_filter
import os
from pose_data_structure import action_list
import glob

model = YOLO("../weights/yolov8l-pose.engine")
videos = glob.glob("../../videos/pose_videos/*")
print(videos)
output_path = "train_data/train.txt"

def input_action():
    action = ""
    while not action in (action_list + ["p"]):#p代表不为当前box标注
        action = input("输入动作名称:")
    print("标签是:{}".format(action))
    return action


#为了方便中断标注后，再次标注时也能够了解当前各个类别已经标注了多少个
def load_action_counter(data_path,action_list):
    action_counter = {i: 0 for i in action_list}

    with open(data_path, 'r') as f:
        for line in f:
            action = line.strip().split(",")[0]
            try:
                action_counter[action] += 1
            except:
                action_counter[action] = 0

    return action_counter

action_counter = load_action_counter(output_path,action_list) if os.path.exists(output_path) else {i: 0 for i in action_list}
print("已标注:{}".format(action_counter))

def annotator(videos):
    for video_path in videos:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        count = 1
        with open(output_path, 'a') as f:
            while ret:
                if count % 15 == 0:  # 跳帧标注
                    try:
                        result = model.track(frame, save=False, verbose=False, persist=True,tracker="../track_config/botsort.yaml")[0]
                    except:
                        print("********帧数据有问题，跳过********")
                        continue
                    boxes = result.boxes.data.cpu().numpy()
                    xyn = result.keypoints.xyn.cpu().numpy()
                    if not len(boxes) == 0:
                        for i, box in enumerate(boxes):
                            text_info = "box:" + str(i)  # index最大的那个就是当前要标注的那个box(cv2.imshow的标题也会提示当前是哪个box)
                            plot_boxes_with_text_single_box(box, frame, text_info=text_info)
                            cv2.imshow("current box:{}".format(i), frame)
                            key = cv2.waitKey(0)
                            if key == ord('q'):
                                cv2.destroyAllWindows()
                            action = input_action()  #手动输入的标签
                            if action == "p":
                                print("不为当前box标注")
                                cv2.destroyAllWindows()
                                continue
                            else:
                                keypoints = xyn[i]
                                # 过滤
                                if keypoints_filter(keypoints):
                                    action_counter[action] += 1
                                    print(action_counter)
                                    keypoints = [str(i) for i in keypoints[3:].flatten()]
                                    keypoints = ",".join(keypoints)
                                    f.write(action + "," + keypoints + "\n")

                ret, frame = cap.read()
                count += 1
        cap.release()

annotator(videos=videos)