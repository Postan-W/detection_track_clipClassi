from ultralytics import YOLO
import cv2
from myutils.cv2_utils import plot_boxes_with_text_single_box
def input_action(xyn=None):
    """
    :param xyn: [x1,y1,x2,y2...,x17,y17]
    :return:
    """
    action = ""
    while not action in ["stand","sit","fall","squat","bend","climb","straddle","p"]:#p代表不为当前box标注
        action = input("输入动作名称:")
    print("标签是:{}".format(action))
    return action


model = YOLO("../weights/yolov8l-pose.pt")
video_path = "../../videos/fanyue/new_fanyue.avi"
output_path = "./data/train.txt"
cap = cv2.VideoCapture(video_path)
ret,frame = cap.read()
count = 1
with open(output_path, 'a') as f:
    while ret:
        if count % 15 == 0:  # 每n帧标注一帧
            result = model.track(frame, save=False, verbose=False, persist=True, tracker="../track_config/botsort.yaml")[0]
            boxes = result.boxes.data.cpu().numpy()
            xyn = result.keypoints.xyn.cpu().numpy()
            if not len(boxes) == 0:
                for i, box in enumerate(boxes):
                    text_info = "index:" + str(i)  # index最大的那个就是当前要标注的那个box(cv2.imshow的标题也会提示当前是哪个box)
                    plot_boxes_with_text_single_box(box, frame, text_info=text_info)
                    cv2.imshow("current box:{}".format(i), frame)
                    cv2.waitKey(0)
                    action = input_action()#手动输入的标签
                    if action == "p":
                        print("不为当前box标注")
                        cv2.destroyAllWindows()
                        continue
                    else:
                        keypoints = ",".join([str(i) for i in xyn[i].flatten()])
                        f.write(action+","+keypoints+"\n")

                    cv2.destroyAllWindows()
        ret, frame = cap.read()
        count += 1



