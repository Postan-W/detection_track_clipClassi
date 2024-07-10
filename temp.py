import cv2
import time

cap = cv2.VideoCapture("./new_fanyue.avi")
ret, frame = cap.read()
while ret:
    print("正在读取视频帧")
    time.sleep(0.5)
    ret, frame = cap.read()