# 本地摄像头推流
import queue
import threading
import cv2
import subprocess as sp

#此处换为你自己的地址
rtsp_url = 'rtsp://127.0.0.1:8554/video'
cap = cv2.VideoCapture("./videos/merged_video.mp4")
print(cap.isOpened())
# Get video information
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
command = ['ffmpeg',
               '-y',
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-pix_fmt', 'bgr24',
               '-s', "{}x{}".format(width, height),
               '-r', str(fps),
               '-i', '-',
               '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p',
               '-preset', 'ultrafast',
               '-f', 'rtsp',
               rtsp_url]
p = sp.Popen(command, stdin=sp.PIPE)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Opening camera is failed")

    else:
        p.stdin.write(frame.tostring())

"""
开启服务器mediamtx.exe，这个软件能够处理很多流类型
或者手动推流(推荐)：ffmpeg -re -stream_loop -1 -i （你的文件名） -c copy -f rtsp rtsp://127.0.0.1:8554/video
简单对参数说明
-re  是以流的方式读取
-stream_loop -1   表示无限循环读取
-i  就是输入的文件
-f  格式化输出到哪里
例：ffmpeg -re -stream_loop -1 -i ./videos/merged_video.mp4 -c copy -f rtsp rtsp://127.0.0.1:8554/video
"""