from detector import *
import time
import sys #版本、模型路径、摄像头路数、设备名称
from torch.multiprocessing import Process,Queue
model_version = sys.argv[1]
model_path = sys.argv[2]
camera_count = int(sys.argv[3])
cuda_device = sys.argv[4]


def task(input_video,output_video,area_point_list = [[[315, 34], [619, 49], [585, 585], [335, 578]]],index=1):
    model = YOLOv5Detector(model=model_path,device=cuda_device) if model_version == "v5" else YOLOv8Detector(model=model_path,device=cuda_device)
    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    height, width, _ = frame.shape
    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
    video.write(frame)
    processed_count = 0
    start_time = time.perf_counter()
    print("\n任务{}启动".format(index))
    while ret:
        model.inference(frame, area_point_list)
        video.write(frame)
        if processed_count % 100 == 0:
            print("\r任务{},{}/{},{}%,{}s".format(index,processed_count, total_frames, round((processed_count / total_frames) * 100, 2),round(time.perf_counter()-start_time),2),end="")
        processed_count += 1
        ret, frame = cap.read()
    print("\n任务{}结束,总耗时:{}".format(index,time.perf_counter()-start_time))

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    p_list = []
    for i in range(camera_count):
        p = Process(target=task,args=("1.mp4","output_{}.mp4".format(i+1),[[[315, 34], [619, 49], [585, 585], [335, 578]]],i+1))
        p_list.append(p)
    for i in range(camera_count):
        p_list[i].start()
    for i in range(camera_count):
        p_list[i].join()
    print("===所有任务运行结束===")