import os

dir_path = "../yolo_val_videos"

files = [os.path.join(dir_path,file) for file in os.listdir(dir_path)]

for i in range(len(files)):
    os.rename(files[i],os.path.join(dir_path,"valvideo"+str(i)+".mp4"))