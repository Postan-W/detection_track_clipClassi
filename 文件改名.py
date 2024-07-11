import os



dir_path = "C:/Users/wmingdru/Desktop/workspace/data/suzhou_videos/camera1/"

files = [os.path.join(dir_path,file) for file in os.listdir(dir_path)]

for i in range(len(files)):
    os.rename(files[i],os.path.join(dir_path,"suzhoucamera1_"+str(i)+".avi"))

