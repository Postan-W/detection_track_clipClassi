import os



dir_path = "C:/Users/wmingdru/Desktop/workspace/data/shuaidao/videos/wanyao/"

files = [os.path.join(dir_path,file) for file in os.listdir(dir_path)]

for i in range(len(files)):
    os.rename(files[i],os.path.join(dir_path,"wanyao"+str(i)+".mp4"))

