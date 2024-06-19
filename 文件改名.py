import os



dir_path = "./videos/val_video/"

files = [os.path.join(dir_path,file) for file in os.listdir(dir_path)]

for i in range(len(files)):
    os.rename(files[i],os.path.join(dir_path,"smallval"+str(i)+".mp4"))

