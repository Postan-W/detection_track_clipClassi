#随机从所有的图片中选取一部分作为验证集，剩下的作为训练集

import os
import random
import shutil

image_path = "C:/Users/wmingdru/Desktop/workspace/data/20240726climb_fall/train/images/"
destination = "C:/Users/wmingdru/Desktop/workspace/data/20240726climb_fall/val/images/"

image_path = [os.path.join(image_path,image) for image in os.listdir(image_path)]

image_count = len(image_path)
print("总文件数:{}".format(image_count))

selected_count = int(image_count*0.1)

selected_index = []
for i in range(selected_count):
    index = random.randint(0,image_count-1)
    if not index in selected_index:
        selected_index.append(index)

print(len(selected_index),selected_index)

for image in selected_index:
    shutil.move(image_path[image],destination)