
"""
测试的图片过多，标注太耗时了，不太可行，所以这里直接用CLIP分类，把两类放到两个文件夹里，然后分别浏览两个文件夹，找到分类不合适得了例子，
在正类文件夹文件个数a,负例文件夹文件个数b，正例文件夹中找出负例的个数为c,负例文件夹找出正例的个数为d，
那么准确率为p=(a-c)/a，也就是判断为正例的真正为正例的比率；那么同时误报率就是1-p=c/a；
召回率为r=(a-c)/(a-c+d)，也就是找出来的正例占所有正例的比率；
注意剔除无用的图片，也就是上有检测到的图片质量不行，比如不属于任何一个描述文本，导致分类器判断困难。
"""

from PIL import Image
import torch
import numpy as np
import os
from transformers import CLIPProcessor, CLIPModel
import time

model_load_start = time.time()
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
print("加载模型耗时:{}".format(time.time()-model_load_start))

source_image = r"D:\wmingzhu\climbing_detection\temp_crops"
positive_destination = r"D:\wmingzhu\climbing_detection\classi_result\positive"
negative_desitnation = r"D:\wmingzhu\climbing_detection\classi_result\negative"

batch_size = 100

start = time.time()

images_path = [os.path.join(source_image,i) for i in os.listdir(source_image)]
num = len(images_path)

steps = int(len(images_path) / batch_size)

for i in range(1,steps+2):
    load_image_start = time.time()
    batch_images_path = [i for i in images_path[(i-1)*batch_size:i*batch_size if i*batch_size<num else num]]
    images = [Image.open(i) for i in batch_images_path]
    print("图片加载耗时:{}".format(time.time()-load_image_start))
    input_process_start = time.time()
    inputs = processor(text=["climbing over the subway gate","passing through the subway gate", "walking on the road"],
                       images=images, return_tensors="pt", padding=True)
    print("数据处理耗时:{}".format(time.time()-input_process_start))
    move_start = time.time()
    # 模型在GPU上时，数据也得在GPU上，但显存不够而且有IO开销
    for key, value in inputs.items():
        inputs[key] = value.cuda()
    print("数据转移到GPU上耗时:{}".format(time.time()-move_start))

    infer_start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    print("推理耗时:{}".format(time.time()-infer_start))

    result_process_start = time.time()
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    result = torch.argmax(probs,dim=1)
    # print("结果处理后所在设备:{}".format(result.device))
    print("推理结果处理耗时:{}".format(time.time()-result_process_start))
    for i,label in enumerate(result.tolist()):
        if label == 0:
            images[i].save(os.path.join(positive_destination,os.path.split(batch_images_path[i])[1]))
        else:
            images[i].save(os.path.join(negative_desitnation,os.path.split(batch_images_path[i])[1]))





