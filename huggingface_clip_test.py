#代码来自于huggingface的对应模型的官方代码示例。但是相同数据的推理结果，本地跑总是比对应的在线API的结果差，没搞明白原因。暂时放弃，转向国内的模搭社区
#从在线API的测试效果看，metaclip最好
from PIL import Image
import torch
import os
from transformers import CLIPProcessor, CLIPModel
import time
proxy_url = "http://127.0.0.1:7890"
os.environ['HTTP_PROXY'] = proxy_url
os.environ['HTTPS_PROXY'] = proxy_url
# Load model directly
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
local_model_directory = "./main_code/models/metaclip-h14-fullcc2_5b"


model = AutoModelForZeroShotImageClassification.from_pretrained(local_model_directory)
processor = AutoProcessor.from_pretrained(local_model_directory)
# model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14",local_files_only=True).cuda()
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14",local_files_only=True)
# model = CLIPModel.from_pretrained("facebook/metaclip-h14-fullcc2.5b",local_files_only=True).cuda()
# processor = CLIPProcessor.from_pretrained("facebook/metaclip-h14-fullcc2.5b",local_files_only=True)


def f1():
    start = time.time()
    image_root = "./images"
    images = []
    for image in os.listdir(image_root):
        images.append(Image.open(os.path.join(image_root, image)))
    labels = "climbing over the subway gate,passing through the subway gate,walking on the road".split(",")
    inputs = processor(text=labels,images=images, return_tensors="pt", padding=True)

    # 模型在GPU上时，数据也得在GPU上
    for key, value in inputs.items():
        inputs[key] = value.cuda()
    infer_start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    print("推理耗时:{}".format(time.time()-infer_start))

    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    result = torch.argmax(probs,dim=1)
    print(result)
    print("总耗时:{}".format(time.time() - start))

def f2():
    start = time.time()
    image_root = "./images"
    for image in os.listdir(image_root):
        image = Image.open(os.path.join(image_root, image))
        labels = "climbing.yaml over the subway gate,passing through the subway gate,walking on the road".split(",")
        inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

        for key, value in inputs.items():
            inputs[key] = value.cuda()
        with torch.no_grad():
            outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image #this is the image-text similarity score
        print(logits_per_image)
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        print(probs)
        result = torch.argmax(probs, dim=1)
        print(result.item())
    print("耗时:{}".format(time.time() - start))

f1()

