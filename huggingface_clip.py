import torch
from PIL import Image
import open_clip
import os

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='metaclip_fullcc')  # for 2.5B use 'metaclip_fullcc' in OpenCLIP or 'metaclip_2_5b' in this repo

image_root = "./images/"
images = [Image.open(i) for i in [os.path.join(image_root,j) for j in os.listdir(image_root)]]
images_trans = []
for image in images:
    images_trans.append(preprocess(image).tolist())

images_trans = torch.tensor(images_trans, dtype=torch.float32)


text = open_clip.tokenize("climbing.yaml over the subway gate,passing through the subway gate,walking on the road".split(","))

with torch.no_grad():
    image_features = model.encode_image(images_trans)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    result = torch.argmax(text_probs, dim=-1)
print("Label probs:", result)