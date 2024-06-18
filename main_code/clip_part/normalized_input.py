#来自https://github.com/Postan-W/Crop-CLIP/blob/master/Crop_CLIP.ipynb

from PIL import Image, ImageFont
import os
import cv2
import torch
import glob
import sys
import warnings
import numpy as np

from PIL import Image


import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

images = torch.stack([preprocess(im) for im in l]).to(device)
with torch.no_grad():
  image_features = model.encode_image(images)
  image_features /= image_features.norm(dim=-1, keepdim=True)

image_features.cpu().numpy()
image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()

images = [preprocess(im) for im in l]
image_input = torch.tensor(np.stack(images)).cuda()
image_input -= image_mean[:, None, None]
image_input /= image_std[:, None, None]
with torch.no_grad():
    image_features = model.encode_image(image_input).float()
image_features /= image_features.norm(dim=-1, keepdim=True)

def similarity_top(similarity_list,N):
  results = zip(range(len(similarity_list)), similarity_list)
  results = sorted(results, key=lambda x: x[1],reverse= True)
  top_images = []
  scores=[]
  for index,score in results[:N]:
    scores.append(score)
    top_images.append(l[index])
  return scores,top_images