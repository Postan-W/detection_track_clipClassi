# author: wmingzhu
# date: 2024/07/02

import torch
from torch.utils.data import Dataset,DataLoader
from pose_data_structure import Action,KeyPoints
import cv2
import numpy as np

#在构建数据的时候标签使用的是名称，这样比索引更直观，而且避免出错,这里在训练前把名称转为index
def name2index(data_path,output_path,action=Action().dict()):
    with open(data_path,"r") as f_in:
        with open(output_path,"w") as f_out:
            for line in f_in:
                t =  line.strip().split(",")
                label = action[t[0].upper()]
                t[0] = label
                f_out.write(",".join(map(str,t)) + "\n")

# name2index("./train_data/train.txt","./train_data/train_indexed.txt")

def plot_boxes_with_text_single_box(box, img, color=[0,0,255], text_info="None",velocity=None, thickness=1, fontsize=0.5, fontthickness=1):
    # Plots bounding boxes on image img
   c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
   cv2.rectangle(img, c1, c2, color, thickness, lineType=cv2.LINE_AA)
   t_size = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize, fontthickness + 2)[0]
   cv2.rectangle(img, c1, (c1[0] + int(t_size[0]), c1[1] + int(t_size[1] * 1.45)), color, -1)
   cv2.putText(img, text_info, (c1[0], c1[1] + t_size[1] + 2),cv2.FONT_HERSHEY_TRIPLEX, fontsize,color=[255, 255, 255], thickness=fontthickness)
   return img

#一些关键点被遮挡的数据影响模型训练和推理效果。在标注数据写入文件之前以及姿势估计的关键点数据被送入到分类器之前做过滤
def keypoints_filter(keypoints:np.array=None)->bool:
    """
    :param keypoints: [[x1,y1],[x2,y2],...[x17,y17]]
    :return: True|False
    """
    d = {}
    zeros = 0 #被遮挡的关键点的个数
    for i,name in enumerate(list(KeyPoints().dict().keys())):#python3.6以后keys是有序的
        if i in [0,1,2]:#舍弃的关键点
            continue
        else:
            xy_sum = sum(keypoints[i])
            if xy_sum == 0:
                zeros += 1
            d[name] = xy_sum

    # 这里设定两个过滤规则，关于过滤规则还待探究
    if zeros >= 8:
        print("***遮挡过多,无效数据***")
        return False

    #如果左膝右膝同时没检测到，以及左脚踝和右脚踝同时没检测到，则认为是无效数据，不能用来分类
    if sum([d["LEFT_KNEE"],d["RIGHT_KNEE"]]) == 0 and sum([d["LEFT_ANKLE"],d["RIGHT_ANKLE"]]) == 0:
        print("***双膝以及双踝被遮挡，无效数据***")
        return False

    return True


# a = np.array([[0.33,0.27],[0.33,0.27],[0.0,0.0],[0.33,0.27],[0,0],[0,0],[0.33,0.27],[0,0],[0,0],[0.33,0.27],[0,0],[0.33,0.27],[0.33,0.27],[0.33,0.27],[0.33,0.27],[0,0],[0,0]])
# b = np.array([[0.33,0.27],[0.33,0.27],[0.33,0.27],[0.33,0.27],[0.33,0.27],[0.33,0.27],[0.33,0.27],[0.33,0.27],[0.33,0.27],[0.33,0.27],[0,0],[0.33,0.27],[0.33,0.27],[0.33,0.27],[0.33,0.27],[0.33,0.27],[0.33,0.27]])
# print(keypoints_filter(b))


