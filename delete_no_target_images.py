"""
在使用标注工具标注yolo目标检测图片时，有些图片没有目标，所以直接跳过，所以labels文件夹
下没有其对应的标签，这样的图片将其删除
"""
images_dir = "../subway_images/train/images"
labels_dir = "../subway_images/train/labels"

