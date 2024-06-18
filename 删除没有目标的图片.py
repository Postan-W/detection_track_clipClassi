"""
在使用标注工具标注yolo目标检测图片时，有些图片没有目标，所以直接跳过，所以labels文件夹
下没有其对应的标签，这样的图片将其删除
"""
import os
import glob
images_dir = "./subway_images/val/images"
labels_dir = "./subway_images/val/labels"

def delete_images(images_dir,labels_dir):
    images = glob.glob(os.path.join(images_dir, "*.jpg"))
    labels = [os.path.splitext(os.path.split(i)[1])[0] for i in glob.glob(os.path.join(labels_dir, "*.txt"))]
    labels.remove('classes')

    for image in images:
        if not os.path.splitext(os.path.split(image)[1])[0] in labels:
            os.remove(image)

# delete_images(images_dir,labels_dir)

#下面是标签不小心多了，删除标签
def delete_labels(images_dir,labels_dir):
    images = [os.path.splitext(os.path.split(i)[1])[0] for i in glob.glob(os.path.join(images_dir,"*.jpg"))]
    labels = glob.glob(os.path.join(labels_dir,"*.txt"))
    for label in labels:
        name = os.path.splitext(os.path.split(label)[1])[0]
        if not name in images:
            if name != "classes":
                os.remove(label)
            else:
                print("classes.txt不用删除")

# delete_labels(images_dir,labels_dir)



