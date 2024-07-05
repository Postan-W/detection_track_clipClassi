#更详细和强大的参数使用(认真看了以下文档，可以找到所以我做下游开发任务需要的参数或方法)，参见https://docs.ultralytics.com/modes/predict/#why-use-ultralytics-yolo-for-inference
from ultralytics import YOLO
# Load a model
model = YOLO("yolov8m.pt")  # pretrained YOLOv8n model.type:nsmlx
print(model.names)

def method1(data):
    # Run batched inference on a list of images
    results = model(data)  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename="result.jpg")  # save to disk

def method2(data):
     # Run batched inference on a list of images
    results = model(data,stream=True)  # return a generator of Results objects

    # Process results generator
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename="result.jpg")  # save to disk

source = "../videos/pose_videos/fanyue_0703_4.avi"
# results = model(source,classes=[0])
results = model(source,classes=[0],save=True,stream=True)#如果stream=True，则返回的results是一个generator,所以必须在下面使用for循环遍历这个生成器才能有结果(即便for循环里什么逻辑都不写)
for result in results:
    print(result.boxes.xyxy.tolist())
    # print(result.orig_img.shape)#hwc
    # result.save(filename="result.jpg")
    # print("------------------------------------------")
