from ultralytics import YOLO
# Load a model
model = YOLO("../weights/yolov8l-pose.pt")  # load an official model

print(model.names)#人体姿势估计，检测的对象只有人
source_path = "../clip_images/pj1_120.jpg"
# Predict with the model
results = model.track("../clip_images/dog.jpeg",save=True,persist=True,tracker="../track_config/botsort.yaml")
result = results[0]
orig_shape = result.orig_img.shape #hwc
#xy->(0，0)意味着关键点置信度过低或者根本不可见所以没检测到
boxes = result.boxes.data.cpu().numpy()
print(boxes,len(boxes),boxes.shape)#空的boxes的len时0
print("xy在原图上的关键点坐标:")
xy = result.keypoints.xy.cpu().numpy()
print(xy,len(xy),xy.shape)#空的xy的len是1，这个要注意
print("框架给出的归一化的值:")
xyn = result.keypoints.xyn.cpu().numpy()
print(xyn)
print("手动归一化,x/w,y/h的结果:")
# print(xy.dtype,xyn.dtype)
xy[:,:,0] = xy[:,:,0] / orig_shape[1]
xy[:,:,1] = xy[:,:,1] / orig_shape[0]
print(xy)
#可知xyn就是x/w,y/h的结果，所以项目中直接用xyn即可





