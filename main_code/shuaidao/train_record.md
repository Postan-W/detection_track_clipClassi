# 注：将检测到的判断为某类行为的人体送到clip分类时，截的图不应该只是box(比如截取范围在box基础上增加原图的1/5)，因为box一是太小，经过下游缩放就更不清晰了，二是box中只有人体，几乎看不到环境信息，所以clip可能很难发挥作用。不过截取时要考虑边界。

# 1.20240624

设置了5种类别，即蹲下squat、摔倒fall、站立stand、弯腰bend、坐sit。

预训练模型选的是yolov8m.pt，不用更大的模型的原因是本次训练的数据较少，更大的模型预计效果反而很难被训练出。

epoch=20;batch=32;

使用的服务器：188.18.18.212//home/ai/wmz/falling

## 2.20240624

workers=16

device=[0,1,2,3,4,5,6,7]

batch=64

epoch=100

设置了5种类别，即蹲下、摔倒、站立、弯腰、坐。

基础模型选用yolov8l.pt(甚至可以选用更大的)，212机器上有8张GPU。

设置了5种类别，即蹲下squat、摔倒fall、站立stand、弯腰bend、坐sit。

squat
stand
bend
fall
sit

训练图片一共379张"C:\Users\wmingdru\Desktop\workspace\data\shuaidao\train_images_20240625\train\images"

**推理时置信度注意调高一点，因为躺下的特征很明显，不怕被pass掉，反而应该防止误报，设为0.75。**



