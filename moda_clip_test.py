# require modelscope>=0.3.7，目前默认已经超过，您检查一下即可
# 按照更新镜像的方法处理或者下面的方法
# pip install --upgrade modelscope -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
# 需要单独安装decord，安装方法：pip install decord
import torch
import os
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
from modelscope.preprocessors.image import load_image

print(torch.__version__,torch.cuda.is_available())
pipeline = pipeline(task=Tasks.multi_modal_embedding,
    model='damo/multi-modal_clip-vit-large-patch14_336_zh', model_revision='v1.0.1')
# input_img = load_image("./images/pj399.jpg") # 支持皮卡丘示例图片路径/本地图片 返回PIL.Image
images = [os.path.join("./images",i) for i in os.listdir("./images")]
input_img = []
for i in images:
    input_img.append(load_image(i))

input_texts = []

# 支持一张图片(PIL.Image)或多张图片(List[PIL.Image])输入，输出归一化特征向量
img_embedding = pipeline.forward({'img': input_img})['img_embedding'] # 2D Tensor, [图片数, 特征维度]

# 支持一条文本(str)或多条文本(List[str])输入，输出归一化特征向量
text_embedding = pipeline.forward({'text': input_texts})['text_embedding'] # 2D Tensor, [文本数, 特征维度]

# 计算图文相似度
with torch.no_grad():
    # 计算内积得到logit，考虑模型temperature
    logits_per_image = (img_embedding / pipeline.model.temperature) @ text_embedding.t()
    # 根据logit计算概率分布
    probs = logits_per_image.softmax(dim=-1)
    print(probs)
    result = torch.argmax(probs,dim=1)

print("图文匹配结果:", result)