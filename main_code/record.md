# 1.目标检测

## 1.1 训练

### 1.1.1 2024/6/6

- **数据**

  设置了两类normal_pass和unnormal_pass。使用了472张图片训练，157张图片验证，每张图片有1-2个目标，总体目标normal和unnormal的比值大概是6:4，接近7:3，所以说unnormal的比重较小，下次标注训练时要平衡下。

- **模型**

  使用的是ultralytics github的基于COCO8 0 pre-trained classes的YOLOv8m.pt。得到模型大小为50M。

- **超参数**

  batch=32;epoch=15

- **验证指标**

  ![1](record.assets/1.png)

  看样子相当地好。

- **测试**

  

