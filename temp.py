f = open("base64.txt","r",encoding="utf-8")
data = f.read()

import base64
from PIL import Image
from io import BytesIO

def base64_to_image(data):
    # 将 base64 字符串解码为字节数据
    image_data = base64.b64decode(data)

    # 创建内存缓冲区对象
    buffer = BytesIO(image_data)

    # 打开图像
    image = Image.open(buffer)

    # 可选：如果需要保存图像
    # image.save("image.jpg")

    return image
# base64_to_image(data).show()
a = [False, False, False, False]
print(sum(a))