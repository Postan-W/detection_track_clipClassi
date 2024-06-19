import cv2
import numpy as np
import base64
def np_to_str(image_np):
    """
    将numpy格式图片转换为流文件
    :param image_np:numpy数组
    :return:
    """
    retval, buffer = cv2.imencode('.jpg', image_np)
    pic_str = base64.b64encode(buffer)
    pic_str = pic_str.decode()
    return pic_str

def str_to_np(img_str):
    """
    将图片流转换为numpy数组
    :param img_str: base64编码图片
    :return: np.array图片
    """
    img_decode = base64.b64decode(img_str)  # base64解码
    img_np = np.frombuffer(img_decode, np.uint8)  # 从str数据读取为np.array形式
    img = cv2.imdecode(img_np, cv2.COLOR_RGB2BGR)  # 转为OpenCV形式

    return img

def cv2_display_single_image(image, width=900, height=600):
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", width, height)
    cv2.moveWindow("Image", 200,200)#窗口左上角处于屏幕(200,200)处
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cv2_plot_boxes(boxes, img, color:tuple=(0,0,255)):
    """
    Args:
        boxes: [[x1,y1,x2,y2],],矩形框的左上角和右下角
        img: numpy.ndarray,维度顺序是hwc，通道是bgr
        color:  bgr数值

    Returns:

    """
    # Plots one bounding box on image(numpy.ndarray), color order is in bgr and dim order is in hwc
    for x in boxes:
        tl = round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))  # 左上和右下坐标
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

def plot_boxes_with_text(boxes, img, color=[100,100,100], text_info="None",velocity=None, thickness=1, fontsize=0.5, fontthickness=1):
    # Plots bounding boxes on image img
   for x in boxes:
       c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
       cv2.rectangle(img, c1, c2, color, thickness, lineType=cv2.LINE_AA)
       t_size = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize, fontthickness + 2)[0]
       cv2.rectangle(img, c1, (c1[0] + int(t_size[0]), c1[1] + int(t_size[1] * 1.45)), color, -1)
       cv2.putText(img, text_info, (c1[0], c1[1] + t_size[1] + 2),cv2.FONT_HERSHEY_TRIPLEX, fontsize,color=[255, 255, 255], thickness=fontthickness)
   return img

def plot_boxes_with_text_for_yolotrack(boxes, img, color=[102, 170, 238],class_name="",velocity=None, thickness=2, fontsize=0.5, fontthickness=1):
    # Plots bounding boxes on image img
   for x in boxes:
       c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
       cv2.rectangle(img, c1, c2, color, thickness, lineType=cv2.LINE_AA)
       text_info = "id:"+ str(x[4]) + " " + class_name + " " + str(round(x[5],2)) if len(x) == 7 else class_name + " " + str(round(x[4],2))
       t_size = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize, fontthickness + 2)[0]
       cv2.rectangle(img, c1, (c1[0] + int(t_size[0]), c1[1] + int(t_size[1] * 1.45)), color, -1)#文字背景
       cv2.putText(img, text_info, (c1[0], c1[1] + t_size[1] + 2),cv2.FONT_HERSHEY_TRIPLEX, fontsize,color=[255,255,255],thickness=fontthickness)
   return img

def cv2_draw_poly_area(img, area_poly_list,color=(0,0, 255)):
    """
    画多边形区域的框
    :param img: numpy.ndarray,维度顺序是hwc，通道是bgr
    :param area_poly: 多个区域，数值类型是np.int32，比如两个区域[[[100,100],[200,200],[300,100]],[[500,500],[600,600],[700,500]]]
    :return:
    """
    area_poly_list = np.array(area_poly_list,np.int32)#这里只是为了把数值类型转为np.int32，并不要求类型为numpy.ndarray,所以list(area_poly_list)也行
    cv2.polylines(img, area_poly_list, isClosed=True, color=color, thickness=3, lineType=cv2.LINE_AA)

