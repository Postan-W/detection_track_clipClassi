# -*- coding: utf-8 -*-
# @Time    : 2022/7/14 9:07
# @Author  : Chuck
import cv2
import time
import torchvision
import torch
import numpy as np
import base64


def draw_poly_area(img, area_poly_list):
    """
    画多边形区域的框
    :param img: numpy类型图片数据
    :param area_poly: 区域像素坐标点数据
    :return:
    """
    for area_poly in area_poly_list:
        area_poly = np.array(area_poly, np.int32)
        cv2.polylines(img, [area_poly], isClosed=True, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)


def is_pt_in_poly_single_region(pt, poly):
    '''判断点是否在多边形内部的(pnpoly 算法)
    '''
    nvert = len(poly)
    vertx = []
    verty = []
    testx = pt[0]
    testy = pt[1]
    for item in poly:
        vertx.append(item[0])
        verty.append(item[1])

    j = nvert - 1
    res = False
    for i in range(nvert):
        if (verty[j] - verty[i]) == 0:
            j = i
            continue
        x = (vertx[j] - vertx[i]) * (testy - verty[i]) / (verty[j] - verty[i]) + vertx[i]
        if ((verty[i] > testy) != (verty[j] > testy)) and (testx < x):
            res = not res
        j = i
    return res

def is_poi_in_poly(pt, poly_list):
    """
    判断点是否在多边形内部的 pnpoly 算法，从一个目标点引出一条射线(任意一条射线)，统计这条射线与对变形的交点个数。如果有奇数个交点，则说明目标点在多边形内，
    若为偶数(0也算)个交点，则在外。
    本代码实现的算法是目标点向右引出的一条射线。
    :param pt: 点坐标 [x,y]
    :param poly: 点多边形坐标 [[[x1,y1],[x2,y2],...],[[x1,y1],[x2,y2],...]...]
    :return: 点是否在多边形之内
    """
    #注意，这里是多个多边形区域
    res_list = []
    for poly in poly_list:
        nvert = len(poly)
        vertx = []
        verty = []
        testx = pt[0]
        testy = pt[1]
        for item in poly:
            vertx.append(item[0])
            verty.append(item[1])
        j = nvert - 1#下面的for语句，第一次循环1选择最后一个点作为比较点
        res = False
        for i in range(nvert):#遍历n-1个点
            #第一次for循环使用最后一个点
            if (verty[j] - verty[i]) == 0:
                j = i
                continue
            #通过画图可以很容易明白，下面公式即得到从(testx,testy)向右引出的水平射线和两点连线的交点的x坐标值
            x = (vertx[j] - vertx[i]) * (testy - verty[i]) / (verty[j] - verty[i]) + vertx[i]
            if ((verty[i] > testy) != (verty[j] > testy)) and (testx < x):
                res = not res#奇数次为True，偶数次为False
            j = i#下次循环时形成下一条边
        res_list.append(res)
    return True in res_list#有一个区域中包含该点即视为区域入侵


def in_poly_area(xyxy, area_poly):
    """
    检测物体是否在多边形区域内
    :param xyxy: 物体框的坐标
    :param area_poly: 区域位置坐标
    :return: True -> 在区域内，False -> 不在区域内
    """
    if not area_poly:  # 为空
        return False
    # 求物体框的中点
    object_x1 = int(xyxy[0])
    object_y1 = int(xyxy[1])
    object_x2 = int(xyxy[2])
    object_y2 = int(xyxy[3])
    object_w = object_x2 - object_x1
    object_h = object_y2 - object_y1
    object_cx = object_x1 + (object_w / 2)
    object_cy = object_y1 + (object_h / 2)
    return is_poi_in_poly([object_cx, object_cy], area_poly)


def plot_one_box(x, img, color):
    # Plots one bounding box on image
    tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))#左上和右下坐标
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)


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


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
    """
    Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]#
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    #clamp和clamp_,前者是返回新tensor，后者是更新本tensor。因为预测框的坐标值可能会超出图像，限定其值(这里是缩放后的)在图片hw范围内
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        #在letterbox中取的是目标尺寸和原尺寸比小的那个r，可以知道这里的gain又重新得到了那个r。gain的两个元素值一个等于r,一个大于等于r(因为原图小边乘以r后经过padding，这里再除以原图该边，故>=r)
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        #下面的操作的结果，一个是0，一个是当时在letterbox中一侧的padding值
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    #只要减去top或者left的padding值，就得到了在缩放后的图像中的目标位置，然后再除以缩放率得到在原图中的位置值
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

#letterbox目的是将长和宽使用相同比率缩放以避免失真，"空出来"的部分用灰色填充。最终新的尺寸为正方形
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    '''
    :param im: 原图 hwc
    :param new_shape:缩放后的尺寸
    :param color: pad的颜色（灰色边框，补齐调整后的区域）
    :param auto:True 保证缩放后的图片保持原图的比例 即 将原图最长边缩放到指定大小，再将原图较短边按原图比例缩放，padding值经过取余操作（不会失真）
                  False 将原图最长边缩放到指定大小，再将原图较短边按原图比例缩放,最后将较短边两边pad操作缩放到最长边大小,也是新尺寸的大小（不会失真）
    :param scaleFill:True 简单粗暴的将原图resize到指定的大小，没有pad操作（失真）
    :param scaleup:True  这种情形下如果原图长宽都比新尺寸小，那么r就大于1，即scale up
                   False 对于大于new_shape的原图进行缩放,小于的不变
    :param stride: 模型下采样次数的2的次方，这个涉及感受野的问题，在YOLOV5中下采样次数为5，则stride为32
    :return:
    '''
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    #Scale ratio (new / old)
    # print("h的比率为:{}".format(new_shape[0] / shape[0]))
    # print("w的比率为:{}".format(new_shape[1] / shape[1]))
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r #width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]#wh要填充的量，当然了，放缩比小的那个为0
    # print("dh的值是:{},dw的值是:{}".format(dh,dw))

    if auto:  # minimum rectangle
        #对dw或dh取余是因为a % b = 0;((a -c) % b + c) % b)=((a%b - c%b)%b + c) % b = (c - c%b)%b=(c%b - c%b%b)%b=0,即取余后加上原图尺寸后仍然可以被stride整除
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  #wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]#width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        #注意resize的第二个参数是(w,h)
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    #round(x)的机制是四舍(包含5)五入。结合上面的除以2操作，可知下面的四个边的值的含义。top和left减去0.1和不减效果一样
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh),new_unpad
