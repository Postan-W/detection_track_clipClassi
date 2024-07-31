import numpy as np
import cv2

def is_poi_in_poly(pt, poly):
    """
    判断点是否在多边形内部的 pnpoly 算法
    :param pt: 点坐标 [x,y]
    :param poly: 点多边形坐标 [[x1,y1],[x2,y2],...] 点要按顺序给出，这样才方便得到各个边，即一个点和下一个点形成
    :return: 点是否在多边形之内
    """
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
def in_poly_area_dangerous(xyxy,area_poly):
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

area = [[250,138],[1925,165],[2023,1027],[836,1207],[336,619]]
p1 = [1275,683]
p2 = [166,1350]
p3 = [2013,1375]
p4 = [1506,904]
print(is_poi_in_poly(p4,area))
if not area:
    print("哈哈")