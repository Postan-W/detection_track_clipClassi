# -*- coding: utf-8 -*-
# @Author: wmingzhu
# @Time: 2024/6/2
# @Software: PyCharm
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