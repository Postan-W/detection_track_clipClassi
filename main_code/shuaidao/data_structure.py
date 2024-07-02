from pydantic import BaseModel
from typing import *
import numpy as np


class Frame(BaseModel):
    """
    用来向队列中存图片帧信息
    boxes, points, train_data, timestamp, url, 和 skip 是 Frame 类的属性。
    boxes 和 points 是可选的，可以包含多个浮点数列表的列表。
    masks: model(image) 的结果
    areas: 经过置信度合类别判断后，的xyxyn坐标列表
    train_data :是一个必须的属性，它是一个 NumPy 数组（np.ndarray）。（图片信息）
    timestamp: 是帧的时间戳，是一个整数。
    url: 是帧对应的图像源的 URL，是一个字符串。
    skip: 是可选的布尔值，表示是否跳帧。
    in_count: 进入人数总数
    out_count: 外出人数总数
    """
    alarm: List[bool] = []
    stops: bool = False
    boxes: Optional[List[List[float]]] = None
    points: Optional[List[List[float]]] = None
    masks: Optional[List[Any]] = None
    mask_flag: Optional[int] = None
    falling: Optional[List[List[float]]] = None
    # RangeDetection
    areas: Optional[List[List[float]]] =None
    roi: Optional[List[List[float]]] = None

    data: np.ndarray = None
    timestamp: int = None
    url: str = None
    skip: Optional[bool] = None
    # IONumber

    results: Optional[List[Tuple[Any, Any, Any, Any, Any, Any, int]]] = []
    in_count: int = 0
    out_count: int = 0


    class Config:
        arbitrary_types_allowed = True
    # Config 类是 Pydantic 提供的一个配置类，用于配置 BaseModel 的行为。
    # 在这里设置了 arbitrary_types_allowed = True，
    # 允许 Frame 类中的属性使用任意类型，而不仅仅是 Pydantic 支持的基本类型
