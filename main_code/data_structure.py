from pandantic import BaseModel
from typing import *
import numpy as np

class Loiter(BaseModel):

    id: Optional[int] = None
    timestamp: Optional[int] = None  # 首次出现的时间，用来更新loitertime
    initial_box: Optional[list[float]] = None  # 存放此id对应的box坐标 左上角，右上角？
    box: Optional[list[float]] = None  # 存放此id对应的box坐标 左上角，右上角？
    speed: Optional[float] = 0.0
    maxspeed: Optional[float] = 0.0
    minspeed: Optional[float] = 0.0
    x_speed: Optional[float] = 0.0
    y_speed: Optional[float] = 0.0
    average_speed: Optional[float] = 0.0
    alarm: Optional[bool] = None
    flag: Optional[bool] = None # 用于判断此id之前是否出现，cache_loiters中是否存在
    initial_center: Optional[List[float]] = None  # 首次出现时，中心点坐标
    center: Optional[List[float]] = None  # 此id 的中心坐标
    loitertime: Optional[int] = None  # 此id存在的时间
    lasttimestamp: Optional[int] = None  # 记录最后出现的时间，用来控制列表的长度 最后出现的时间与当前时间相差50s就可以删除了，基本不会存在。



class Intruder(BaseModel):
    # 记录区域入侵检测信息： id，box，flag；是否已经推送过报警;是否已存在于pre_intruder，在（True）则不做处理，不在（False）则推送此次检测结果。
    id: Optional[int] = None
    box: Optional[list[float]] = None
    flag: Optional[bool] = None
    timestamp: Optional[Union[int, float]] = None

class Frame(BaseModel):
    """
    用来向队列中存图片帧信息
    boxes, points, data, timestamp, url, 和 skip 是 Frame 类的属性。
    boxes 和 points 是可选的，可以包含多个浮点数列表的列表。
    masks: model(image) 的结果
    areas: 经过置信度合类别判断后，的xyxyn坐标列表
    data :是一个必须的属性，它是一个 NumPy 数组（np.ndarray）。（图片信息）
    timestamp: 是帧的时间戳，是一个整数。
    url: 是帧对应的图像源的 URL，是一个字符串。
    skip: 是可选的布尔值，表示是否跳帧。
    in_count: 进入人数总数
    out_count: 外出人数总数
    """
    stops: bool = False
    boxes: Optional[List[List[float]]] = None
    points: Optional[List[List[float]]] = None
    masks: Optional[List[Any]] = None
    mask_flag: Optional[int] = None
    falling: Optional[List[List[float]]] = None
    # RangeDetection
    areas: Optional[List[List[float]]] =None
    roi: Optional[List[List[float]]] = None
    intruders: Optional[List[Intruder]] = []
    data: np.ndarray = None
    timestamp: int = None
    url: str = None
    skip: Optional[bool] = None
    # IONumber
    loiters: Optional[List[Loiter]] = []
    results: Optional[List[Tuple[Any, Any, Any, Any, Any, Any, int]]] = []
    in_count: int = 0
    out_count: int = 0
    inCount: dict[str, int] = {}  # str：线段id， int：通过此线段进入的人数
    outCount: dict[str, int] = {}  # str：线段id， int：通过此线段外出的人数

    class Config:
        arbitrary_types_allowed = True
    # Config 类是 Pydantic 提供的一个配置类，用于配置 BaseModel 的行为。
    # 在这里设置了 arbitrary_types_allowed = True，
    # 允许 Frame 类中的属性使用任意类型，而不仅仅是 Pydantic 支持的基本类型
