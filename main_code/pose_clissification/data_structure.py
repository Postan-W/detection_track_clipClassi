from pydantic import BaseModel

"""
前5个关键点是位于头上的，它们相对于其他关键点的空间位置也能反映出行为动作，比如：翻越时膝盖和头的距离就比较近、
摔倒时头部的h坐标就和身体其他部位的关键点的h坐标比较接近
"""
class KeyPoints(BaseModel):
    NOSE:           int = 0
    LEFT_EYE:       int = 1
    RIGHT_EYE:      int = 2
    LEFT_EAR:       int = 3
    RIGHT_EAR:      int = 4
    LEFT_SHOULDER:  int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW:     int = 7
    RIGHT_ELBOW:    int = 8
    LEFT_WRIST:     int = 9
    RIGHT_WRIST:    int = 10
    LEFT_HIP:       int = 11
    RIGHT_HIP:      int = 12
    LEFT_KNEE:      int = 13
    RIGHT_KNEE:     int = 14
    LEFT_ANKLE:     int = 15
    RIGHT_ANKLE:    int = 16

#这里的动作定义既包含了摔倒的(FALL)也包含了翻越的(CLIMB)
class Action(BaseModel):
    STAND: int = 0
    SIT: int = 1
    FALL: int = 2
    SQUAT: int = 3
    BEND: int = 4
    CLIMB: int = 5
    STRADDLE: int = 6 #跨这个动作暂时不作为翻越的行为，以免人行走时跨步产生误报



