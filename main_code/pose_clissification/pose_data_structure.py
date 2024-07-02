from pydantic import BaseModel

"""
人在各种动作中头部相对于其他身体部位的空间位置也是具有明显特征的，比如横躺在画面中，头部的关键点的h坐标和身体的各部位基本
一致而w坐标明显不同；坐着时头部的w坐标和上半身的各关键点w坐标比较接近，而h坐标明显不同。
这里我选取有代表性的LEFT_EAR、RIGHT_EAR作为头部的关键点，NOSE、LEFT_EYE、RIGHT_EYE被舍弃了，因为左右耳无论在
各种角度一般总会有一个被看见，由此能代表头部的空间位置，而鼻子和左右眼则在背面时被完全遮挡，在一些非正面角度时也常被大幅度
遮挡，这会给模型训练以及推理的泛化能力带来阻碍。比如一个人其他身体部位关键点数据都具备翻越闸机的特点，而头部的左右眼及鼻子
都没检测到，这样会不会使分类模型错误判断为非翻越，这是很可能发生的，因为训练数据中头部的关键点一般都是具备的。
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

action_list = ["stand","sit","fall","squat","bend","climb","straddle"]


