import numpy as np
def input_action(xyn=None):
    """
    :param xyn: [x1,y1,x2,y2...,x17,y17]
    :return:
    """
    action = ""
    while not action in ["stand","sit","fall","squat","bend","climb","straddle"]:
        action = input("输入动作名称:")
    print("标签是:{}".format(action))

a = np.array([[1,2],[3,4]])
for i in a:
    print(i)