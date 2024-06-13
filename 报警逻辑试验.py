import time#真实的时间应该使用fps计算来的时间
import random
d = {}
l = [1,22,33,123,45,56,1,22,1,12,33,99,89,99,22,56,678]

for i in range(100):
    time.sleep(1)
    id = l[random.randint(0,len(l)-1)]
    if id in d.keys():
        if int(time.time() - d[id]) < 2:
            print("一个动作之内，不需要重复报警")

        else:
            d[id] = time.time()
            print("出现过的，但是间隔时间较长，认为新的行为发生，所以也报警")
    else:
        d[id] = time.time()
        print("新发现，报警")
