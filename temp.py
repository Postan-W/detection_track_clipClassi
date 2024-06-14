import base64
from threading import Thread
from queue import Queue
import time
import cv2
def task(queue,i):
    while True:
        time.sleep(2)
        print("任务{}输出:{}".format(i,queue.get()))

if __name__ == '__main__':
    # queue = Queue()
    # t1 = Thread(target=task,args=(queue,1))
    # t2 = Thread(target=task,args=(queue,2))
    # t1.start()
    # t2.start()
    # for i in range(10):
    #     queue.put(i)
    from pathlib import Path
    import os
    d = str(Path(__file__).resolve().parent)
    print(d)
    import datetime
    print(str(datetime.datetime.now()))
    img = "./main_code/images/positive1_10.jpg"
    img = cv2.imread(img)
    ret,image = cv2.imencode(".jpg",img)
    encoded = base64.b64encode(image).decode("utf-8")
