from threading import Thread
from subprocess import Popen
import time
def task():
    p = Popen(["python","temp2.py"])
    time.sleep(20)
    print("线程结束")

t = Thread(target=task)
t.start()

