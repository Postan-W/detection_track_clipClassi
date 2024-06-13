#进程结束，对应的线程必然结束，但是子进程不一定结束(包括进程里线程创建的进程，如果不是守护进程的话)
from subprocess import Popen
import time
import os
import signal
p = Popen(["python","报警逻辑试验.py"])

time.sleep(10)

p.terminate()
while True:
    pass