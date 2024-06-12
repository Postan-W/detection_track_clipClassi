import time

def task2():
    for i in range(1,20):
        time.sleep(i)
        print("子进程的i值是：{}".format(i))
    print("线程中的子进程结束")

task2()
