from threading import Thread
import time

def task(sleeptimes,task_id):
    for i in range(10):
        time.sleep(sleeptimes)
        print("来自任务:{}".format(task_id))

if __name__ == '__main__':
    t1 = Thread(target=task,args=[2,1])
    t2 = Thread(target=task,args=[2,2])
    t1.start()
    t2.start()
    print("------------------------主线程结束-------------------")