import psutil
import time
import GPUtil
import py3nvml.py3nvml as pynv
import pynvml
import os
import re
def print_cpu_usage():
    cpu_usage = psutil.cpu_percent(interval=1)
    print(f"CPU Usage: {cpu_usage}%")

def get_memory_usage():
    memory_info = psutil.virtual_memory()
    total_memory = memory_info.total / (1024 ** 3)  # Convert bytes to GB
    used_memory = memory_info.used / (1024 ** 3)  # Convert bytes to GB
    percent_used = memory_info.percent
    return total_memory, used_memory, percent_used

def memory():
    total, used, percent = get_memory_usage()
    print(f"Total Memory: {total:.2f} GB")
    print(f"Used Memory: {used:.2f} GB")
    print(f"Memory Usage: {percent}%")
    #或者更简单的如下
    mem = psutil.virtual_memory()
    mem_usage = 100 * (1 - mem.available / mem.total)
    print("内存使用率:{}".format(mem_usage))


def get_disk_usage(path='/'):
    disk_usage = psutil.disk_usage(path)
    total_space = disk_usage.total / (1024 ** 3)  # Convert bytes to GB
    used_space = disk_usage.used / (1024 ** 3)  # Convert bytes to GB
    free_space = disk_usage.free / (1024 ** 3)  # Convert bytes to GB
    percent_used = disk_usage.percent
    return total_space, used_space, free_space, percent_used

def disk_usage():
    disk_partitions = psutil.disk_partitions(all=True)
    for partition in disk_partitions:
        total, used, free, percent = get_disk_usage(partition.mountpoint)
        print("挂载点名称:{}".format(partition.device))
        print(f"Total Disk Space: {total:.2f} GB")
        print(f"Used Disk Space: {used:.2f} GB")
        print(f"Free Disk Space: {free:.2f} GB")
        print(f"Disk Usage: {percent}%")




def get_gpu_power(id):
    pynv.nvmlInit()
    handle = pynv.nvmlDeviceGetHandleByIndex(id)  # Assuming you have only one GPU
    max_power = pynv.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000
    current_power = pynv.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert from mW to W
    pynv.nvmlShutdown()
    return max_power, current_power



def check_gpu_usage():
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id}:")
        print(gpu.name,gpu.load)
        print(f"  Utilization: {gpu.memoryUtil * 100}%")
        print(f"  Free Memory: {gpu.memoryFree} MB")
        print(f"  Used Memory: {gpu.memoryUsed} MB")
        print(f"  Temperature: {gpu.temperature} °C")
        max_power, current_power = get_gpu_power(gpu.id)
        print(f"  Max Power: {max_power}")
        print(f"  Current Power:{current_power}")




if __name__ == "__main__":
        print_cpu_usage()
        print("="*100)
        memory()
        print("="*100)
        disk_usage()
        print("="*100)
        check_gpu_usage()
        print("="*100)
        print("=" * 100)

