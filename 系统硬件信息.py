#https://zhuanlan.zhihu.com/p/691700976
import psutil
# 获取CPU信息
cpus = psutil.cpu_times(percpu=True)
print(cpus)

# 获取内存信息
memory = psutil.virtual_memory()
print(memory)
#cpu使用率
print(psutil.cpu_percent())
# 获取硬盘信息
disks = psutil.disk_partitions()
print(disks)

# 获取磁盘使用情况
disk_usage = psutil.disk_usage('/')
print(disk_usage)

# 获取网络信息
net_io = psutil.net_io_counters()
print(net_io)
class DeviceInfo:
    @classmethod
    def get_cpu_info(cls):
        return psutil.cpu_percent()

print(DeviceInfo.get_cpu_info())