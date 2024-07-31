import pynvml

# 初始化NVIDIA管理库
pynvml.nvmlInit()

# 获取GPU的数量
gpu_count = pynvml.nvmlDeviceGetCount()

# 遍历每个GPU，打印信息
for i in range(gpu_count):
    # 获取GPU的handle
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    # 获取GPU的名称
    name = pynvml.nvmlDeviceGetName(handle)
    # 获取GPU的使用率和总内存
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    # 打印信息
    print(f"GPU {i}: {name}")
    print(f"  Free Memory: {mem_info.free // 1024 ** 2} MB")
    print(f"  Total Memory: {mem_info.total // 1024 ** 2} MB")

# 关闭NVIDIA管理库
pynvml.nvmlShutdown()