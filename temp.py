import psutil
import time


def get_network_stats():
    # 获取所有网络接口的信息
    net_stats = psutil.net_io_counters(pernic=True)

    # 打印每个网络接口的信息
    for interface, stats in net_stats.items():
        print(f"Interface: {interface}")
        print(f"    Bytes sent:   {stats.bytes_sent}")
        print(f"    Bytes received: {stats.bytes_recv}")
        print(f"    Packets sent:   {stats.packets_sent}")
        print(f"    Packets received: {stats.packets_recv}")


# 获取初始网络状态
get_network_stats()

# 等待一段时间来接收数据
time.sleep(5)

# 再次获取网络状态，用于计算带宽
get_network_stats()