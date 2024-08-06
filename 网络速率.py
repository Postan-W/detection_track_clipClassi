import psutil
import time


def get_network_usage(interface):
    """
    获取指定网络接口的传输速度（字节/秒）
    """
    old_counters = psutil.net_io_counters(pernic=True)[interface]
    time.sleep(0.2)  # 等待
    new_counters = psutil.net_io_counters(pernic=True)[interface]
    bytes_sent = new_counters.bytes_sent - old_counters.bytes_sent
    bytes_recv = new_counters.bytes_recv - old_counters.bytes_recv

    return bytes_sent, bytes_recv


def calculate_network_usage_percentage(interface):
    """
    计算指定网络接口的占有率
    """
    total_usage = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv#这里只是所有网卡的进出流量总和，并不是某个网卡的最大带宽
    print(psutil.net_io_counters(pernic=True))
    print(psutil.net_io_counters().bytes_sent,psutil.net_io_counters().bytes_recv)
    interface_usage = psutil.net_io_counters(pernic=True)[interface]
    usage = interface_usage.bytes_sent + interface_usage.bytes_recv
    return (usage / total_usage) * 100


# 获取所有网络接口的名称
network_interfaces = psutil.net_io_counters(pernic=True).keys()

# 打印每个网络接口的传输速度
for interface in network_interfaces:
    bytes_sent, bytes_recv = get_network_usage(interface)
    print(f"Interface: {interface}")
    print(f"  Sent: {round(bytes_sent/(1024*1024),2)*5} MB/s")
    print(f"  Recv: {round(bytes_recv/(1024*1024),2)*5} MB/s")
