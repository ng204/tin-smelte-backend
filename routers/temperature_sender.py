"""
温度数据发送服务
支持通过UDP向指定端口发送温度数据
"""
import asyncio
import json
import socket
from datetime import datetime
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# 导入配置
from .temperature_config import current_config

router = APIRouter(prefix="/api/temperature-sender", tags=["温度数据发送"])

# 全局发送任务管理
active_senders: Dict[str, Dict] = {}


class SendConfig(BaseModel):
    """发送配置"""
    device_type: str  # 'furnace' 或 'crystallizer'
    local_port: int = 5000  # 本地发送端口
    interval: float = 1.5  # 发送间隔（秒）


class TemperatureData(BaseModel):
    """温度数据"""
    device_type: str
    timestamp: str
    data: dict


# 模拟温度数据生成（使用配置参数）
def generate_furnace_temp() -> float:
    """生成顶吹炉温度数据"""
    import random
    config = current_config['furnace']
    base_temp = config['base_temp']
    fluctuation = config['fluctuation']
    # 标准差为fluctuation/3，使95%的数据在±fluctuation范围内
    return round(random.gauss(base_temp, fluctuation / 3), 1)


def generate_crystallizer_temps() -> dict:
    """生成结晶机多区域温度数据"""
    import random
    zones_config = current_config['crystallizer']['zones']
    
    result = {}
    for zone_config in zones_config:
        zone_id = zone_config['id']
        name = zone_config['name']
        base_temp = zone_config['base_temp']
        fluctuation = zone_config['fluctuation']
        
        # 标准差为fluctuation/3
        temperature = round(random.gauss(base_temp, fluctuation / 3), 1)
        
        result[zone_id] = {
            "name": name,
            "temperature": temperature
        }
    
    return result


async def send_temperature_task(device_type: str, local_port: int, interval: float):
    """温度数据发送任务 - UDP广播模式"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 设置为广播模式
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    # 绑定到本地端口（可选，用于标识数据源）
    try:
        sock.bind(('0.0.0.0', 0))  # 绑定任意可用端口
    except:
        pass
    
    # 广播地址
    broadcast_addr = ('255.255.255.255', local_port)
    
    print(f"\n{'='*60}")
    print(f"[温度发送服务] 启动发送任务")
    print(f"  设备类型: {device_type}")
    print(f"  本地端口: {local_port} (UDP广播)")
    print(f"  发送间隔: {interval}秒")
    print(f"{'='*60}\n")
    
    try:
        while True:
            # 检查任务是否应该停止
            if device_type not in active_senders:
                print(f"[温度发送服务] {device_type} - 任务已停止")
                break
            
            # 生成温度数据
            if device_type == "furnace":
                temp_data = {
                    "device_type": "furnace",
                    "temperature": generate_furnace_temp(),
                    "unit": "°C"
                }
            else:  # crystallizer
                temp_data = {
                    "device_type": "crystallizer",
                    "zones": generate_crystallizer_temps(),
                    "unit": "°C"
                }
            
            # 添加时间戳
            temp_data["timestamp"] = datetime.now().isoformat()
            
            # 广播数据
            message = json.dumps(temp_data, ensure_ascii=False).encode('utf-8')
            sock.sendto(message, broadcast_addr)
            
            # 更新发送状态
            if device_type in active_senders:
                active_senders[device_type]["sent_count"] += 1
                active_senders[device_type]["last_data"] = temp_data
                
            
            await asyncio.sleep(interval)
    
    except Exception as e:
        print(f"[错误] {device_type} 发送任务异常: {e}")
    finally:
        sock.close()
        if device_type in active_senders:
            del active_senders[device_type]
        print(f"[温度发送服务] {device_type} - 发送任务已结束\n")


@router.post("/start")
async def start_sending(config: SendConfig):
    """开始发送温度数据"""
    device_type = config.device_type
    
    # 检查是否已在发送
    if device_type in active_senders:
        print(f"[警告] {device_type} 已在发送数据中")
        return {
            "success": False,
            "message": "该设备已在发送数据中",
            "status": active_senders[device_type]
        }
    
    # 创建发送任务
    active_senders[device_type] = {
        "local_port": config.local_port,
        "interval": config.interval,
        "sent_count": 0,
        "last_data": None,
        "start_time": datetime.now().isoformat()
    }
    
    print(f"\n[温度发送服务] 接收到启动请求")
    print(f"  设备: {device_type}")
    print(f"  本地端口: {config.local_port} (UDP广播)")
    print(f"  间隔: {config.interval}秒\n")
    
    # 启动异步发送任务
    asyncio.create_task(
        send_temperature_task(device_type, config.local_port, config.interval)
    )
    
    return {
        "success": True,
        "message": f"开始从端口 {config.local_port} 广播温度数据",
        "config": {
            "device_type": device_type,
            "local_port": config.local_port,
            "interval": config.interval
        }
    }


@router.post("/stop/{device_type}")
async def stop_sending(device_type: str):
    """停止发送温度数据"""
    if device_type not in active_senders:
        print(f"[警告] {device_type} 未在发送数据")
        return {
            "success": False,
            "message": "该设备未在发送数据"
        }
    
    sender_info = active_senders[device_type].copy()
    del active_senders[device_type]
    
    print(f"\n[温度发送服务] 停止发送")
    print(f"  设备: {device_type}")
    print(f"  总发送次数: {sender_info['sent_count']}")
    print(f"  启动时间: {sender_info.get('start_time')}\n")
    
    return {
        "success": True,
        "message": "已停止发送温度数据",
        "statistics": {
            "sent_count": sender_info["sent_count"],
            "duration": sender_info.get("start_time")
        }
    }


@router.get("/status/{device_type}")
async def get_status(device_type: str):
    """获取发送状态"""
    if device_type not in active_senders:
        return {
            "is_sending": False,
            "device_type": device_type
        }
    
    sender_info = active_senders[device_type]
    return {
        "is_sending": True,
        "device_type": device_type,
        "local_port": sender_info['local_port'],
        "interval": sender_info["interval"],
        "sent_count": sender_info["sent_count"],
        "start_time": sender_info["start_time"],
        "last_data": sender_info["last_data"]
    }


@router.get("/status")
async def get_all_status():
    """获取所有设备的发送状态"""
    return {
        "active_senders": list(active_senders.keys()),
        "details": {
            device_type: {
                "local_port": info['local_port'],
                "sent_count": info["sent_count"],
                "start_time": info["start_time"]
            }
            for device_type, info in active_senders.items()
        }
    }
