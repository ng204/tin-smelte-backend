from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

router = APIRouter(prefix='/crystallizer', tags=["结晶机温控"])

# 模拟存储温区配置
zone_configs: Dict[int, dict] = {
    1: {
        "temp_sensor": "LW1",
        "heater_sensor": "LJ1",
        "target_temp": 460.0,
        "temp_range": 5.0
    },
    2: {
        "temp_sensor": "LW2",
        "heater_sensor": "LJ2",
        "target_temp": 380.0,
        "temp_range": 5.0
    },
    3: {
        "temp_sensor": "LW3",
        "heater_sensor": "LJ3",
        "target_temp": 320.0,
        "temp_range": 5.0
    },
    4: {
        "temp_sensor": "LW4",
        "heater_sensor": "LJ4",
        "target_temp": 300.0,
        "temp_range": 5.0
    },
    5: {
        "temp_sensor": "LW5",
        "heater_sensor": "LJ5",
        "target_temp": 260.0,
        "temp_range": 5.0
    }
}

# 模拟实时数据
realtime_data: Dict[int, dict] = {
    1: {"current_temp": 458.5, "power": 75.2},
    2: {"current_temp": 378.3, "power": 68.5},
    3: {"current_temp": 321.2, "power": 62.3},
    4: {"current_temp": 299.8, "power": 55.7},
    5: {"current_temp": 261.5, "power": 48.9}
}

# 运行状态
system_status = {
    "running": False,
    "start_time": None,
    "logs": []
}

class ZoneConfig(BaseModel):
    """温区配置模型"""
    zone_id: int
    temp_sensor: str
    heater_sensor: str
    target_temp: float
    temp_range: float

class ZoneConfigUpdate(BaseModel):
    """批量更新配置"""
    zones: List[ZoneConfig]

class RealtimeData(BaseModel):
    """实时数据模型"""
    zone_id: int
    current_temp: float
    power: float

class SystemStatus(BaseModel):
    """系统状态"""
    running: bool
    start_time: Optional[str]
    logs: List[dict]

@router.get("/config")
async def get_config():
    """获取所有温区配置"""
    try:
        configs = [
            {
                "zone_id": zone_id,
                **config
            }
            for zone_id, config in zone_configs.items()
        ]
        return {
            "success": True,
            "data": configs
        }
    except Exception as e:
        logger.error(f"获取配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/config")
async def update_config(update: ZoneConfigUpdate):
    """更新温区配置"""
    try:
        for zone in update.zones:
            if zone.zone_id not in zone_configs:
                raise ValueError(f"无效的温区ID: {zone.zone_id}")
            
            zone_configs[zone.zone_id] = {
                "temp_sensor": zone.temp_sensor,
                "heater_sensor": zone.heater_sensor,
                "target_temp": zone.target_temp,
                "temp_range": zone.temp_range
            }
        
        # 记录日志
        system_status["logs"].append({
            "timestamp": datetime.now().isoformat(),
            "type": "config",
            "message": f"配置已更新，共 {len(update.zones)} 个温区"
        })
        
        logger.info(f"配置更新成功，共 {len(update.zones)} 个温区")
        
        return {
            "success": True,
            "message": "配置更新成功"
        }
    except Exception as e:
        logger.error(f"配置更新失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/realtime")
async def get_realtime_data():
    """获取实时温度和加热状态数据"""
    try:
        import random
        
        # 模拟温度波动
        data = []
        for zone_id, config in zone_configs.items():
            target = config["target_temp"]
            range_val = config["temp_range"]
            
            # 当前温度在目标温度±波动范围内随机变化
            current_temp = target + random.uniform(-range_val * 1.5, range_val * 1.5)
            
            # 计算加热状态
            # 如果温度高于（目标+范围），停止加热（断开）
            # 如果温度低于（目标-范围），继续加热（连通）
            max_temp = target + range_val
            min_temp = target - range_val
            
            if current_temp > max_temp:
                heating_status = "断开"
                status = "warning"  # 超温
            elif current_temp < min_temp:
                heating_status = "连通"
                status = "warning"  # 欠温
            else:
                # 在范围内，根据是否需要加热判断
                if current_temp < target:
                    heating_status = "连通"
                else:
                    heating_status = "断开"
                status = "normal"
            
            data.append({
                "zone_id": zone_id,
                "current_temp": round(current_temp, 1),
                "target_temp": target,
                "temp_range": range_val,
                "heating_status": heating_status,
                "status": status
            })
        
        return {
            "success": True,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"获取实时数据失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start")
async def start_control():
    """启动温控系统"""
    try:
        if system_status["running"]:
            return {
                "success": False,
                "message": "温控系统已在运行中"
            }
        
        system_status["running"] = True
        system_status["start_time"] = datetime.now().isoformat()
        system_status["logs"].append({
            "timestamp": datetime.now().isoformat(),
            "type": "system",
            "message": "温控系统已启动"
        })
        
        logger.info("温控系统启动")
        
        return {
            "success": True,
            "message": "温控系统启动成功"
        }
    except Exception as e:
        logger.error(f"启动失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop")
async def stop_control():
    """停止温控系统"""
    try:
        if not system_status["running"]:
            return {
                "success": False,
                "message": "温控系统未运行"
            }
        
        system_status["running"] = False
        system_status["logs"].append({
            "timestamp": datetime.now().isoformat(),
            "type": "system",
            "message": "温控系统已停止"
        })
        
        logger.info("温控系统停止")
        
        return {
            "success": True,
            "message": "温控系统已停止"
        }
    except Exception as e:
        logger.error(f"停止失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_status():
    """获取系统状态和运行日志"""
    try:
        # 只返回最近50条日志
        recent_logs = system_status["logs"][-50:] if system_status["logs"] else []
        
        return {
            "success": True,
            "data": {
                "running": system_status["running"],
                "start_time": system_status["start_time"],
                "logs": recent_logs
            }
        }
    except Exception as e:
        logger.error(f"获取状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sensors")
async def get_sensor_options():
    """获取传感器接线编号选项"""
    return {
        "success": True,
        "data": {
            "temp_sensors": ["LW1", "LW2", "LW3", "LW4", "LW5"],
            "heater_sensors": ["LJ1", "LJ2", "LJ3", "LJ4", "LJ5"]
        }
    }
