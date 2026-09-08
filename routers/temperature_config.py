"""
温度参数配置管理
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import hashlib

router = APIRouter(prefix="/api/temperature-config", tags=["温度参数配置"])

# 配置访问密码（SHA-256加密存储）
# 原始密码: Kust01150312
CONFIG_PASSWORD_HASH = "3e9124ffc57682a39f2ab82f0c446324fee76163bae82007d8620f42ae711b28"  # "Kust01150312" 的 SHA-256

# 默认配置
DEFAULT_CONFIG = {
    "furnace": {
        "base_temp": 1100,
        "fluctuation": 15
    },
    "crystallizer": {
        "zones": [
            {"id": "zone1", "name": "区域一", "base_temp": 420, "fluctuation": 10},
            {"id": "zone2", "name": "区域二", "base_temp": 350, "fluctuation": 8},
            {"id": "zone3", "name": "区域三", "base_temp": 310, "fluctuation": 6},
            {"id": "zone4", "name": "区域四", "base_temp": 260, "fluctuation": 5},
            {"id": "zone5", "name": "区域五", "base_temp": 240, "fluctuation": 3},
        ]
    }
}

# 内存存储配置（实际应用中应该存储到数据库）
current_config = DEFAULT_CONFIG.copy()


class PasswordVerifyRequest(BaseModel):
    """密码验证请求"""
    password: str


class FurnaceConfig(BaseModel):
    """顶吹炉配置"""
    base_temp: float
    fluctuation: float


class ZoneConfig(BaseModel):
    """结晶机区域配置"""
    id: str
    name: str
    base_temp: float
    fluctuation: float


class CrystallizerConfig(BaseModel):
    """结晶机配置"""
    zones: List[ZoneConfig]


class TemperatureConfig(BaseModel):
    """温度配置"""
    furnace: FurnaceConfig
    crystallizer: CrystallizerConfig


def verify_password(password: str) -> bool:
    """验证密码"""
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    return password_hash == CONFIG_PASSWORD_HASH


@router.post("/verify-password")
async def verify_config_password(request: PasswordVerifyRequest):
    """验证参数配置访问密码"""
    is_valid = verify_password(request.password)
    
    if is_valid:
        print(f"\n[温度配置] 密码验证成功")
        return {
            "success": True,
            "message": "密码验证成功"
        }
    else:
        print(f"\n[温度配置] 密码验证失败")
        return {
            "success": False,
            "message": "密码错误"
        }


@router.get("/")
async def get_config():
    """获取当前温度配置"""
    return {
        "success": True,
        "data": current_config
    }


@router.post("/")
async def update_config(config: TemperatureConfig):
    """更新温度配置"""
    global current_config
    
    current_config = {
        "furnace": {
            "base_temp": config.furnace.base_temp,
            "fluctuation": config.furnace.fluctuation
        },
        "crystallizer": {
            "zones": [
                {
                    "id": zone.id,
                    "name": zone.name,
                    "base_temp": zone.base_temp,
                    "fluctuation": zone.fluctuation
                }
                for zone in config.crystallizer.zones
            ]
        }
    }
    
    print(f"\n[温度配置] 配置已更新")
    print(f"  顶吹炉: {current_config['furnace']['base_temp']}°C ± {current_config['furnace']['fluctuation']}°C")
    for zone in current_config['crystallizer']['zones']:
        print(f"  {zone['name']}: {zone['base_temp']}°C ± {zone['fluctuation']}°C")
    
    return {
        "success": True,
        "message": "配置更新成功",
        "data": current_config
    }


@router.post("/reset")
async def reset_config():
    """重置为默认配置"""
    global current_config
    current_config = DEFAULT_CONFIG.copy()
    
    print(f"\n[温度配置] 配置已重置为默认值")
    
    return {
        "success": True,
        "message": "配置已重置为默认值",
        "data": current_config
    }
