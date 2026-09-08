from fastapi import APIRouter, HTTPException, status, WebSocket, WebSocketDisconnect, Depends, Query
from pydantic import BaseModel, Field, validator
import json
import logging
import asyncio
from datetime import datetime, timedelta
import os
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from enum import Enum

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix='/real-time-monitoring', tags=["实时数据监测"])

# 是否启用模拟数据（默认关闭，使用真实遥测数据）。设置环境变量 SIMULATION_MODE=true 可开启
SIMULATION_MODE = os.getenv("SIMULATION_MODE", "false").lower() == "true"

# 数据源类型枚举
class DataSourceType(str, Enum):
    FACTORY = "factory"
    SIMULATION = "simulation"
    HISTORICAL = "historical"
    EXTERNAL = "external"

# 数据类型枚举
class DataType(str, Enum):
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    FLOW = "flow"
    GAS = "gas"
    OXYGEN = "oxygen"
    COOLING = "cooling"
    CURRENT = "current"
    VOLTAGE = "voltage"
    POWER = "power"
    LEVEL = "level"
    PH = "ph"
    CONDUCTIVITY = "conductivity"
    TURBIDITY = "turbidity"
    DISSOLVED_OXYGEN = "dissolved_oxygen"

# 数据质量状态枚举
class DataQuality(str, Enum):
    GOOD = "good"
    BAD = "bad"
    UNCERTAIN = "uncertain"
    NO_DATA = "no_data"

# 实时数据存储 - 扩展为更完整的数据结构
real_time_data = {
    # 基础工艺参数
    'crystallizer_temperature': {'value': 0.0, 'unit': '°C', 'quality': DataQuality.NO_DATA, 'timestamp': datetime.now()},
    'furnace_temperature': {'value': 0.0, 'unit': '°C', 'quality': DataQuality.NO_DATA, 'timestamp': datetime.now()},
    'pressure': {'value': 98.7, 'unit': 'kPa', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    'flow': {'value': 496.8, 'unit': 'm³/h', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    'gas': {'value': 4327, 'unit': 'm³/h', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    'oxygen': {'value': 78.9, 'unit': '%', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    'cooling': {'value': 87.2, 'unit': '°C', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    
    # 设备状态参数
    'Offgas CO Analyzer': {'value': 4327, 'unit': 'ppm', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    'Roof & Ports CW Return': {'value': 31.3, 'unit': '°C', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    'Transition CW Return': {'value': 33.6, 'unit': '°C', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    'Pb T/Blk2 Iner CW Ret P2': {'value': 31.7, 'unit': '°C', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    'Bin 6 PV Feedrate': {'value': 3, 'unit': 't/h', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    'CW Flw Transition Cool': {'value': 98.7, 'unit': 'm³/h', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    'CW Flw Roof & Ports': {'value': 65.1, 'unit': 'm³/h', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    'Total Cooling Water Flow': {'value': 496.7, 'unit': 'm³/h', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    'Oxygen Purity PV': {'value': 78.9, 'unit': '%', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    'CW Flw Shell Cone': {'value': 87.2, 'unit': 'm³/h', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    'Process Mode': {'value': 4, 'unit': '', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    'Lance Air Flow': {'value': 14776, 'unit': 'm³/h', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    'Fuel Coal SV': {'value': 6200, 'unit': 'kg/h', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    'Coal Carrier Air Flow': {'value': 495, 'unit': 'm³/h', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    'Tip Flow Demand(air eqv)': {'value': 38738, 'unit': 'm³/h', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    'Tip Flow Supply(air eqv)': {'value': 36465, 'unit': 'm³/h', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    'Total Concentrate': {'value': 576704.2, 'unit': 'kg', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    'L=150mm H=3500mm': {'value': 250.1, 'unit': 'mm', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    
    # 新增参数
    'current': {'value': 15.2, 'unit': 'A', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    'voltage': {'value': 240.0, 'unit': 'V', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    'power': {'value': 8.5, 'unit': 'kW', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    'level': {'value': 75.3, 'unit': '%', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    'ph': {'value': 7.2, 'unit': '', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    'conductivity': {'value': 1250, 'unit': 'μS/cm', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    'turbidity': {'value': 2.1, 'unit': 'NTU', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()},
    'dissolved_oxygen': {'value': 8.5, 'unit': 'mg/L', 'quality': DataQuality.GOOD, 'timestamp': datetime.now()}
}

# 数据历史记录 - 扩展为更完整的历史数据结构
data_history = {}
for key in real_time_data:
    data_history[key] = []

# 数据质量配置
data_quality_config = {
    'crystallizer_temperature': {'min': -50, 'max': 1200, 'expected_range': (0, 1000)},
    'furnace_temperature': {'min': -50, 'max': 1600, 'expected_range': (600, 1400)},
    'pressure': {'min': 0, 'max': 1000, 'expected_range': (50, 150)},
    'flow': {'min': 0, 'max': 10000, 'expected_range': (100, 1000)},
    'gas': {'min': 0, 'max': 50000, 'expected_range': (1000, 10000)},
    'oxygen': {'min': 0, 'max': 100, 'expected_range': (70, 95)},
    'cooling': {'min': 0, 'max': 100, 'expected_range': (20, 80)},
    'current': {'min': 0, 'max': 1000, 'expected_range': (5, 50)},
    'voltage': {'min': 0, 'max': 1000, 'expected_range': (200, 400)},
    'power': {'min': 0, 'max': 1000, 'expected_range': (1, 100)},
    'level': {'min': 0, 'max': 100, 'expected_range': (20, 90)},
    'ph': {'min': 0, 'max': 14, 'expected_range': (6, 9)},
    'conductivity': {'min': 0, 'max': 10000, 'expected_range': (100, 2000)},
    'turbidity': {'min': 0, 'max': 100, 'expected_range': (0, 10)},
    'dissolved_oxygen': {'min': 0, 'max': 20, 'expected_range': (5, 15)}
}

# WebSocket连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[str, List[WebSocket]] = {}  # 按数据类型订阅

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket连接已建立，当前连接数: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        # 清理订阅
        for data_type, connections in self.subscriptions.items():
            if websocket in connections:
                connections.remove(websocket)
        logger.info(f"WebSocket连接已断开，当前连接数: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"发送消息失败: {str(e)}")
                self.active_connections.remove(connection)

    async def broadcast_to_subscribers(self, data_type: str, message: str):
        """向特定数据类型的订阅者广播消息"""
        if data_type in self.subscriptions:
            for connection in self.subscriptions[data_type]:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"向订阅者发送消息失败: {str(e)}")
                    self.subscriptions[data_type].remove(connection)

    def subscribe(self, websocket: WebSocket, data_type: str):
        """订阅特定数据类型"""
        if data_type not in self.subscriptions:
            self.subscriptions[data_type] = []
        if websocket not in self.subscriptions[data_type]:
            self.subscriptions[data_type].append(websocket)

    def unsubscribe(self, websocket: WebSocket, data_type: str):
        """取消订阅特定数据类型"""
        if data_type in self.subscriptions and websocket in self.subscriptions[data_type]:
            self.subscriptions[data_type].remove(websocket)

manager = ConnectionManager()

# 数据模型 - 扩展为更完整的数据结构
class RealTimeDataPoint(BaseModel):
    value: float = Field(..., description="数据值")
    unit: str = Field(..., description="数据单位")
    quality: DataQuality = Field(DataQuality.GOOD, description="数据质量")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    source: DataSourceType = Field(DataSourceType.SIMULATION, description="数据源")
    device_id: Optional[str] = Field(None, description="设备ID")
    location: Optional[str] = Field(None, description="位置信息")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")

    @validator('value')
    def validate_value(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError('数据值必须是数字')
        return v

class RealTimeDataRequest(BaseModel):
    data_type: str = Field(..., description="数据类型")
    value: float = Field(..., description="数据值")
    unit: Optional[str] = Field(None, description="数据单位")
    quality: Optional[DataQuality] = Field(DataQuality.GOOD, description="数据质量")
    timestamp: Optional[datetime] = Field(None, description="时间戳")
    source: DataSourceType = Field(DataSourceType.FACTORY, description="数据源")
    device_id: Optional[str] = Field(None, description="设备ID")
    location: Optional[str] = Field(None, description="位置信息")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")

class RealTimeDataResponse(BaseModel):
    success: bool = Field(..., description="操作是否成功")
    data: Dict[str, RealTimeDataPoint] = Field(..., description="实时数据")
    message: str = Field(..., description="响应消息")
    timestamp: str = Field(..., description="响应时间戳")

class DataHistoryResponse(BaseModel):
    success: bool = Field(..., description="操作是否成功")
    data: Dict[str, List[RealTimeDataPoint]] = Field(..., description="历史数据")
    message: str = Field(..., description="响应消息")
    timestamp: str = Field(..., description="响应时间戳")

class SystemStatusResponse(BaseModel):
    success: bool = Field(..., description="操作是否成功")
    status: Dict[str, Any] = Field(..., description="系统状态")
    message: str = Field(..., description="响应消息")
    timestamp: str = Field(..., description="响应时间戳")

class DataQualityResponse(BaseModel):
    success: bool = Field(..., description="操作是否成功")
    quality_metrics: Dict[str, Any] = Field(..., description="数据质量指标")
    message: str = Field(..., description="响应消息")
    timestamp: str = Field(..., description="响应时间戳")

class BatchDataRequest(BaseModel):
    data_points: List[RealTimeDataRequest] = Field(..., description="批量数据点")
    batch_id: Optional[str] = Field(None, description="批次ID")
    source_system: Optional[str] = Field(None, description="源系统")

# 数据质量检查函数
def check_data_quality(data_type: str, value: float) -> DataQuality:
    """检查数据质量"""
    if data_type not in data_quality_config:
        return DataQuality.UNCERTAIN
    
    config = data_quality_config[data_type]
    min_val, max_val = config['min'], config['max']
    expected_min, expected_max = config['expected_range']
    
    # 检查是否在有效范围内
    if value < min_val or value > max_val:
        return DataQuality.BAD
    
    # 检查是否在预期范围内
    if value < expected_min or value > expected_max:
        return DataQuality.UNCERTAIN
    
    return DataQuality.GOOD

# 数据转换函数
def convert_data_unit(value: float, from_unit: str, to_unit: str) -> float:
    """数据单位转换"""
    # 温度转换
    if from_unit == '°F' and to_unit == '°C':
        return (value - 32) * 5/9
    elif from_unit == '°C' and to_unit == '°F':
        return value * 9/5 + 32
    
    # 压力转换
    elif from_unit == 'psi' and to_unit == 'kPa':
        return value * 6.89476
    elif from_unit == 'kPa' and to_unit == 'psi':
        return value / 6.89476
    
    # 流量转换
    elif from_unit == 'gpm' and to_unit == 'm³/h':
        return value * 0.227125
    elif from_unit == 'm³/h' and to_unit == 'gpm':
        return value / 0.227125
    
    # 默认返回原值
    return value

# 模拟数据更新 - 扩展为更智能的模拟
def update_simulation_data():
    """模拟实时数据更新"""
    global real_time_data, data_history
    
    # 模拟数据变化 - 更真实的波动
    for key in real_time_data:
        if key in data_quality_config:
            current_value = real_time_data[key]['value']
            config = data_quality_config[key]
            expected_min, expected_max = config['expected_range']
            
            # 生成更真实的波动
            base_value = (expected_min + expected_max) / 2
            variation = (np.random.random() - 0.5) * (expected_max - expected_min) * 0.1
            new_value = base_value + variation
            
            # 确保在合理范围内
            new_value = max(config['min'], min(config['max'], new_value))
            
            # 更新数据
            real_time_data[key]['value'] = new_value
            real_time_data[key]['timestamp'] = datetime.now()
            real_time_data[key]['quality'] = check_data_quality(key, new_value)
            
            # 更新历史数据
            data_point = RealTimeDataPoint(
                value=new_value,
                unit=real_time_data[key]['unit'],
                quality=real_time_data[key]['quality'],
                timestamp=real_time_data[key]['timestamp'],
                source=DataSourceType.SIMULATION
            )
            
            if key not in data_history:
                data_history[key] = []
            
            data_history[key].append(data_point)
            # 保持最近1000个数据点
            if len(data_history[key]) > 1000:
                data_history[key] = data_history[key][-1000:]

# API端点 - 扩展为更完整的接口
@router.get("/current-data", response_model=RealTimeDataResponse)
async def get_current_data(
    data_types: Optional[str] = Query(None, description="数据类型，逗号分隔"),
    include_quality: bool = Query(True, description="是否包含质量信息")
):
    """获取当前实时数据"""
    try:
        # 仅在模拟模式下更新模拟数据，避免覆盖真实遥测数据
        if SIMULATION_MODE:
            update_simulation_data()
        
        # 过滤数据类型
        if data_types:
            requested_types = [t.strip() for t in data_types.split(',')]
            filtered_data = {k: v for k, v in real_time_data.items() if k in requested_types}
        else:
            filtered_data = real_time_data
        
        # 转换为响应格式
        response_data = {}
        for key, data in filtered_data.items():
            response_data[key] = RealTimeDataPoint(
                value=data['value'],
                unit=data['unit'],
                quality=data['quality'] if include_quality else DataQuality.GOOD,
                timestamp=data['timestamp'],
                source=DataSourceType.SIMULATION
            )
        
        return RealTimeDataResponse(
            success=True,
            data=response_data,
            message="获取实时数据成功",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"获取实时数据失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取实时数据失败: {str(e)}"
        )

@router.get("/data-history", response_model=DataHistoryResponse)
async def get_data_history(
    data_types: Optional[str] = Query(None, description="数据类型，逗号分隔"),
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间"),
    limit: int = Query(100, description="返回数据点数量限制")
):
    """获取数据历史记录"""
    try:
        # 过滤数据类型
        if data_types:
            requested_types = [t.strip() for t in data_types.split(',')]
            filtered_history = {k: v for k, v in data_history.items() if k in requested_types}
        else:
            filtered_history = data_history
        
        # 时间过滤
        if start_time or end_time:
            for key in filtered_history:
                filtered_history[key] = [
                    point for point in filtered_history[key]
                    if (not start_time or point.timestamp >= start_time) and
                       (not end_time or point.timestamp <= end_time)
                ]
        
        # 限制数据点数量
        for key in filtered_history:
            if len(filtered_history[key]) > limit:
                filtered_history[key] = filtered_history[key][-limit:]
        
        return DataHistoryResponse(
            success=True,
            data=filtered_history,
            message="获取数据历史成功",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"获取数据历史失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取数据历史失败: {str(e)}"
        )

@router.post("/update-data", response_model=RealTimeDataResponse)
async def update_real_time_data(request: RealTimeDataRequest):
    """更新实时数据（工厂数据接入）"""
    try:
        # 验证数据类型
        if request.data_type not in real_time_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的数据类型: {request.data_type}"
            )
        
        # 数据质量检查
        quality = check_data_quality(request.data_type, request.value)
        
        # 单位转换（如果需要）
        unit = request.unit or real_time_data[request.data_type]['unit']
        converted_value = request.value
        if request.unit and request.unit != real_time_data[request.data_type]['unit']:
            converted_value = convert_data_unit(request.value, request.unit, real_time_data[request.data_type]['unit'])
        
        # 更新数据
        real_time_data[request.data_type] = {
            'value': converted_value,
            'unit': real_time_data[request.data_type]['unit'],
            'quality': quality,
            'timestamp': request.timestamp or datetime.now()
        }
        
        # 更新历史数据
        data_point = RealTimeDataPoint(
            value=converted_value,
            unit=real_time_data[request.data_type]['unit'],
            quality=quality,
            timestamp=real_time_data[request.data_type]['timestamp'],
            source=request.source,
            device_id=request.device_id,
            location=request.location,
            metadata=request.metadata
        )
        
        if request.data_type not in data_history:
            data_history[request.data_type] = []
        
        data_history[request.data_type].append(data_point)
        # 保持最近1000个数据点
        if len(data_history[request.data_type]) > 1000:
            data_history[request.data_type] = data_history[request.data_type][-1000:]
        
        # 广播数据更新
        await manager.broadcast(json.dumps({
            "type": "data_update",
            "data_type": request.data_type,
            "value": converted_value,
            "unit": real_time_data[request.data_type]['unit'],
            "quality": quality.value,
            "timestamp": real_time_data[request.data_type]['timestamp'].isoformat(),
            "source": request.source.value
        }))
        
        logger.debug(f"实时数据已更新: {request.data_type} = {converted_value}")
        
        return RealTimeDataResponse(
            success=True,
            data={request.data_type: data_point},
            message=f"{request.data_type}数据更新成功",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"更新实时数据失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新实时数据失败: {str(e)}"
        )

@router.post("/batch-update", response_model=RealTimeDataResponse)
async def batch_update_data(request: BatchDataRequest):
    """批量更新实时数据"""
    try:
        updated_data = {}
        
        for data_point in request.data_points:
            # 验证数据类型
            if data_point.data_type not in real_time_data:
                logger.warning(f"跳过不支持的数据类型: {data_point.data_type}")
                continue
            
            # 数据质量检查
            quality = check_data_quality(data_point.data_type, data_point.value)
            
            # 单位转换（如果需要）
            unit = data_point.unit or real_time_data[data_point.data_type]['unit']
            converted_value = data_point.value
            if data_point.unit and data_point.unit != real_time_data[data_point.data_type]['unit']:
                converted_value = convert_data_unit(data_point.value, data_point.unit, real_time_data[data_point.data_type]['unit'])
            
            # 更新数据
            real_time_data[data_point.data_type] = {
                'value': converted_value,
                'unit': real_time_data[data_point.data_type]['unit'],
                'quality': quality,
                'timestamp': data_point.timestamp or datetime.now()
            }
            
            # 更新历史数据
            history_point = RealTimeDataPoint(
                value=converted_value,
                unit=real_time_data[data_point.data_type]['unit'],
                quality=quality,
                timestamp=real_time_data[data_point.data_type]['timestamp'],
                source=data_point.source,
                device_id=data_point.device_id,
                location=data_point.location,
                metadata=data_point.metadata
            )
            
            if data_point.data_type not in data_history:
                data_history[data_point.data_type] = []
            
            data_history[data_point.data_type].append(history_point)
            # 保持最近1000个数据点
            if len(data_history[data_point.data_type]) > 1000:
                data_history[data_point.data_type] = data_history[data_point.data_type][-1000:]
            
            updated_data[data_point.data_type] = history_point
        
        # 广播批量更新
        await manager.broadcast(json.dumps({
            "type": "batch_data_update",
            "batch_id": request.batch_id,
            "source_system": request.source_system,
            "updated_count": len(updated_data),
            "timestamp": datetime.now().isoformat()
        }))
        
        logger.info(f"批量数据更新成功，更新了 {len(updated_data)} 个数据点")
        
        return RealTimeDataResponse(
            success=True,
            data=updated_data,
            message=f"批量数据更新成功，更新了 {len(updated_data)} 个数据点",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"批量更新数据失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量更新数据失败: {str(e)}"
        )

@router.get("/data-quality", response_model=DataQualityResponse)
async def get_data_quality_metrics():
    """获取数据质量指标"""
    try:
        quality_metrics = {
            'total_data_points': len(real_time_data),
            'good_quality_count': 0,
            'bad_quality_count': 0,
            'uncertain_quality_count': 0,
            'no_data_count': 0,
            'quality_distribution': {},
            'data_freshness': {},
            'alerts': []
        }
        
        for data_type, data in real_time_data.items():
            quality = data['quality']
            quality_metrics['quality_distribution'][data_type] = quality.value
            
            if quality == DataQuality.GOOD:
                quality_metrics['good_quality_count'] += 1
            elif quality == DataQuality.BAD:
                quality_metrics['bad_quality_count'] += 1
                quality_metrics['alerts'].append({
                    'type': 'data_quality',
                    'data_type': data_type,
                    'message': f'{data_type} 数据质量异常',
                    'timestamp': datetime.now().isoformat()
                })
            elif quality == DataQuality.UNCERTAIN:
                quality_metrics['uncertain_quality_count'] += 1
            elif quality == DataQuality.NO_DATA:
                quality_metrics['no_data_count'] += 1
            
            # 检查数据新鲜度
            age = datetime.now() - data['timestamp']
            if age > timedelta(minutes=5):
                quality_metrics['data_freshness'][data_type] = {
                    'age_seconds': age.total_seconds(),
                    'status': 'stale' if age > timedelta(minutes=10) else 'warning'
                }
        
        return DataQualityResponse(
            success=True,
            quality_metrics=quality_metrics,
            message="获取数据质量指标成功",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"获取数据质量指标失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取数据质量指标失败: {str(e)}"
        )

@router.get("/system-status", response_model=SystemStatusResponse)
async def get_system_status():
    """获取系统状态"""
    try:
        # 计算系统状态指标
        total_connections = len(manager.active_connections)
        active_data_types = len([d for d in real_time_data.values() if d['quality'] != DataQuality.NO_DATA])
        total_data_types = len(real_time_data)
        
        # 检查数据质量
        quality_issues = len([d for d in real_time_data.values() if d['quality'] == DataQuality.BAD])
        
        # 检查数据新鲜度
        stale_data = len([
            d for d in real_time_data.values() 
            if datetime.now() - d['timestamp'] > timedelta(minutes=10)
        ])
        
        status_info = {
            'system_health': 'healthy' if quality_issues == 0 and stale_data == 0 else 'warning',
            'active_connections': total_connections,
            'data_coverage': f"{active_data_types}/{total_data_types}",
            'quality_issues': quality_issues,
            'stale_data_count': stale_data,
            'last_update': datetime.now().isoformat(),
            'uptime': "24h",  # 这里可以添加实际的运行时间计算
            'version': "1.0.0"
        }
        
        return SystemStatusResponse(
            success=True,
            status=status_info,
            message="获取系统状态成功",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"获取系统状态失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取系统状态失败: {str(e)}"
        )

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket连接端点"""
    await manager.connect(websocket)
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # 处理订阅请求
            if message.get('type') == 'subscribe':
                data_type = message.get('data_type')
                if data_type:
                    manager.subscribe(websocket, data_type)
                    await manager.send_personal_message(json.dumps({
                        'type': 'subscription_confirmed',
                        'data_type': data_type,
                        'message': f'已订阅 {data_type} 数据'
                    }), websocket)
            
            # 处理取消订阅请求
            elif message.get('type') == 'unsubscribe':
                data_type = message.get('data_type')
                if data_type:
                    manager.unsubscribe(websocket, data_type)
                    await manager.send_personal_message(json.dumps({
                        'type': 'unsubscription_confirmed',
                        'data_type': data_type,
                        'message': f'已取消订阅 {data_type} 数据'
                    }), websocket)
            
            # 处理数据请求
            elif message.get('type') == 'get_data':
                data_type = message.get('data_type')
                if data_type and data_type in real_time_data:
                    await manager.send_personal_message(json.dumps({
                        'type': 'data_response',
                        'data_type': data_type,
                        'data': real_time_data[data_type]
                    }), websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket错误: {str(e)}")
        manager.disconnect(websocket)

@router.post("/factory-data", response_model=RealTimeDataResponse)
async def receive_factory_data(data: Dict[str, Any]):
    """接收工厂数据（预留接口）"""
    try:
        # 这里可以添加工厂数据的具体处理逻辑
        # 例如：数据格式转换、验证、存储等
        
        logger.debug(f"接收到工厂数据: {data}")
        
        # 模拟处理工厂数据
        processed_data = {}
        for key, value in data.items():
            if key in real_time_data:
                # 处理工厂数据格式
                if isinstance(value, dict):
                    processed_value = value.get('value', 0)
                    unit = value.get('unit', real_time_data[key]['unit'])
                    quality = DataQuality(value.get('quality', DataQuality.GOOD.value))
                else:
                    processed_value = value
                    unit = real_time_data[key]['unit']
                    quality = DataQuality.GOOD
                
                # 更新数据
                real_time_data[key] = {
                    'value': processed_value,
                    'unit': unit,
                    'quality': quality,
                    'timestamp': datetime.now()
                }
                
                processed_data[key] = RealTimeDataPoint(
                    value=processed_value,
                    unit=unit,
                    quality=quality,
                    timestamp=datetime.now(),
                    source=DataSourceType.FACTORY
                )
        
        return RealTimeDataResponse(
            success=True,
            data=processed_data,
            message="工厂数据处理成功",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"处理工厂数据失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理工厂数据失败: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "service": "实时数据监测服务",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "active_connections": len(manager.active_connections)
    } 