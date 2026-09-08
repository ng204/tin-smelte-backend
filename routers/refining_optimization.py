from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

router = APIRouter(prefix='/refining-optimization', tags=["锡精炼优化"])

# CSV文件路径
CSV_FILE_PATH = os.path.join("data", "trajectory_run_3_start_2345 (3).csv")

class OptimizationResponse(BaseModel):
    """优化响应模型"""
    success: bool
    message: str
    data: Optional[Dict] = None

class TrajectoryData(BaseModel):
    """温度控制轨迹数据模型"""
    timesteps: List[int]
    temp_1: List[float]
    temp_2: List[float]
    temp_3: List[float]
    temp_4: List[float]
    temp_5: List[float]
    action_1: List[int]
    action_2: List[int]
    action_3: List[int]
    action_4: List[int]
    action_5: List[int]
    phase: List[str]
    target_temps: Dict[str, float]  # 目标温度

@router.post("/start", response_model=OptimizationResponse)
async def start_optimization():
    """
    开始锡精炼优化 - 基于MBRL的结晶机温度控制
    
    读取trajectory CSV数据，返回温度控制轨迹可视化数据
    """
    try:
        logger.info(f"开始读取优化轨迹数据: {CSV_FILE_PATH}")
        
        # 检查文件是否存在
        if not os.path.exists(CSV_FILE_PATH):
            raise FileNotFoundError(f"数据文件不存在: {CSV_FILE_PATH}")
        
        # 读取CSV文件
        df = pd.read_csv(CSV_FILE_PATH)
        logger.info(f"成功读取数据，共 {len(df)} 行")
        
        # 提取数据
        trajectory_data = TrajectoryData(
            timesteps=df['timestep'].tolist(),
            temp_1=df['temp_1'].tolist(),
            temp_2=df['temp_2'].tolist(),
            temp_3=df['temp_3'].tolist(),
            temp_4=df['temp_4'].tolist(),
            temp_5=df['temp_5'].tolist(),
            action_1=df['action_1'].tolist(),
            action_2=df['action_2'].tolist(),
            action_3=df['action_3'].tolist(),
            action_4=df['action_4'].tolist(),
            action_5=df['action_5'].tolist(),
            phase=df['phase'].tolist(),
            target_temps={
                'temp_1': 183.0,  # 目标温度（基于图示）
                'temp_2': 198.0,
                'temp_3': 214.0,
                'temp_4': 228.0,
                'temp_5': 232.0
            }
        )
        
        logger.info("优化数据处理完成")
        
        return OptimizationResponse(
            success=True,
            message="优化启动成功",
            data=trajectory_data.dict()
        )
        
    except FileNotFoundError as e:
        logger.error(f"文件未找到: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"优化启动失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"优化启动失败: {str(e)}")

@router.get("/status")
async def get_optimization_status():
    """
    获取优化状态
    """
    try:
        # 检查数据文件是否存在
        file_exists = os.path.exists(CSV_FILE_PATH)
        
        return {
            "success": True,
            "ready": file_exists,
            "data_file": CSV_FILE_PATH,
            "message": "系统就绪" if file_exists else "数据文件缺失"
        }
    except Exception as e:
        logger.error(f"获取状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
