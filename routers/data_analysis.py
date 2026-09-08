from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix='/data-analysis', tags=["数据分析"])

class DataAnalysisRequest(BaseModel):
    data_type: str  # "production", "quality", "process"
    analysis_type: str  # "statistics", "trend", "correlation"
    parameters: Optional[Dict[str, Any]] = None

class DataAnalysisResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    message: str
    timestamp: str

# 模拟数据路径
DATA_PATHS = {
    "production": "data/production_data.csv",
    "quality": "data/quality_data.csv", 
    "process": "data/process_data.csv"
}

def load_sample_data():
    """加载示例数据"""
    # 生产数据
    production_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'tin_concentration': np.random.normal(45, 5, 100),
        'temperature': np.random.normal(1200, 50, 100),
        'pressure': np.random.normal(1.2, 0.1, 100),
        'flow_rate': np.random.normal(100, 10, 100),
        'yield_rate': np.random.normal(0.85, 0.05, 100)
    })
    
    # 质量数据
    quality_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'purity': np.random.normal(0.95, 0.02, 100),
        'impurity_content': np.random.normal(0.03, 0.01, 100),
        'particle_size': np.random.normal(50, 5, 100),
        'moisture': np.random.normal(0.02, 0.005, 100)
    })
    
    # 工艺数据
    process_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'reaction_time': np.random.normal(120, 10, 100),
        'catalyst_usage': np.random.normal(0.5, 0.05, 100),
        'energy_consumption': np.random.normal(800, 50, 100),
        'waste_generation': np.random.normal(0.1, 0.02, 100)
    })
    
    return {
        "production": production_data,
        "quality": quality_data,
        "process": process_data
    }

def calculate_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """计算数据统计信息"""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats = {}
        
        for col in numeric_cols:
            stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median()),
                'q25': float(df[col].quantile(0.25)),
                'q75': float(df[col].quantile(0.75))
            }
        
        return {
            'summary': stats,
            'total_records': len(df),
            'missing_values': df.isnull().sum().to_dict()
        }
    except Exception as e:
        logger.error(f"统计计算失败: {str(e)}")
        raise

def calculate_trends(df: pd.DataFrame) -> Dict[str, Any]:
    """计算趋势分析"""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        trends = {}
        
        for col in numeric_cols:
            # 计算线性趋势
            x = np.arange(len(df))
            y = df[col].values
            slope, intercept = np.polyfit(x, y, 1)
            
            # 计算相关系数
            correlation = np.corrcoef(x, y)[0, 1]
            
            trends[col] = {
                'slope': float(slope),
                'intercept': float(intercept),
                'correlation': float(correlation),
                'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                'trend_strength': abs(correlation)
            }
        
        return trends
    except Exception as e:
        logger.error(f"趋势分析失败: {str(e)}")
        raise

def calculate_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    """计算相关性分析"""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        # 找出强相关性（绝对值 > 0.7）
        strong_correlations = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_correlations.append({
                        'variable1': numeric_cols[i],
                        'variable2': numeric_cols[j],
                        'correlation': float(corr_value),
                        'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate'
                    })
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': strong_correlations
        }
    except Exception as e:
        logger.error(f"相关性分析失败: {str(e)}")
        raise

@router.post("/analyze", response_model=DataAnalysisResponse)
async def analyze_data(request: DataAnalysisRequest):
    """
    数据分析API
    
    Args:
        request: 数据分析请求
        
    Returns:
        DataAnalysisResponse: 分析结果
    """
    try:
        logger.info(f"开始数据分析: {request.data_type}, {request.analysis_type}")
        
        # 加载数据
        sample_data = load_sample_data()
        
        if request.data_type not in sample_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的数据类型: {request.data_type}"
            )
        
        df = sample_data[request.data_type]
        
        # 根据分析类型执行相应的分析
        if request.analysis_type == "statistics":
            result = calculate_statistics(df)
        elif request.analysis_type == "trend":
            result = calculate_trends(df)
        elif request.analysis_type == "correlation":
            result = calculate_correlations(df)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的分析类型: {request.analysis_type}"
            )
        
        logger.info(f"数据分析完成: {request.data_type}, {request.analysis_type}")
        
        return DataAnalysisResponse(
            success=True,
            data=result,
            message=f"{request.data_type}数据{request.analysis_type}分析完成",
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"数据分析失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"数据分析失败: {str(e)}"
        )

@router.get("/data-summary")
async def get_data_summary():
    """获取数据概览"""
    try:
        sample_data = load_sample_data()
        summary = {}
        
        for data_type, df in sample_data.items():
            summary[data_type] = {
                'total_records': len(df),
                'columns': list(df.columns),
                'date_range': {
                    'start': df['date'].min().strftime('%Y-%m-%d'),
                    'end': df['date'].max().strftime('%Y-%m-%d')
                }
            }
        
        return {
            "success": True,
            "data": summary,
            "message": "数据概览获取成功"
        }
    except Exception as e:
        logger.error(f"获取数据概览失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取数据概览失败: {str(e)}"
        )

@router.get("/available-analyses")
async def get_available_analyses():
    """获取可用的分析类型"""
    return {
        "data_types": ["production", "quality", "process"],
        "analysis_types": ["statistics", "trend", "correlation"],
        "descriptions": {
            "statistics": "计算基本统计信息（均值、标准差、分位数等）",
            "trend": "分析数据趋势和变化方向",
            "correlation": "计算变量间的相关性"
        }
    } 