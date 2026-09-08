from fastapi import APIRouter, HTTPException, status, Query, Depends
from pydantic import BaseModel, Field, validator
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from enum import Enum
import hashlib
import uuid

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix='/data-services', tags=["数据服务"])

# 数据格式枚举
class DataFormat(str, Enum):
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    EXCEL = "excel"
    PARQUET = "parquet"

# 数据处理类型枚举
class ProcessingType(str, Enum):
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    AGGREGATION = "aggregation"
    FILTERING = "filtering"
    ENRICHMENT = "enrichment"

# 数据模型
class DataProcessingRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="输入数据")
    processing_type: ProcessingType = Field(..., description="处理类型")
    parameters: Optional[Dict[str, Any]] = Field(None, description="处理参数")
    output_format: DataFormat = Field(DataFormat.JSON, description="输出格式")
    include_metadata: bool = Field(True, description="是否包含元数据")

class DataValidationRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="待验证数据")
    validation_rules: Dict[str, Any] = Field(..., description="验证规则")
    strict_mode: bool = Field(False, description="严格模式")

class DataTransformationRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="输入数据")
    transformations: List[Dict[str, Any]] = Field(..., description="转换规则")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="输出模式")

class DataExportRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="导出数据")
    format: DataFormat = Field(..., description="导出格式")
    filename: Optional[str] = Field(None, description="文件名")
    include_timestamp: bool = Field(True, description="是否包含时间戳")

class DataImportRequest(BaseModel):
    file_content: str = Field(..., description="文件内容（base64编码）")
    format: DataFormat = Field(..., description="文件格式")
    encoding: str = Field("utf-8", description="文件编码")
    options: Optional[Dict[str, Any]] = Field(None, description="导入选项")

class DataProcessingResponse(BaseModel):
    success: bool = Field(..., description="处理是否成功")
    processed_data: Dict[str, Any] = Field(..., description="处理后的数据")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    message: str = Field(..., description="响应消息")
    timestamp: str = Field(..., description="响应时间戳")

class DataValidationResponse(BaseModel):
    success: bool = Field(..., description="验证是否成功")
    is_valid: bool = Field(..., description="数据是否有效")
    validation_results: Dict[str, Any] = Field(..., description="验证结果")
    errors: List[str] = Field(..., description="错误信息")
    warnings: List[str] = Field(..., description="警告信息")
    timestamp: str = Field(..., description="响应时间戳")

# 数据验证规则
validation_rules = {
    'temperature': {
        'type': 'numeric',
        'min': -50,
        'max': 200,
        'required': True,
        'unit': '°C'
    },
    'pressure': {
        'type': 'numeric',
        'min': 0,
        'max': 1000,
        'required': True,
        'unit': 'kPa'
    },
    'flow': {
        'type': 'numeric',
        'min': 0,
        'max': 10000,
        'required': True,
        'unit': 'm³/h'
    },
    'gas': {
        'type': 'numeric',
        'min': 0,
        'max': 50000,
        'required': True,
        'unit': 'm³/h'
    },
    'oxygen': {
        'type': 'numeric',
        'min': 0,
        'max': 100,
        'required': True,
        'unit': '%'
    }
}

# 数据转换规则
transformation_rules = {
    'temperature': {
        'celsius_to_fahrenheit': lambda x: x * 9/5 + 32,
        'fahrenheit_to_celsius': lambda x: (x - 32) * 5/9,
        'celsius_to_kelvin': lambda x: x + 273.15,
        'kelvin_to_celsius': lambda x: x - 273.15
    },
    'pressure': {
        'kpa_to_psi': lambda x: x / 6.89476,
        'psi_to_kpa': lambda x: x * 6.89476,
        'kpa_to_bar': lambda x: x / 100,
        'bar_to_kpa': lambda x: x * 100
    },
    'flow': {
        'm3h_to_gpm': lambda x: x / 0.227125,
        'gpm_to_m3h': lambda x: x * 0.227125,
        'm3h_to_lpm': lambda x: x * 16.6667,
        'lpm_to_m3h': lambda x: x / 16.6667
    }
}

# 数据验证函数
def validate_data(data: Dict[str, Any], rules: Dict[str, Any], strict_mode: bool = False) -> Dict[str, Any]:
    """验证数据"""
    results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'validated_fields': {},
        'invalid_fields': {}
    }
    
    for field_name, field_rules in rules.items():
        if field_name not in data:
            if field_rules.get('required', False):
                results['errors'].append(f"缺少必需字段: {field_name}")
                results['invalid_fields'][field_name] = "missing"
                if strict_mode:
                    results['is_valid'] = False
            continue
        
        field_value = data[field_name]
        field_type = field_rules.get('type', 'any')
        
        # 类型检查
        if field_type == 'numeric':
            try:
                numeric_value = float(field_value)
                data[field_name] = numeric_value
            except (ValueError, TypeError):
                results['errors'].append(f"{field_name} 必须是数字")
                results['invalid_fields'][field_name] = "invalid_type"
                if strict_mode:
                    results['is_valid'] = False
                continue
        
        # 范围检查
        if 'min' in field_rules and field_value < field_rules['min']:
            results['warnings'].append(f"{field_name} 值 {field_value} 低于最小值 {field_rules['min']}")
            results['invalid_fields'][field_name] = "below_min"
        
        if 'max' in field_rules and field_value > field_rules['max']:
            results['warnings'].append(f"{field_name} 值 {field_value} 高于最大值 {field_rules['max']}")
            results['invalid_fields'][field_name] = "above_max"
        
        # 格式检查
        if 'format' in field_rules:
            # 这里可以添加更复杂的格式验证
            pass
        
        results['validated_fields'][field_name] = field_value
    
    return results

# 数据转换函数
def transform_data(data: Dict[str, Any], transformations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """转换数据"""
    transformed_data = data.copy()
    
    for transformation in transformations:
        transform_type = transformation.get('type')
        field_name = transformation.get('field')
        target_field = transformation.get('target_field', field_name)
        parameters = transformation.get('parameters', {})
        
        if field_name not in transformed_data:
            continue
        
        if transform_type == 'unit_conversion':
            from_unit = parameters.get('from_unit')
            to_unit = parameters.get('to_unit')
            if from_unit and to_unit and field_name in transformation_rules:
                conversion_key = f"{from_unit}_to_{to_unit}"
                if conversion_key in transformation_rules[field_name]:
                    original_value = transformed_data[field_name]
                    converted_value = transformation_rules[field_name][conversion_key](original_value)
                    transformed_data[target_field] = converted_value
        
        elif transform_type == 'calculation':
            formula = parameters.get('formula')
            if formula:
                # 简单的公式计算
                try:
                    # 这里可以实现更复杂的公式解析
                    if formula == 'power_factor':
                        voltage = transformed_data.get('voltage', 0)
                        current = transformed_data.get('current', 0)
                        power = transformed_data.get('power', 0)
                        if voltage > 0 and current > 0:
                            apparent_power = voltage * current
                            power_factor = power / apparent_power if apparent_power > 0 else 0
                            transformed_data[target_field] = power_factor
                except Exception as e:
                    logger.warning(f"计算转换失败: {str(e)}")
        
        elif transform_type == 'aggregation':
            aggregation_type = parameters.get('aggregation_type', 'mean')
            if aggregation_type == 'mean':
                values = [v for v in transformed_data.values() if isinstance(v, (int, float))]
                if values:
                    transformed_data[target_field] = sum(values) / len(values)
            elif aggregation_type == 'sum':
                values = [v for v in transformed_data.values() if isinstance(v, (int, float))]
                transformed_data[target_field] = sum(values)
        
        elif transform_type == 'filtering':
            condition = parameters.get('condition')
            if condition:
                # 简单的条件过滤
                if condition.get('type') == 'range':
                    min_val = condition.get('min')
                    max_val = condition.get('max')
                    if min_val is not None and transformed_data[field_name] < min_val:
                        transformed_data[field_name] = min_val
                    if max_val is not None and transformed_data[field_name] > max_val:
                        transformed_data[field_name] = max_val
    
    return transformed_data

# 数据导出函数
def export_data(data: Dict[str, Any], format: DataFormat, filename: Optional[str] = None) -> Dict[str, Any]:
    """导出数据"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data_export_{timestamp}"
    
    export_result = {
        'filename': filename,
        'format': format.value,
        'data_size': len(str(data)),
        'export_time': datetime.now().isoformat(),
        'content': None
    }
    
    if format == DataFormat.JSON:
        export_result['content'] = json.dumps(data, indent=2, ensure_ascii=False)
        export_result['filename'] += '.json'
    
    elif format == DataFormat.CSV:
        # 将嵌套字典转换为CSV格式
        csv_lines = []
        for key, value in data.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    csv_lines.append(f"{key}.{sub_key},{sub_value}")
            else:
                csv_lines.append(f"{key},{value}")
        
        export_result['content'] = '\n'.join(csv_lines)
        export_result['filename'] += '.csv'
    
    elif format == DataFormat.XML:
        # 简单的XML格式转换
        xml_lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<data>']
        for key, value in data.items():
            if isinstance(value, dict):
                xml_lines.append(f'  <{key}>')
                for sub_key, sub_value in value.items():
                    xml_lines.append(f'    <{sub_key}>{sub_value}</{sub_key}>')
                xml_lines.append(f'  </{key}>')
            else:
                xml_lines.append(f'  <{key}>{value}</{key}>')
        xml_lines.append('</data>')
        
        export_result['content'] = '\n'.join(xml_lines)
        export_result['filename'] += '.xml'
    
    return export_result

# 数据导入函数
def import_data(file_content: str, format: DataFormat, encoding: str = "utf-8", options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """导入数据"""
    import_result = {
        'success': False,
        'data': {},
        'errors': [],
        'warnings': [],
        'import_time': datetime.now().isoformat()
    }
    
    try:
        if format == DataFormat.JSON:
            import_result['data'] = json.loads(file_content)
            import_result['success'] = True
        
        elif format == DataFormat.CSV:
            lines = file_content.strip().split('\n')
            if len(lines) < 2:
                import_result['errors'].append("CSV文件至少需要标题行和一行数据")
                return import_result
            
            headers = lines[0].split(',')
            data = {}
            
            for line in lines[1:]:
                values = line.split(',')
                if len(values) == len(headers):
                    for i, header in enumerate(headers):
                        try:
                            # 尝试转换为数字
                            data[header] = float(values[i])
                        except ValueError:
                            data[header] = values[i]
            
            import_result['data'] = data
            import_result['success'] = True
        
        elif format == DataFormat.XML:
            # 简单的XML解析
            import_result['warnings'].append("XML解析功能需要更复杂的实现")
            import_result['data'] = {'xml_content': file_content}
            import_result['success'] = True
        
        else:
            import_result['errors'].append(f"不支持的格式: {format.value}")
    
    except Exception as e:
        import_result['errors'].append(f"导入失败: {str(e)}")
    
    return import_result

# API端点
@router.post("/process", response_model=DataProcessingResponse)
async def process_data(request: DataProcessingRequest):
    """数据处理端点"""
    try:
        processed_data = request.data.copy()
        metadata = {
            'processing_type': request.processing_type.value,
            'input_size': len(str(request.data)),
            'processing_time': datetime.now().isoformat()
        }
        
        if request.processing_type == ProcessingType.VALIDATION:
            # 数据验证
            validation_rules = request.parameters or {}
            validation_result = validate_data(processed_data, validation_rules)
            metadata['validation_result'] = validation_result
        
        elif request.processing_type == ProcessingType.TRANSFORMATION:
            # 数据转换
            transformations = request.parameters.get('transformations', [])
            processed_data = transform_data(processed_data, transformations)
            metadata['transformations_applied'] = len(transformations)
        
        elif request.processing_type == ProcessingType.AGGREGATION:
            # 数据聚合
            aggregation_type = request.parameters.get('aggregation_type', 'mean')
            if aggregation_type == 'mean':
                numeric_values = [v for v in processed_data.values() if isinstance(v, (int, float))]
                if numeric_values:
                    processed_data['aggregated_mean'] = sum(numeric_values) / len(numeric_values)
            metadata['aggregation_type'] = aggregation_type
        
        elif request.processing_type == ProcessingType.FILTERING:
            # 数据过滤
            filter_conditions = request.parameters.get('conditions', {})
            for field, condition in filter_conditions.items():
                if field in processed_data:
                    if condition.get('type') == 'range':
                        min_val = condition.get('min')
                        max_val = condition.get('max')
                        if min_val is not None and processed_data[field] < min_val:
                            processed_data[field] = min_val
                        if max_val is not None and processed_data[field] > max_val:
                            processed_data[field] = max_val
            metadata['filters_applied'] = len(filter_conditions)
        
        elif request.processing_type == ProcessingType.ENRICHMENT:
            # 数据丰富
            enrichment_data = request.parameters.get('enrichment_data', {})
            processed_data.update(enrichment_data)
            metadata['enrichment_fields_added'] = len(enrichment_data)
        
        metadata['output_size'] = len(str(processed_data))
        
        return DataProcessingResponse(
            success=True,
            processed_data=processed_data,
            metadata=metadata if request.include_metadata else None,
            message="数据处理成功",
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"数据处理失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"数据处理失败: {str(e)}"
        )

@router.post("/validate", response_model=DataValidationResponse)
async def validate_data_endpoint(request: DataValidationRequest):
    """数据验证端点"""
    try:
        validation_result = validate_data(
            request.data, 
            request.validation_rules, 
            request.strict_mode
        )
        
        return DataValidationResponse(
            success=True,
            is_valid=validation_result['is_valid'],
            validation_results=validation_result,
            errors=validation_result['errors'],
            warnings=validation_result['warnings'],
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"数据验证失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"数据验证失败: {str(e)}"
        )

@router.post("/transform", response_model=DataProcessingResponse)
async def transform_data_endpoint(request: DataTransformationRequest):
    """数据转换端点"""
    try:
        transformed_data = transform_data(request.data, request.transformations)
        
        metadata = {
            'transformations_applied': len(request.transformations),
            'input_size': len(str(request.data)),
            'output_size': len(str(transformed_data))
        }
        
        return DataProcessingResponse(
            success=True,
            processed_data=transformed_data,
            metadata=metadata,
            message="数据转换成功",
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"数据转换失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"数据转换失败: {str(e)}"
        )

@router.post("/export", response_model=DataProcessingResponse)
async def export_data_endpoint(request: DataExportRequest):
    """数据导出端点"""
    try:
        export_result = export_data(
            request.data, 
            request.format, 
            request.filename
        )
        
        metadata = {
            'export_format': request.format.value,
            'filename': export_result['filename'],
            'data_size': export_result['data_size']
        }
        
        return DataProcessingResponse(
            success=True,
            processed_data={'export_result': export_result},
            metadata=metadata,
            message="数据导出成功",
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"数据导出失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"数据导出失败: {str(e)}"
        )

@router.post("/import", response_model=DataProcessingResponse)
async def import_data_endpoint(request: DataImportRequest):
    """数据导入端点"""
    try:
        import_result = import_data(
            request.file_content, 
            request.format, 
            request.encoding, 
            request.options
        )
        
        if not import_result['success']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"数据导入失败: {'; '.join(import_result['errors'])}"
            )
        
        metadata = {
            'import_format': request.format.value,
            'encoding': request.encoding,
            'import_errors': import_result['errors'],
            'import_warnings': import_result['warnings']
        }
        
        return DataProcessingResponse(
            success=True,
            processed_data=import_result['data'],
            metadata=metadata,
            message="数据导入成功",
            timestamp=datetime.now().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"数据导入失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"数据导入失败: {str(e)}"
        )

@router.get("/validation-rules")
async def get_validation_rules():
    """获取验证规则"""
    return {
        "success": True,
        "rules": validation_rules,
        "message": "获取验证规则成功",
        "timestamp": datetime.now().isoformat()
    }

@router.get("/transformation-rules")
async def get_transformation_rules():
    """获取转换规则"""
    return {
        "success": True,
        "rules": transformation_rules,
        "message": "获取转换规则成功",
        "timestamp": datetime.now().isoformat()
    }

@router.get("/supported-formats")
async def get_supported_formats():
    """获取支持的格式"""
    return {
        "success": True,
        "formats": [format.value for format in DataFormat],
        "message": "获取支持格式成功",
        "timestamp": datetime.now().isoformat()
    }

@router.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "service": "数据服务",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    } 