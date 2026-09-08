from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import logging
import json
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix='/warehouse', tags=["仓储物流管理"])

# 数据持久化文件路径
DATA_DIR = Path("warehouse_data")
MATRICES_FILE = DATA_DIR / "matrices.json"
DETAILS_FILE = DATA_DIR / "details.json"

# 确保数据目录存在
DATA_DIR.mkdir(exist_ok=True)

# 每个货架的最大容量（吨）
MAX_SHELF_CAPACITY = 5.0

def save_warehouse_data():
    """保存仓库数据到文件"""
    try:
        # 保存矩阵数据
        matrices_dict = {
            str(k): v.tolist() for k, v in warehouse_matrices.items()
        }
        with open(MATRICES_FILE, 'w', encoding='utf-8') as f:
            json.dump(matrices_dict, f, ensure_ascii=False, indent=2)
        
        # 保存详细信息
        with open(DETAILS_FILE, 'w', encoding='utf-8') as f:
            json.dump(warehouse_shelf_details, f, ensure_ascii=False, indent=2)
        
        logger.info("仓库数据已保存")
    except Exception as e:
        logger.error(f"保存仓库数据失败: {str(e)}")

def load_warehouse_data():
    """从文件加载仓库数据"""
    global warehouse_matrices, warehouse_shelf_details
    
    try:
        # 加载矩阵数据
        if MATRICES_FILE.exists():
            with open(MATRICES_FILE, 'r', encoding='utf-8') as f:
                matrices_dict = json.load(f)
                warehouse_matrices = {
                    int(k): np.array(v, dtype=float) for k, v in matrices_dict.items()
                }
            logger.info("仓库矩阵数据已加载")
        else:
            logger.info("未找到矩阵数据文件，使用初始化数据")
            warehouse_matrices = {
                1: np.zeros((6, 4), dtype=float),
                2: np.zeros((6, 4), dtype=float),
                3: np.zeros((6, 4), dtype=float),
            }
        
        # 加载详细信息
        if DETAILS_FILE.exists():
            with open(DETAILS_FILE, 'r', encoding='utf-8') as f:
                loaded_details = json.load(f)
                warehouse_shelf_details = {
                    int(k): v for k, v in loaded_details.items()
                }
            logger.info("仓库详细信息已加载")
        else:
            logger.info("未找到详细信息文件，使用初始化数据")
            warehouse_shelf_details = {1: {}, 2: {}, 3: {}}
            
    except Exception as e:
        logger.error(f"加载仓库数据失败: {str(e)}")
        # 如果加载失败，使用初始化数据
        warehouse_matrices = {
            1: np.zeros((6, 4), dtype=float),
            2: np.zeros((6, 4), dtype=float),
            3: np.zeros((6, 4), dtype=float),
        }
        warehouse_shelf_details = {1: {}, 2: {}, 3: {}}

# 三个仓库的货架状态矩阵 (6行 x 4列 = 24个货架)
# 0表示空闲，>=1表示占用的重量（吨）
warehouse_matrices = {}

# 货架详细信息存储 {warehouse_id: {shelf_name: {material, weight, inbound_time}}}
warehouse_shelf_details = {}

# 启动时加载数据
load_warehouse_data()

def matrix_to_shelves(warehouse_id: int, matrix: np.ndarray) -> List[dict]:
    """将矩阵转换为货架列表"""
    shelves = []
    rows = ['A', 'B', 'C', 'D', 'E', 'F']
    details = warehouse_shelf_details[warehouse_id]
    
    for i in range(6):
        for j in range(4):
            shelf_name = f"{rows[i]}{j+1}"
            weight = float(matrix[i][j])
            status = "occupied" if weight > 0 else "free"
            
            # 获取货架详细信息
            detail_info = details.get(shelf_name, {})
            
            shelves.append({
                "id": f"{warehouse_id}-{shelf_name}",
                "name": shelf_name,
                "status": status,
                "weight": weight,
                "capacity": MAX_SHELF_CAPACITY,
                "material": detail_info.get("material", ""),
                "materialName": detail_info.get("materialName", ""),
                "inboundTime": detail_info.get("inboundTime", "")
            })
    return shelves

class WarehouseInfo(BaseModel):
    """仓库信息"""
    id: int
    name: str
    totalShelves: int
    freeShelves: int
    shelves: List[dict]

class WarehouseResponse(BaseModel):
    """仓库列表响应"""
    warehouses: List[WarehouseInfo]
    totalWarehouses: int
    freeShelvesCount: int
    occupiedShelvesCount: int
    totalShelvesCount: int
    success: bool = True

@router.get("/info", response_model=WarehouseResponse)
async def get_warehouse_info():
    """
    获取所有仓库信息
    
    Returns:
        WarehouseResponse: 包含所有仓库的详细信息
    """
    try:
        warehouses = []
        total_free = 0
        total_occupied = 0
        total_shelves = 0
        
        for wh_id in [1, 2, 3]:
            matrix = warehouse_matrices[wh_id]
            shelves = matrix_to_shelves(wh_id, matrix)
            
            free_count = int(np.sum(matrix == 0))
            occupied_count = int(np.sum(matrix > 0))
            
            warehouses.append(WarehouseInfo(
                id=wh_id,
                name=f"{wh_id}号仓库",
                totalShelves=24,
                freeShelves=free_count,
                shelves=shelves
            ))
            
            total_free += free_count
            total_occupied += occupied_count
            total_shelves += 24
        
        return WarehouseResponse(
            warehouses=warehouses,
            totalWarehouses=3,
            freeShelvesCount=total_free,
            occupiedShelvesCount=total_occupied,
            totalShelvesCount=total_shelves,
            success=True
        )
        
    except Exception as e:
        logger.error(f"获取仓库信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取仓库信息失败: {str(e)}")

class ShelfUpdateRequest(BaseModel):
    """货架状态更新请求"""
    warehouseId: int
    shelfName: str
    weight: float  # 货架上的重量（吨），0表示空闲

@router.post("/update-shelf")
async def update_shelf_status(request: ShelfUpdateRequest):
    """
    更新单个货架状态
    
    Args:
        request: 包含仓库ID、货架名称和重量的请求
        
    Returns:
        dict: 更新结果
    """
    try:
        if request.warehouseId not in [1, 2, 3]:
            raise HTTPException(status_code=400, detail="无效的仓库ID")
        
        if request.weight < 0 or request.weight > MAX_SHELF_CAPACITY:
            raise HTTPException(status_code=400, detail=f"重量必须在0到{MAX_SHELF_CAPACITY}吨之间")
        
        # 解析货架名称 (如 "A1" -> row=0, col=0)
        rows_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
        shelf_row = request.shelfName[0].upper()
        shelf_col = int(request.shelfName[1:]) - 1
        
        if shelf_row not in rows_map or shelf_col < 0 or shelf_col > 3:
            raise HTTPException(status_code=400, detail="无效的货架名称")
        
        row = rows_map[shelf_row]
        
        # 更新矩阵
        warehouse_matrices[request.warehouseId][row][shelf_col] = request.weight
        
        logger.info(f"更新货架状态: 仓库{request.warehouseId}, 货架{request.shelfName}, 重量={request.weight}吨")
        
        # 保存数据到文件
        save_warehouse_data()
        
        return {
            "success": True,
            "message": f"货架{request.shelfName}状态已更新",
            "warehouseId": request.warehouseId,
            "shelfName": request.shelfName,
            "weight": request.weight
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新货架状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新货架状态失败: {str(e)}")

@router.get("/matrix/{warehouse_id}")
async def get_warehouse_matrix(warehouse_id: int):
    """
    获取指定仓库的矩阵数据
    
    Args:
        warehouse_id: 仓库ID (1, 2, 或 3)
        
    Returns:
        dict: 包含矩阵数据
    """
    try:
        if warehouse_id not in [1, 2, 3]:
            raise HTTPException(status_code=400, detail="无效的仓库ID")
        
        matrix = warehouse_matrices[warehouse_id]
        
        return {
            "success": True,
            "warehouseId": warehouse_id,
            "matrix": matrix.tolist(),
            "freeShelves": int(np.sum(matrix == 0)),
            "occupiedShelves": int(np.sum(matrix > 0))
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取仓库矩阵失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取仓库矩阵失败: {str(e)}")

@router.post("/reset")
async def reset_all_warehouses():
    """
    重置所有仓库状态为空闲
    
    Returns:
        dict: 重置结果
    """
    try:
        for wh_id in [1, 2, 3]:
            warehouse_matrices[wh_id] = np.zeros((6, 4), dtype=float)
            warehouse_shelf_details[wh_id] = {}
        
        logger.info("所有仓库已重置为空闲状态")
        
        # 保存数据到文件
        save_warehouse_data()
        
        return {
            "success": True,
            "message": "所有仓库已重置为空闲状态"
        }
        
    except Exception as e:
        logger.error(f"重置仓库失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"重置仓库失败: {str(e)}")

class ScheduleRequest(BaseModel):
    """调度请求"""
    operationType: str  # inbound, outbound, transfer
    materialType: str
    quantity: float
    sourceWarehouse: Optional[str] = None
    targetWarehouse: Optional[str] = None

@router.post("/schedule")
async def schedule_warehouse(request: ScheduleRequest):
    """
    仓储调度算法（带5吨限制和最优仓库选择）
    
    Args:
        request: 调度请求参数
        
    Returns:
        dict: 调度结果
    """
    try:
        import time
        import math
        
        # 模拟计算延时
        time.sleep(0.5)
        
        # 物料名称映射
        material_names = {
            'tin_ore': '锡精矿',
            'crude_tin': '粗锡',
            'refined_tin': '精锡',
            'coal': '还原煤',
            'limestone': '石灰石',
            'fluorite': '萤石',
            'slag': '炉渣'
        }
        
        # 物料处理时间系数（分钟/吨）- 最少30分钟/吨
        material_time_factors = {
            'tin_ore': 50.0,       # 锡精矿，密度大，难处理
            'crude_tin': 45.0,     # 粗锡，较重
            'refined_tin': 40.0,   # 精锡
            'coal': 30.0,          # 还原煤，最轻
            'limestone': 35.0,     # 石灰石
            'fluorite': 35.0,      # 萤石
            'slag': 48.0           # 炉渣，重且难处理
        }
        
        rows = ['A', 'B', 'C', 'D', 'E', 'F']
        shelf_allocations = []
        selected_warehouse = None
        
        def get_shelf_coords(shelf_name: str):
            """获取货架的行列坐标"""
            rows_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
            return rows_map[shelf_name[0]], int(shelf_name[1:]) - 1
        
        def find_best_warehouse_for_inbound(quantity: float, preferred_wh: int = None):
            """找到最优仓库用于入库"""
            warehouses_to_check = [preferred_wh] if preferred_wh else [1, 2, 3]
            if preferred_wh:
                warehouses_to_check.extend([w for w in [1, 2, 3] if w != preferred_wh])
            
            for wh_id in warehouses_to_check:
                matrix = warehouse_matrices[wh_id]
                available_shelves = []
                
                for i in range(6):
                    for j in range(4):
                        remaining_capacity = MAX_SHELF_CAPACITY - matrix[i][j]
                        if remaining_capacity > 0:
                            available_shelves.append({
                                'name': f"{rows[i]}{j+1}",
                                'capacity': remaining_capacity,
                                'row': i,
                                'col': j
                            })
                
                # 计算该仓库能否容纳该批次物料
                total_capacity = sum(s['capacity'] for s in available_shelves)
                if total_capacity >= quantity:
                    return wh_id, available_shelves
            
            return None, []
        
        if request.operationType == 'inbound':
            # 入库：找到最优仓库
            preferred_wh = int(request.targetWarehouse) if request.targetWarehouse else None
            selected_warehouse, available_shelves = find_best_warehouse_for_inbound(request.quantity, preferred_wh)
            
            if not selected_warehouse:
                raise HTTPException(status_code=400, detail="所有仓库容量不足，无法完成入库")
            
            # 分配货架
            remaining = request.quantity
            allocated_shelves = []
            
            # 按容量排序，优先使用容量大的货架
            available_shelves.sort(key=lambda x: x['capacity'], reverse=True)
            
            for shelf in available_shelves:
                if remaining <= 0:
                    break
                
                alloc_weight = min(remaining, shelf['capacity'])
                allocated_shelves.append({
                    'name': shelf['name'],
                    'weight': round(alloc_weight, 2)
                })
                remaining -= alloc_weight
            
            if allocated_shelves:
                shelf_allocations.append({
                    'warehouse': f'{selected_warehouse}号仓库',
                    'shelves': [s['name'] for s in allocated_shelves],
                    'weights': [s['weight'] for s in allocated_shelves],
                    'totalWeight': request.quantity
                })
        
        elif request.operationType == 'outbound':
            # 出库：从指定仓库的占用货架，根据物料类型筛选
            source_wh = int(request.sourceWarehouse)
            matrix = warehouse_matrices[source_wh]
            details = warehouse_shelf_details[source_wh]
            selected_warehouse = source_wh
            
            occupied_shelves = []
            for i in range(6):
                for j in range(4):
                    if matrix[i][j] > 0:
                        shelf_name = f"{rows[i]}{j+1}"
                        shelf_detail = details.get(shelf_name, {})
                        # 只选择匹配物料类型的货架
                        if shelf_detail.get('material') == request.materialType:
                            occupied_shelves.append({
                                'name': shelf_name,
                                'weight': float(matrix[i][j]),
                                'row': i,
                                'col': j
                            })
            
            if not occupied_shelves:
                raise HTTPException(status_code=400, detail=f"{source_wh}号仓库没有{material_names[request.materialType]}可出库")
            
            # 计算该物料的总可用重量
            total_available = sum(s['weight'] for s in occupied_shelves)
            
            # 如果请求的数量超过可用数量，使用最大可用数量
            actual_quantity = min(request.quantity, total_available)
            if actual_quantity < request.quantity:
                logger.warning(f"{source_wh}号仓库{material_names[request.materialType]}不足，请求{request.quantity}吨，实际可出库{actual_quantity}吨")
            
            # 分配出库货架
            remaining = actual_quantity
            allocated_shelves = []
            
            for shelf in occupied_shelves:
                if remaining <= 0:
                    break
                
                alloc_weight = min(remaining, shelf['weight'])
                allocated_shelves.append({
                    'name': shelf['name'],
                    'weight': round(alloc_weight, 2)
                })
                remaining -= alloc_weight
            
            shelf_allocations.append({
                'warehouse': f'{source_wh}号仓库',
                'shelves': [s['name'] for s in allocated_shelves],
                'weights': [s['weight'] for s in allocated_shelves],
                'totalWeight': actual_quantity,
                'requestedWeight': request.quantity,
                'shortage': request.quantity - actual_quantity if actual_quantity < request.quantity else 0
            })
            
            # 更新实际数量用于时间计算
            request.quantity = actual_quantity
        
        else:  # transfer
            # 库内调度：根据物料类型筛选
            source_wh = int(request.sourceWarehouse)
            target_wh = int(request.targetWarehouse)
            
            # 源仓库出库
            source_matrix = warehouse_matrices[source_wh]
            source_details = warehouse_shelf_details[source_wh]
            source_occupied = []
            for i in range(6):
                for j in range(4):
                    if source_matrix[i][j] > 0:
                        shelf_name = f"{rows[i]}{j+1}"
                        shelf_detail = source_details.get(shelf_name, {})
                        # 只选择匹配物料类型的货架
                        if shelf_detail.get('material') == request.materialType:
                            source_occupied.append({
                                'name': shelf_name,
                                'weight': float(source_matrix[i][j]),
                                'row': i,
                                'col': j
                            })
            
            if not source_occupied:
                raise HTTPException(status_code=400, detail=f"{source_wh}号仓库没有{material_names[request.materialType]}可调度")
            
            # 计算源仓库该物料的总可用重量
            total_available = sum(s['weight'] for s in source_occupied)
            
            # 如果请求的数量超过可用数量，使用最大可用数量
            actual_quantity = min(request.quantity, total_available)
            if actual_quantity < request.quantity:
                logger.warning(f"{source_wh}号仓库{material_names[request.materialType]}不足，请求{request.quantity}吨，实际可调度{actual_quantity}吨")
            
            # 源仓库分配
            remaining = actual_quantity
            source_allocated = []
            for shelf in source_occupied:
                if remaining <= 0:
                    break
                alloc_weight = min(remaining, shelf['weight'])
                source_allocated.append({
                    'name': shelf['name'],
                    'weight': round(alloc_weight, 2)
                })
                remaining -= alloc_weight
            
            shelf_allocations.append({
                'warehouse': f'{source_wh}号仓库（源）',
                'shelves': [s['name'] for s in source_allocated],
                'weights': [s['weight'] for s in source_allocated],
                'totalWeight': actual_quantity
            })
            
            # 目标仓库入库
            _, target_available = find_best_warehouse_for_inbound(actual_quantity, target_wh)
            if not target_available:
                raise HTTPException(status_code=400, detail=f"{target_wh}号仓库容量不足")
            
            target_allocated = []
            remaining = actual_quantity
            for shelf in target_available:
                if remaining <= 0:
                    break
                alloc_weight = min(remaining, shelf['capacity'])
                target_allocated.append({
                    'name': shelf['name'],
                    'weight': round(alloc_weight, 2)
                })
                remaining -= alloc_weight
            
            shelf_allocations.append({
                'warehouse': f'{target_wh}号仓库（目标）',
                'shelves': [s['name'] for s in target_allocated],
                'weights': [s['weight'] for s in target_allocated],
                'totalWeight': actual_quantity
            })
            
            selected_warehouse = target_wh
            
            # 更新实际数量用于时间计算
            request.quantity = actual_quantity
        
        # 计算预计时间（根据物料类型和重量）
        time_factor = material_time_factors.get(request.materialType, 3.5)
        base_time = request.quantity * time_factor
        
        # 根据操作类型调整时间
        if request.operationType == 'transfer':
            if request.sourceWarehouse != request.targetWarehouse:
                base_time *= 1.6  # 跨库调度增加60%时间
            else:
                base_time *= 1.2  # 同库调度增加20%时间
        elif request.operationType == 'outbound':
            base_time *= 1.15  # 出库增加15%时间（需要称重复核）
        
        # 格式化时间显示（小时和分钟）
        hours = int(base_time // 60)
        minutes = int(base_time % 60)
        if hours > 0:
            estimated_time = f"{hours}小时{minutes}分钟" if minutes > 0 else f"{hours}小时"
        else:
            estimated_time = f"{minutes}分钟"
        
        result = {
            'status': 'success',
            'operationType': request.operationType,
            'materialType': request.materialType,
            'materialName': material_names[request.materialType],
            'quantity': request.quantity,
            'selectedWarehouse': selected_warehouse,
            'shelfAllocations': shelf_allocations,
            'estimatedTime': estimated_time
        }
        
        logger.info(f"调度计算完成: {request.operationType} - {material_names[request.materialType]} - {request.quantity}吨 - 仓库{selected_warehouse}")
        
        return {
            'success': True,
            'result': result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"调度计算失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"调度计算失败: {str(e)}")


class ExecuteRequest(BaseModel):
    """执行调度请求"""
    operationType: str
    materialType: str
    materialName: str
    quantity: float
    shelfAllocations: List[dict]

@router.post("/execute")
async def execute_schedule(request: ExecuteRequest):
    """
    执行调度操作，更新仓库矩阵和货架详细信息
    
    Args:
        request: 执行请求，包含调度结果
        
    Returns:
        dict: 执行结果
    """
    try:
        from datetime import datetime
        
        rows_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for allocation in request.shelfAllocations:
            # 解析仓库ID
            warehouse_str = allocation['warehouse']
            if '（源）' in warehouse_str or '（目标）' in warehouse_str:
                wh_id = int(warehouse_str[0])
                is_source = '（源）' in warehouse_str
            else:
                wh_id = int(warehouse_str[0])
                is_source = False
            
            matrix = warehouse_matrices[wh_id]
            details = warehouse_shelf_details[wh_id]
            shelves = allocation['shelves']
            weights = allocation['weights']
            
            # 更新货架状态
            for shelf_name, weight in zip(shelves, weights):
                row = rows_map[shelf_name[0]]
                col = int(shelf_name[1:]) - 1
                
                if request.operationType == 'inbound':
                    # 入库：增加重量并记录详细信息
                    matrix[row][col] += weight
                    details[shelf_name] = {
                        'material': request.materialType,
                        'materialName': request.materialName,
                        'weight': float(matrix[row][col]),
                        'inboundTime': current_time
                    }
                elif request.operationType == 'outbound':
                    # 出库：减少重量
                    matrix[row][col] = max(0, matrix[row][col] - weight)
                    # 如果货架清空，删除详细信息
                    if matrix[row][col] == 0:
                        details.pop(shelf_name, None)
                    else:
                        # 更新重量
                        if shelf_name in details:
                            details[shelf_name]['weight'] = float(matrix[row][col])
                elif request.operationType == 'transfer':
                    # 调度：源仓库减少，目标仓库增加
                    if is_source:
                        matrix[row][col] = max(0, matrix[row][col] - weight)
                        if matrix[row][col] == 0:
                            details.pop(shelf_name, None)
                        elif shelf_name in details:
                            details[shelf_name]['weight'] = float(matrix[row][col])
                    else:
                        matrix[row][col] += weight
                        details[shelf_name] = {
                            'material': request.materialType,
                            'materialName': request.materialName,
                            'weight': float(matrix[row][col]),
                            'inboundTime': current_time
                        }
        
        logger.info(f"执行调度成功: {request.operationType} - {request.materialName} - {request.quantity}吨")
        
        # 保存数据到文件
        save_warehouse_data()
        
        return {
            'success': True,
            'message': '调度执行成功，仓库状态已更新'
        }
        
    except Exception as e:
        logger.error(f"执行调度失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"执行调度失败: {str(e)}")
