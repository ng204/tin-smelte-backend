from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import subprocess
import sys
import re
from typing import Dict, Tuple
from datetime import datetime
from pathlib import Path
import threading

# 轻量推理所需
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from parameter_optimization.env import TinSmeltingEnv

router = APIRouter(prefix="/param-opt", tags=["参数优化"])

# 缓存最近一次脚本输出
_last_state: Dict[str, float] | None = None
_last_actions: Dict[str, float] | None = None
_last_updated: str | None = None

# 预加载资源（内存推理）
_ready: bool = False
_lock = threading.Lock()
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_env: TinSmeltingEnv | None = None
_scaler = None
_action_ranges: Dict[str, Dict[str, float]] | None = None
_STATE_COLUMNS = [
    '氧气含量百分比', '炉底中部温度', '炉底外部温度',
    '炉底内温', '炉升高的温度', '炉压', '废气CO分析', 'Sn'
]
_ACTION_NAMES = [
    '总物料', '燃料煤流量', '载煤风气流', '燃煤背压',
    '喷枪背压', '抢位', '氧气流量', '喷枪注入空气流量'
]
_actor: nn.Module | None = None
_df: pd.DataFrame | None = None

# 中文字段名称
STATE_LABELS = [
    '氧气含量百分比', '炉底中部温度', '炉底外部温度',
    '炉底内温', '炉升高的温度', '炉压', '废气CO分析', 'Sn'
]
ACTION_LABELS = [
    '总物料', '燃料煤流量', '载煤风气流', '燃煤背压',
    '喷枪背压', '抢位', '氧气流量', '喷枪注入空气流量'
]

PARAM_DIR = Path(__file__).resolve().parents[1] / 'parameter_optimization'
SCRIPT_PATH = PARAM_DIR / 'diaoyong2.py'
MODEL_PATH = PARAM_DIR / 'final_actor.pth'
EXCEL_PATH = PARAM_DIR / '2024years.xlsx'

STATE_PATTERN = re.compile(r"^\s*({})\s*[:：]\s*([+-]?\d+(?:\.\d+)?)".format("|".join(map(re.escape, STATE_LABELS))), re.M)
ACTION_PATTERN = re.compile(r"^\s*({})\s*[:：]\s*([+-]?\d+(?:\.\d+)?)".format("|".join(map(re.escape, ACTION_LABELS))), re.M)


def _run_script() -> Tuple[Dict[str, float], Dict[str, float], str]:
    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(f"diaoyong2.py 不存在: {SCRIPT_PATH}")
    try:
        proc = subprocess.run(
            [sys.executable, '-u', str(SCRIPT_PATH)],
            cwd=str(PARAM_DIR),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except Exception as e:
        raise RuntimeError(f"启动脚本失败: {e}")

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        # 即使非0，也尝试解析 stdout
        pass

    # 解析状态
    state_matches = STATE_PATTERN.findall(stdout)
    state: Dict[str, float] = {}
    for label, val in state_matches:
        try:
            state[label] = float(val)
        except Exception:
            continue

    # 解析动作
    action_matches = ACTION_PATTERN.findall(stdout)
    actions: Dict[str, float] = {}
    for label, val in action_matches:
        try:
            actions[label] = float(val)
        except Exception:
            continue

    if not state:
        # 将 stderr 附带在错误中便于排查
        raise HTTPException(status_code=500, detail=f"未能解析到状态数据。stderr={stderr[:500]}")

    ts = datetime.utcnow().isoformat() + 'Z'
    return state, actions, ts


def bootstrap_on_startup() -> None:
    global _last_state, _last_actions, _last_updated, _ready
    try:
        _init_in_memory()
        s, a, ts = _refresh_in_memory()
        _last_state, _last_actions, _last_updated = s, a, ts
        _ready = True
        print("[param-opt] in-memory bootstrap completed")
    except Exception as e:
        print(f"[param-opt] in-memory bootstrap failed, fallback to script: {e}")
        try:
            _last_state, _last_actions, _last_updated = _run_script()
            _ready = False
        except Exception as e2:
            # 启动失败不阻断服务，等待后续刷新
            _last_state = None
            _last_actions = None
            _last_updated = None
            print(f"[param-opt] 启动时获取失败: {e2}")


# ==================== 内存推理实现 ====================
class Actor(nn.Module):
    def __init__(self, state_size: int, action_size: int, seed: int = 42, fc1_units: int = 330, fc2_units: int = 300):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self._reset_parameters()

    def _reset_parameters(self):
        def uni_init(layer):
            fan_in = layer.weight.data.size()[0]
            lim = 1.0 / np.sqrt(fan_in)
            return (-lim, lim)
        self.fc1.weight.data.uniform_(*uni_init(self.fc1))
        self.fc2.weight.data.uniform_(*uni_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


def _load_actor_model_safely(model_path: Path, state_dim: int, action_dim: int, device) -> nn.Module:
    model = Actor(state_dim, action_dim, seed=42)
    try:
        state_dict = torch.load(str(model_path), map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _init_in_memory():
    """预加载环境、模型和 Excel 数据，显著降低刷新时延。"""
    global _env, _scaler, _action_ranges, _actor, _df
    with _lock:
        if _env is not None and _actor is not None and _df is not None:
            return
        env = TinSmeltingEnv(data_path=str(EXCEL_PATH))
        _env = env
        _scaler = env.scaler_X
        _action_ranges = env.action_ranges
        _actor = _load_actor_model_safely(MODEL_PATH, state_dim=8, action_dim=8, device=_DEVICE)
        # 提前将 Excel 载入内存
        _df = pd.read_excel(str(EXCEL_PATH))
        if _df.empty:
            raise RuntimeError("Excel 数据为空")


def _preprocess_sensor_data(sensor_data: np.ndarray) -> np.ndarray:
    sensor_data = np.array(sensor_data, dtype=np.float32)
    sensor_data = np.nan_to_num(sensor_data, nan=0.0)
    if sensor_data.ndim == 1:
        sensor_data = sensor_data.reshape(1, -1)
    if sensor_data.shape[1] != 8:
        raise ValueError(f"期望 8 维状态，实际 {sensor_data.shape[1]} 维")
    # 拼接 8 个零动作 → 14 维输入
    dummy_action = np.zeros((sensor_data.shape[0], 8), dtype=np.float32)
    model_input = np.hstack([sensor_data[:, :6], dummy_action])
    scaled = _scaler.transform(model_input)
    # 构造 8 维标准化状态
    state = np.zeros((sensor_data.shape[0], 8), dtype=np.float32)
    state[:, :6] = scaled[:, :6]
    state[:, 6:] = sensor_data[:, 6:] / [5000.0, 10.0]  # CO / 5000, Sn / 10
    return state


def _denormalize_action(norm_action: np.ndarray) -> np.ndarray:
    norm_action = np.clip(norm_action, -1, 1)
    norm_01 = (norm_action + 1) / 2.0
    real = np.empty_like(norm_action)
    for i, col in enumerate(_ACTION_NAMES):
        rng = _action_ranges[col]
        mn, mx = rng['min'], rng['max']
        real[i] = mn if mx == mn else mn + (mx - mn) * norm_01[i]
    return real


def _refresh_in_memory() -> Tuple[Dict[str, float], Dict[str, float], str]:
    if _env is None or _actor is None or _df is None:
        raise RuntimeError("内存推理尚未初始化")
    with _lock:
        # 随机抽样一行状态
        row = _df[_STATE_COLUMNS].sample(n=1, random_state=None).iloc[0]
        sensor_data = row.values.astype(np.float32)
        # 生成动作
        state = _preprocess_sensor_data(sensor_data)
        tensor = torch.FloatTensor(state).to(_DEVICE)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            act = _actor(tensor).cpu().numpy().flatten()
        real_act = _denormalize_action(act)

        # 构造返回 - 使用原始值而不是归一化值
        state_dict = {c: float(v) for c, v in zip(_STATE_COLUMNS, sensor_data)}
        actions_dict = {name: float(val) for name, val in zip(_ACTION_NAMES, real_act)}
        ts = datetime.utcnow().isoformat() + 'Z'
        return state_dict, actions_dict, ts


class StateResponse(BaseModel):
    success: bool
    state: Dict[str, float]
    updated_at: str | None = None


class ActionsResponse(BaseModel):
    success: bool
    actions: Dict[str, float]
    updated_at: str | None = None


@router.get('/state', response_model=StateResponse)
def get_state():
    global _last_state, _last_actions, _last_updated
    if _last_state is None:
        # 无缓存则立即生成一次
        try:
            if _ready:
                _last_state, _last_actions, _last_updated = _refresh_in_memory()
            else:
                _last_state, _last_actions, _last_updated = _run_script()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return StateResponse(success=True, state=_last_state or {}, updated_at=_last_updated)


@router.post('/optimize', response_model=ActionsResponse)
def optimize():
    global _last_state, _last_actions, _last_updated
    # 若无缓存（例如首次请求），运行一次保证生成动作
    if _last_actions is None:
        try:
            if _ready:
                _last_state, _last_actions, _last_updated = _refresh_in_memory()
            else:
                _last_state, _last_actions, _last_updated = _run_script()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    # 仅返回动作，前端在点击优化之前保持 0
    actions = {label: float(_last_actions.get(label, 0.0)) for label in ACTION_LABELS}
    return ActionsResponse(success=True, actions=actions, updated_at=_last_updated)


@router.post('/refresh', response_model=StateResponse)
def refresh():
    global _last_state, _last_actions, _last_updated
    try:
        if _ready:
            _last_state, _last_actions, _last_updated = _refresh_in_memory()
        else:
            _last_state, _last_actions, _last_updated = _run_script()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # 刷新后仅返回状态，前端需将优化结果清零
    return StateResponse(success=True, state=_last_state or {}, updated_at=_last_updated)
