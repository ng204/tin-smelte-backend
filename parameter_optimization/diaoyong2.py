#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
锡冶炼强化学习模型推理脚本
功能：从 Excel 随机抽取当前状态 → 使用训练好的 Actor 模型输出动作建议
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import joblib
import random
from env import TinSmeltingEnv  # 你的环境文件：env.py


# ============================== Actor 网络 ==============================
class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=330, fc2_units=300):
        super(Actor, self).__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*self.uniform_init(self.fc1))
        self.fc2.weight.data.uniform_(*self.uniform_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def uniform_init(self, layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1.0 / np.sqrt(fan_in)
        return (-lim, lim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # 输出 [-1, 1]


# ============================== 模型加载 ==============================
def load_actor_model(model_path, state_dim, action_dim, device):
    model = Actor(state_dim, action_dim, seed=42)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    for name, p in model.named_parameters():
        if torch.isnan(p).any():
            print(f"[WARN] 参数 {name} 包含 NaN")
    return model


# ============================== 状态标准化 ==============================
def preprocess_sensor_data(sensor_data, scaler):
    sensor_data = np.array(sensor_data, dtype=np.float32)
    sensor_data = np.nan_to_num(sensor_data, nan=0.0)
    if sensor_data.ndim == 1:
        sensor_data = sensor_data.reshape(1, -1)

    if sensor_data.shape[1] != 8:
        raise ValueError(f"期望 8 维状态，实际 {sensor_data.shape[1]} 维")

    # 拼接 8 个零动作 → 14 维输入
    dummy_action = np.zeros((sensor_data.shape[0], 8), dtype=np.float32)
    model_input = np.hstack([sensor_data[:, :6], dummy_action])
    scaled = scaler.transform(model_input)

    # 构造 8 维标准化状态
    state = np.zeros((sensor_data.shape[0], 8), dtype=np.float32)
    state[:, :6] = scaled[:, :6]
    state[:, 6:] = sensor_data[:, 6:] / [5000.0, 10.0]  # CO / 5000, Sn / 10

    return state


# ============================== 动作反归一化 ==============================
def denormalize_action(norm_action, action_ranges):
    norm_action = np.clip(norm_action, -1, 1)
    norm_01 = (norm_action + 1) / 2.0
    real = np.empty_like(norm_action)
    for i, col in enumerate(action_ranges):
        mn, mx = action_ranges[col]['min'], action_ranges[col]['max']
        real[i] = mn if mx == mn else mn + (mx - mn) * norm_01[i]
    return real


# ============================== 推理函数 ==============================
def get_action(model, sensor_data, scaler, action_ranges, device):
    state = preprocess_sensor_data(sensor_data, scaler)
    tensor = torch.FloatTensor(state).to(device)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        act = model(tensor).cpu().numpy().flatten()
    return denormalize_action(act, action_ranges)


# ============================== 主函数 ==============================
def main():
    # ==================== 1. 清除旧种子 + 设置新种子 ====================
    np.random.seed(None)      # 清除 NumPy 旧状态
    random.seed(None)         # 清除 Python 随机状态

    # 安全生成 32 位随机种子（避免 int32 溢出）
    master_seed = random.randint(0, 2_000_000_000)  # < 2^31
    print(f"\n[INFO] 本次全局随机种子: {master_seed}")
    np.random.seed(master_seed)
    torch.manual_seed(master_seed)
    random.seed(master_seed)
    # =================================================================

    # ==================== 2. 配置 ====================
    MODEL_PATH = 'final_actor.pth'
    STATE_DIM = 8
    ACTION_DIM = 8
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 使用设备: {DEVICE}")

    # ==================== 3. 加载环境 ====================
    try:
        env = TinSmeltingEnv(data_path='2024years.xlsx')
        action_ranges = env.action_ranges
        scaler = env.scaler_X
        state_columns = env.state_columns
        print("[INFO] 环境加载成功")
    except Exception as e:
        print(f"[ERROR] 环境加载失败: {e}")
        return

    # 打印动作范围
    print("\n[INFO] 动作物理范围:")
    for col, r in action_ranges.items():
        print(f"  {col}: {r['min']:.2f} ~ {r['max']:.2f}")

    # ==================== 4. 加载 Actor 模型 ====================
    try:
        actor = load_actor_model(MODEL_PATH, STATE_DIM, ACTION_DIM, DEVICE)
        print("[INFO] Actor 模型加载成功")
    except Exception as e:
        print(f"[ERROR] 模型加载失败: {e}")
        return

    # ==================== 5. 随机抽取初始状态 ====================
    STATE_COLUMNS = [
        '氧气含量百分比', '炉底中部温度', '炉底外部温度',
        '炉底内温', '炉升高的温度', '炉压', '废气CO分析', 'Sn'
    ]

    try:
        df = pd.read_excel('2024years.xlsx')
        if df.empty:
            raise ValueError("Excel 文件为空")

        missing = [c for c in STATE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"缺少列: {missing}")

        # 计算归一化所需 min / max
        state_min = df[STATE_COLUMNS].min()
        state_max = df[STATE_COLUMNS].max()

        # 真正随机抽样（random_state=None 确保每次不同）
        row = df[STATE_COLUMNS].sample(n=1, random_state=None).iloc[0]
        sensor_data = row.values.astype(np.float32)

        # 归一化
        sensor_data = (row - state_min) / (state_max - state_min + 1e-8)
        sensor_data = sensor_data.values.astype(np.float32)

        print("\n[INFO] 随机抽取的当前状态:")
        for c, v in zip(STATE_COLUMNS, sensor_data):
            print(f"  {c}: {v:.2f}")

    except Exception as e:
        print(f"[WARN] Excel 读取失败: {e}")
        print("使用默认示例状态")
        sensor_data = np.array([20.0, 1200.0, 1100.0, 1150.0, 100.0, 50.0, 1400.0, 2.0])

    # ==================== 6. 生成动作建议 ====================
    action = get_action(actor, sensor_data, scaler, action_ranges, DEVICE)

    ACTION_NAMES = [
        '总物料', '燃料煤流量', '载煤风气流', '燃煤背压',
        '喷枪背压', '抢位', '氧气流量', '喷枪注入空气流量'
    ]

    print("\n" + "="*50)
    print("           动作控制建议")
    print("="*50)
    for name, val in zip(ACTION_NAMES, action):
        print(f"{name:>12}: {val:8.2f}")
    print("="*50)
    print("推理完成！\n")


if __name__ == '__main__':
    main()