import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import joblib
import random
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class AttentionModule(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        attended = torch.sum(attention_weights * x, dim=1)
        return attended

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 输入转换
        self.input_fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 多层LSTM
        self.lstm1 = nn.LSTM(hidden_size, hidden_size, num_layers=2, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size*2, hidden_size, num_layers=2,
                           batch_first=True, dropout=dropout, bidirectional=True)
        
        # 注意力模块
        self.attention1 = AttentionModule(hidden_size*2)
        self.attention2 = AttentionModule(hidden_size*2)
        
        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size*2)
        self.layer_norm2 = nn.LayerNorm(hidden_size*2)
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*4, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, output_size)
        )
        
    def forward(self, x):
        # 输入转换
        x = self.input_fc(x)
        
        # 第一个LSTM层
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.layer_norm1(lstm1_out)
        attended1 = self.attention1(lstm1_out)
        
        # 第二个LSTM层
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.layer_norm2(lstm2_out)
        attended2 = self.attention2(lstm2_out)
        
        # 连接两个注意力输出
        combined = torch.cat([attended1, attended2], dim=1)
        
        # 输出层
        predictions = self.fc(combined)
        return predictions

def prepare_sequence_data(states, actions, window_size=20):
    """准备序列数据，包括计算统计特征"""
    # 计算差分特征
    states_diff = np.diff(states, axis=0)
    actions_diff = np.diff(actions, axis=0)
    
    # 计算滑动统计特征
    def compute_rolling_stats(data, window=5):
        means = np.array([np.mean(data[max(0, i-window):i+1], axis=0) for i in range(len(data))])
        stds = np.array([np.std(data[max(0, i-window):i+1], axis=0) for i in range(len(data))])
        return means, stds
    
    states_mean, states_std = compute_rolling_stats(states)
    actions_mean, actions_std = compute_rolling_stats(actions)
    
    # 组合所有特征
    features = np.concatenate([
        states,
        actions,
        np.pad(states_diff, ((0, 1), (0, 0)), mode='edge'),  # 补齐差分序列
        np.pad(actions_diff, ((0, 1), (0, 0)), mode='edge'),
        states_mean,
        states_std,
        actions_mean,
        actions_std
    ], axis=1)
    
    return features[window_size-1:]

def predict_next_states(model, states, actions, scaler_states, scaler_actions, scaler_y, device):
    """预测下一个状态"""
    # 准备序列数据
    sequence_data = prepare_sequence_data(states, actions)
    
    # 转换为tensor
    x = torch.FloatTensor(sequence_data).unsqueeze(0).to(device)  # 添加batch维度
    
    # 预测
    model.eval()
    with torch.no_grad():
        predictions = model(x)
        predictions = predictions.cpu().numpy()
    
    # 反标准化预测结果
    predictions_original = scaler_y.inverse_transform(predictions)
    
    return predictions_original[0]  # 返回预测结果

def main():
    # 定义列名
    action_columns = [
        '总物料', '燃料煤流量', '载煤风气流', '燃煤背压', 
        '喷枪背压', '抢位', '氧气流量', '喷枪注入空气流量'
    ]

    state_columns = [
        '氧气含量百分比', '炉底中部温度', '炉底外部温度',
        '炉底内温', '炉升高的温度', '炉压', '废气CO分析'
    ]
    
    # 加载数据
    print("加载数据...")
    data = pd.read_excel('2024years.xlsx')
    
    # 随机选择一段连续的数据
    window_size = 20
    start_idx = random.randint(0, len(data) - window_size - 1)
    sample_data = data.iloc[start_idx:start_idx + window_size + 1]
    
    # 提取状态和动作数据
    states = sample_data[state_columns].to_numpy()
    actions = sample_data[action_columns].to_numpy()
    
    # 加载标准化器
    print("加载标准化器...")
    scaler_states = joblib.load('lstm_scaler_states_v2.pkl')
    scaler_actions = joblib.load('lstm_scaler_actions_v2.pkl')
    scaler_y = joblib.load('lstm_scaler_y_v2.pkl')
    
    # 标准化数据
    states_scaled = scaler_states.transform(states)
    actions_scaled = scaler_actions.transform(actions)
    
    # 加载模型
    print("加载模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('lstm_model_v2.pth', map_location=device)
    model_config = checkpoint['model_config']
    
    model = LSTMPredictor(
        input_size=model_config['input_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        output_size=model_config['output_size'],
        dropout=model_config['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 进行预测
    print("\n进行预测...")
    predictions = predict_next_states(
        model, states_scaled, actions_scaled,
        scaler_states, scaler_actions, scaler_y,
        device
    )
    
    # 显示实际值和预测值
    actual_values = states[-1]
    print("\n预测结果对比：")
    print(f"{'状态名称':<15} {'实际值':>10} {'预测值':>10} {'误差':>10}")
    print("-" * 50)
    
    for i, col in enumerate(state_columns):
        error = abs(predictions[i] - actual_values[i])
        error_percent = (error / actual_values[i]) * 100
        print(f"{col:<15} {actual_values[i]:>10.2f} {predictions[i]:>10.2f} {error_percent:>9.2f}%")

if __name__ == "__main__":
    main() 