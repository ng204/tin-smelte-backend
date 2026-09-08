import gym
import numpy as np
from gym import spaces
import joblib
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import xgboost as xgb
try:
    from .predict_states import LSTMPredictor, prepare_sequence_data
except ImportError:
    from predict_states import LSTMPredictor, prepare_sequence_data
from torch import nn
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class StatePredictor:
    def __init__(self, model_path, scaler_X_path, scaler_y_path):
        # 加载模型和标准化器
        self.model = joblib.load(model_path)
        self.scaler_X = joblib.load(scaler_X_path)
        self.scaler_y = joblib.load(scaler_y_path)
        
        # 定义状态和动作列
        self.state_columns = [
            '氧气含量百分比', '炉底中部温度', '炉底外部温度',
            '炉底内温', '炉升高的温度', '炉压'
        ]
        self.action_columns = [
            '总物料', '燃料煤流量', '载煤风气流', '燃煤背压',
            '喷枪背压', '抢位', '氧气流量', '喷枪注入空气流量'
        ]

    def predict(self, current_state, current_action):
        """预测下一个状态"""
        # 准备输入数据（8个动作 + 6个状态）
        state_action = np.concatenate([current_action, current_state])
        
        # 标准化输入
        state_action_scaled = self.scaler_X.transform(state_action.reshape(1, -1))
        
        # 预测状态
        next_state_scaled = self.model.predict(state_action_scaled)
        
        # 逆标准化
        next_state = self.scaler_y.inverse_transform(next_state_scaled)
        return next_state.flatten()

class GRUPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)  # [batch, seq_len, hidden_size]
        out = out[:, -1, :]  # 取最后一个时间步输出
        return self.fc(out)  # [batch, 1]

class StateUpdater:
    def __init__(self, state_predictor_path, co_predictor_path, scaler_path):
        # 加载状态预测模型
        self.state_predictor = StatePredictor(
            model_path='state_predictor2.pkl',
            scaler_X_path='state_scaler_X2.pkl',
            scaler_y_path='state_scaler_y2.pkl'
        )
        
        # 加载CO预测模型
        self.co_predictor = GRUPredictor(input_size=4)  # 4个特征
        self.co_predictor.load_state_dict(torch.load(co_predictor_path))
        self.co_predictor.eval()
        
        # 加载CO预测的标准化器
        self.co_scaler = joblib.load(scaler_path)
        
        # 定义CO预测使用的特征
        self.co_features = ['燃料煤流量', '氧气流量', '氧气含量百分比', '炉压']

    def predict_next_state(self, current_state, current_action):
        """预测下一个状态"""
        return self.state_predictor.predict(current_state, current_action)

    def predict_co(self, current_state, action):
        """预测CO浓度"""
        try:
            # 组装5个特征，最后一位目标值用0占位
            co_input = np.array([
                current_state[0],   # 氧气含量百分比
                current_state[5],   # 炉压
                action[6],          # 氧气流量
                action[1],          # 燃料煤流量
                0                   # 目标占位
            ])
            # 标准化
            co_input_scaled = self.co_scaler.transform(co_input.reshape(1, -1))
            # 只取前4个输入特征送入模型
            input_tensor = torch.FloatTensor(co_input_scaled[0, :4]).reshape(1, 1, 4)
            # 预测
            with torch.no_grad():
                co_scaled = self.co_predictor(input_tensor).numpy()
            # 逆标准化，组装5个特征，最后一位用预测值
            dummy = np.zeros((1, 5))
            dummy[0, :4] = co_input_scaled[0, :4]
            dummy[0, 4] = co_scaled[0, 0]
            co = self.co_scaler.inverse_transform(dummy)[0, 4]
            return float(co)
        except Exception as e:
            print(f"预测CO时发生错误: {str(e)}")
            print(f"当前状态: {current_state}")
            print(f"执行动作: {action}")
            return float(current_state[6])  # 如果预测失败，返回当前的CO值

    def update_state(self, current_state, action):
        """更新状态"""
        try:
            # 预测6个状态变量
            predicted_states = self.predict_states(current_state, action)
            
            # 预测CO浓度
            co = self.predict_co(current_state, action)
            
            # 预测Sn
            predicted_sn = self.predict_sn(current_state, action)
            
            # 构建新的状态
            next_state = np.array([
                *predicted_states,       # 6个预测状态
                co,                      # 预测的CO
                predicted_sn             # 预测的Sn
            ])
            
            # 打印调试信息
            print("\n=== 状态更新 ===")
            print(f"当前状态: {current_state}")
            print(f"执行动作: {action}")
            print(f"预测下一状态: {next_state}")
            print("===============\n")
            
            return next_state
            
        except Exception as e:
            print(f"状态更新过程中出现错误: {str(e)}")
            return current_state  # 如果出错，返回当前状态

    def predict_states(self, current_state, action):
        """
        使用训练好的MLP模型预测下一时刻的状态
        Args:
            current_state: 当前状态 [氧气含量百分比, ..., Sn]
            action: 当前动作 [总物料, ...]
        Returns:
            next_state: 预测的下一时刻状态 [氧气含量百分比, ..., 炉压]
        """
        try:
            # 只取需要的6个数值状态（不包括CO和Sn）
            current_state_numeric = current_state[:6]
            
            # 准备模型输入：将当前状态和动作拼接
            model_input = np.hstack([current_state_numeric, action])
            model_input = np.nan_to_num(model_input, nan=0.0)  # 处理可能的NaN值
            
            # 确保输入数据的形状正确
            if len(model_input.shape) == 1:
                model_input = model_input.reshape(1, -1)
            
            # 标准化输入数据
            model_input_scaled = self.state_predictor.scaler_X.transform(model_input)
            
            # 预测下一状态
            predicted_next_state_scaled = self.state_predictor.model.predict(model_input_scaled)
            
            # 反标准化预测结果
            next_state = self.state_predictor.scaler_y.inverse_transform(predicted_next_state_scaled.reshape(1, -1))[0]
            
            # 确保预测结果在合理范围内
            next_state = np.clip(next_state, self.state_predictor.state_min, self.state_predictor.state_max)
            
            return next_state.tolist()
            
        except Exception as e:
            print(f"预测过程中出现错误: {str(e)}")
            print(f"当前状态: {current_state}")
            print(f"执行动作: {action}")
            return current_state[:6].tolist()

    def create_features(self, current_state, action):
        """创建XGBoost模型所需的特征，生成所有特征后严格按训练时特征名顺序返回"""
        # 1. 还原十个基本特征
        feature_columns = [
            '氧气流量', '炉底中部温度', '燃料煤流量', '喷枪背压',
            '载煤风气流', '燃煤背压', '炉压', '废气CO分析',
            '抢位', '炉升高的温度'
        ]
        feature_values = {
            '氧气流量': action[6],
            '炉底中部温度': current_state[1],
            '燃料煤流量': action[1],
            '喷枪背压': action[4],
            '载煤风气流': action[2],
            '燃煤背压': action[3],
            '炉压': current_state[5],
            '废气CO分析': current_state[6],
            '抢位': action[5],
            '炉升高的温度': current_state[4]
        }
        df = pd.DataFrame({k: [v] for k, v in feature_values.items()})

        # 2. 滚动统计特征（只有一条数据，全部用当前值或0占位）
        windows = [3, 5]
        for feature in feature_columns:
            for window in windows:
                df[f'{feature}_rolling_mean_{window}'] = df[feature]
                df[f'{feature}_rolling_std_{window}'] = 0.0

        # 3. 关键特征组合
        key_features = ['废气CO分析', '炉压', '炉底中部温度']
        for i, feat1 in enumerate(key_features):
            for feat2 in key_features[i+1:]:
                df[f'{feat1}_{feat2}_ratio'] = df[feat1] / (df[feat2] + 1e-10)

        # 4. 补齐所有训练时的特征，没有的补0
        for col in self.state_predictor.state_columns:
            if col not in df.columns:
                df[col] = 0.0
        # 5. 严格按训练时顺序
        features = df[self.state_predictor.state_columns]
        return features

    def predict_sn(self, current_state, action):
        """预测Sn含量"""
        try:
            # 创建特征（已保证和训练时一致）
            features = self.create_features(current_state, action)
            # 标准化特征
            features_scaled = self.state_predictor.sn_scaler_X.transform(features)
            # 转换为DMatrix格式
            dmatrix = xgb.DMatrix(features_scaled)
            # 预测Sn
            predicted_sn_scaled = self.state_predictor.xgb_model.predict(dmatrix)
            # 反标准化预测结果
            predicted_sn = self.state_predictor.sn_scaler_y.inverse_transform(
                predicted_sn_scaled.reshape(-1, 1)
            ).ravel()[0]
            return float(predicted_sn)
        except Exception as e:
            print(f"预测Sn时发生错误: {str(e)}")
            return float(current_state[7])  # 如果预测失败，返回当前的Sn值

    def calculate_reward(self, state):
        """
        计算奖励函数
        目标：CO在1400附近奖励最大，Sn越低越好，极端情况有惩罚
        """
        try:
            co = float(state[6])
            sn = float(state[7])

            # CO奖励：高斯型，中心1400，标准差400
            co_reward = np.exp(-((co - 1400) / 400) ** 2)
            # Sn奖励：2最优，递减
            sn_reward = np.exp(-1.5 * (sn - 2))

            # 权重可调
            reward = 0.7 * co_reward + 0.3 * sn_reward

            # 极端惩罚
            if co > 5000 or co < 0:
                reward -= 1
            if sn > 5 or sn < 1:
                reward -= 0.5

            # 奖励裁剪
            reward = np.clip(reward, -2, 2)

            # 打印调试信息
            print(f'CO: {co:.2f}, CO奖励: {co_reward:.2f} | Sn: {sn:.2f}, Sn奖励: {sn_reward:.2f} | 总奖励: {reward:.2f}')
            return reward
        except Exception as e:
            print(f"奖励计算过程中出现错误: {str(e)}")
            print(f"当前状态: {state}")
            return -1.0  # 如果出错，返回负奖励

class TinSmeltingEnv(gym.Env):
    def __init__(self, data_path):
        super(TinSmeltingEnv, self).__init__()

        # 读取数据
        self.data = pd.read_excel(data_path)
        print(f"加载数据集，共 {len(self.data)} 条记录")
        
        # 恢复对Sn列的填充，防止后续actor使用时出错
        self.data['Sn'] = self.data['Sn'].fillna(method='ffill')  # 向前填充
        self.data['Sn'] = self.data['Sn'].fillna(method='bfill')  # 向后填充
        self.data['Sn'] = self.data['Sn'].fillna(self.data['Sn'].mean())  # 如果还有空值，用均值填充
        
        # 定义动作和状态列
        self.action_columns = [
            '总物料', '燃料煤流量', '载煤风气流', '燃煤背压', 
            '喷枪背压', '抢位', '氧气流量', '喷枪注入空气流量'
        ]
        
        self.state_columns = [
            '氧气含量百分比', '炉底中部温度', '炉底外部温度',
            '炉底内温', '炉升高的温度', '炉压', '废气CO分析', 'Sn'
        ]
        
        # 获取动作的实际范围
        self.action_ranges = {
            col: {'min': self.data[col].min(), 'max': self.data[col].max()}
            for col in self.action_columns
        }
        print("\n动作范围:")
        for col, range_dict in self.action_ranges.items():
            print(f"{col}: {range_dict['min']:.2f} ~ {range_dict['max']:.2f}")
        
        # 加载状态预测模型和标准化器
        try:
            self.scaler_X = joblib.load(str(BASE_DIR / 'state_scaler_X2.pkl'))
            self.scaler_y = joblib.load(str(BASE_DIR / 'state_scaler_y2.pkl'))
            self.state_predictor = joblib.load(str(BASE_DIR / 'state_predictor2.pkl'))
            print("\n成功加载状态预测模型和标准化器")
        except Exception as e:
            print(f"加载状态预测模型或标准化器失败: {str(e)}")
            raise e
        
        # 加载CO预测模型和标准化器
        try:
            # 加载模型配置
            checkpoint = torch.load(str(BASE_DIR / 'saved_models' / 'co_predictor.pth'))
            self.co_predictor = GRUPredictor(
                input_size=checkpoint['input_size'],
                hidden_size=checkpoint['model_config']['hidden_size'],
                num_layers=checkpoint['model_config']['num_layers'],
                dropout=checkpoint['model_config']['dropout']
            )
            self.co_predictor.load_state_dict(checkpoint['model_state_dict'])
            self.co_predictor.eval()
            self.co_scaler = joblib.load(str(BASE_DIR / 'saved_models' / '2scaler.pkl'))
            print("成功加载CO预测模型和标准化器")
        except Exception as e:
            print(f"加载CO预测模型或标准化器失败: {str(e)}")
            raise e
        
        # 加载XGBoost模型和标准化器
        try:
            self.xgb_model = xgb.Booster()
            self.xgb_model.load_model(str(BASE_DIR / 'xgboost_model.json'))
            self.sn_scaler_X = joblib.load(str(BASE_DIR / 'scaler_X.pkl'))
            self.sn_scaler_y = joblib.load(str(BASE_DIR / 'scaler_y.pkl'))
            print("成功加载XGBoost模型和标准化器")
        except Exception as e:
            print(f"加载XGBoost模型或标准化器失败: {str(e)}")
            raise e
        
        # 定义动作空间和状态空间
        self.action_space = len(self.action_columns)
        self.state_space = len(self.state_columns)
        
        # 定义动作范围
        self.action_low = np.array([self.action_ranges[col]['min'] for col in self.action_columns])
        self.action_high = np.array([self.action_ranges[col]['max'] for col in self.action_columns])
        
        # 定义状态范围，只取前6个状态变量（不包括CO和Sn）
        self.state_min = self.data[self.state_columns[:6]].min().values
        self.state_max = self.data[self.state_columns[:6]].max().values
        
        # 初始化历史数据存储
        self.history_states = []
        self.history_actions = []
        self.history_length = 5  # 保存最近5条记录
        
        # 初始化当前状态
        self.current_state = None
        self.current_step = 0

        # 加载Sn特征名列表，保证特征顺序和数量一致
        self.sn_feature_names = joblib.load(str(BASE_DIR / 'sn_feature_names.pkl'))

    def reset(self):
        """
        重置环境到初始状态
        从历史数据中随机选择一行作为起始状态
        """
        # 随机选择一个起始点
        self.current_step = np.random.randint(0, len(self.data))
        print(f"重置环境，从第 {self.current_step} 条记录开始")
        
        # 获取初始状态
        initial_state = self.data.iloc[self.current_step][self.state_columns].values
        self.current_state = np.array([float(x) for x in initial_state])
        
        return self.current_state

    def step(self, action):
        """
        执行动作并返回新的状态、奖励和是否结束
        Args:
            action: actor网络输出的动作（范围在-1到1之间）
        Returns:
            next_state: 下一个状态
            reward: 奖励值
            done: 是否结束
        """
        # 将归一化的动作（-1到1）转换为实际范围
        real_action = self.denormalize_action(action)
        
        # 预测下一个状态
        next_state = self.update_state(self.current_state, real_action)
        
        # 计算奖励
        reward = self.calculate_reward(next_state)
        
        # 更新当前状态
        self.current_state = next_state
        
        # 更新步数
        self.current_step += 1
        
        # 检查是否结束
        done = False
        if self.current_step >= len(self.data) - 1:  # 到达数据末尾
            done = True
        elif np.abs(reward) > 10.0:  # 奖励过大或过小
            done = True
        elif np.any(np.isnan(next_state)):  # 状态出现NaN
            done = True
            reward = -10.0  # 对NaN状态进行惩罚
        
        return next_state, reward, done

    def denormalize_action(self, normalized_action):
        """
        将归一化的动作（-1到1）转换为实际范围
        Args:
            normalized_action: 归一化后的动作值（-1到1）
        Returns:
            real_action: 实际范围的动作值
        """
        # 确保动作在-1到1范围内
        normalized_action = np.clip(normalized_action, -1, 1)
        
        # 将-1到1映射到0到1
        normalized_action_0_1 = (normalized_action + 1) / 2
        
        # 映射到实际范围
        real_action = np.zeros_like(normalized_action)
        for i, col in enumerate(self.action_columns):
            min_val = self.action_ranges[col]['min']
            max_val = self.action_ranges[col]['max']
            real_action[i] = min_val + (max_val - min_val) * normalized_action_0_1[i]
        
        return real_action

    def predict_co(self, current_state, action):
        """预测CO浓度"""
        try:
            # 组装5个特征，最后一位目标值用0占位
            co_input = np.array([
                current_state[0],   # 氧气含量百分比
                current_state[5],   # 炉压
                action[6],          # 氧气流量
                action[1],          # 燃料煤流量
                0                   # 目标占位
            ])
            # 标准化
            co_input_scaled = self.co_scaler.transform(co_input.reshape(1, -1))
            # 只取前4个输入特征送入模型
            input_tensor = torch.FloatTensor(co_input_scaled[0, :4]).reshape(1, 1, 4)
            # 预测
            with torch.no_grad():
                co_scaled = self.co_predictor(input_tensor).numpy()
            # 逆标准化，组装5个特征，最后一位用预测值
            dummy = np.zeros((1, 5))
            dummy[0, :4] = co_input_scaled[0, :4]
            dummy[0, 4] = co_scaled[0, 0]
            co = self.co_scaler.inverse_transform(dummy)[0, 4]
            return float(co)
        except Exception as e:
            print(f"预测CO时发生错误: {str(e)}")
            print(f"当前状态: {current_state}")
            print(f"执行动作: {action}")
            return float(current_state[6])  # 如果预测失败，返回当前的CO值

    def update_state(self, current_state, action):
        """更新状态"""
        try:
            # 预测6个状态变量
            predicted_states = self.predict_states(current_state, action)
            
            # 预测CO浓度
            co = self.predict_co(current_state, action)
            
            # 预测Sn
            predicted_sn = self.predict_sn(current_state, action)
            
            # 构建新的状态
            next_state = np.array([
                *predicted_states,       # 6个预测状态
                co,                      # 预测的CO
                predicted_sn             # 预测的Sn
            ])
            
            # 打印调试信息
            print("\n=== 状态更新 ===")
            print(f"当前状态: {current_state}")
            print(f"执行动作: {action}")
            print(f"预测下一状态: {next_state}")
            print("===============\n")
            
            return next_state
            
        except Exception as e:
            print(f"状态更新过程中出现错误: {str(e)}")
            return current_state  # 如果出错，返回当前状态

    def predict_states(self, current_state, action):
        """
        使用训练好的MLP模型预测下一时刻的状态
        Args:
            current_state: 当前状态 [氧气含量百分比, ..., Sn]
            action: 当前动作 [总物料, ...]
        Returns:
            next_state: 预测的下一时刻状态 [氧气含量百分比, ..., 炉压]
        """
        try:
            # 只取需要的6个数值状态（不包括CO和Sn）
            current_state_numeric = current_state[:6]
            
            # 准备模型输入：将当前状态和动作拼接
            model_input = np.hstack([current_state_numeric, action])
            model_input = np.nan_to_num(model_input, nan=0.0)  # 处理可能的NaN值
            
            # 确保输入数据的形状正确
            if len(model_input.shape) == 1:
                model_input = model_input.reshape(1, -1)
            
            # 标准化输入数据
            model_input_scaled = self.scaler_X.transform(model_input)
            
            # 预测下一状态
            predicted_next_state_scaled = self.state_predictor.predict(model_input_scaled)
            
            # 反标准化预测结果
            next_state = self.scaler_y.inverse_transform(predicted_next_state_scaled.reshape(1, -1))[0]
            
            # 确保预测结果在合理范围内
            next_state = np.clip(next_state, self.state_min, self.state_max)
            
            return next_state.tolist()
            
        except Exception as e:
            print(f"预测过程中出现错误: {str(e)}")
            print(f"当前状态: {current_state}")
            print(f"执行动作: {action}")
            return current_state[:6].tolist()

    def create_features(self, current_state, action):
        """创建XGBoost模型所需的特征，生成所有特征后严格按训练时特征名顺序返回"""
        # 1. 还原十个基本特征
        feature_columns = [
            '氧气流量', '炉底中部温度', '燃料煤流量', '喷枪背压',
            '载煤风气流', '燃煤背压', '炉压', '废气CO分析',
            '抢位', '炉升高的温度'
        ]
        feature_values = {
            '氧气流量': action[6],
            '炉底中部温度': current_state[1],
            '燃料煤流量': action[1],
            '喷枪背压': action[4],
            '载煤风气流': action[2],
            '燃煤背压': action[3],
            '炉压': current_state[5],
            '废气CO分析': current_state[6],
            '抢位': action[5],
            '炉升高的温度': current_state[4]
        }
        df = pd.DataFrame({k: [v] for k, v in feature_values.items()})

        # 2. 滚动统计特征（只有一条数据，全部用当前值或0占位）
        windows = [3, 5]
        for feature in feature_columns:
            for window in windows:
                df[f'{feature}_rolling_mean_{window}'] = df[feature]
                df[f'{feature}_rolling_std_{window}'] = 0.0

        # 3. 关键特征组合
        key_features = ['废气CO分析', '炉压', '炉底中部温度']
        for i, feat1 in enumerate(key_features):
            for feat2 in key_features[i+1:]:
                df[f'{feat1}_{feat2}_ratio'] = df[feat1] / (df[feat2] + 1e-10)

        # 4. 补齐所有训练时的特征，没有的补0
        for col in self.sn_feature_names:
            if col not in df.columns:
                df[col] = 0.0
        # 5. 严格按训练时顺序
        features = df[self.sn_feature_names]
        return features

    def predict_sn(self, current_state, action):
        """预测Sn含量"""
        try:
            # 创建特征（已保证和训练时一致）
            features = self.create_features(current_state, action)
            # 标准化特征
            features_scaled = self.sn_scaler_X.transform(features)
            # 转换为DMatrix格式
            dmatrix = xgb.DMatrix(features_scaled)
            # 预测Sn
            predicted_sn_scaled = self.xgb_model.predict(dmatrix)
            # 反标准化预测结果
            predicted_sn = self.sn_scaler_y.inverse_transform(
                predicted_sn_scaled.reshape(-1, 1)
            ).ravel()[0]
            return float(predicted_sn)
        except Exception as e:
            print(f"预测Sn时发生错误: {str(e)}")
            return float(current_state[7])  # 如果预测失败，返回当前的Sn值

    def calculate_reward(self, state):
        """
        改进后的奖励函数设计：
        - CO以1800为中心，高斯分布鼓励靠近，标准差扩大保证高值区分度
        - Sn越低奖励越高，指数衰减
        - 连续惩罚机制替代突变惩罚
        """
        try:
            co = float(state[6])
            sn = float(state[7])

            # CO奖励：调整后的高斯函数（中心1800，标准差2000）
            co_center = 1800
            co_std = 2000  # 增大标准差使高CO值奖励变化更平缓
            co_reward = np.exp(-((co - co_center) / co_std) ** 2)

            # Sn奖励：Sn越低越好，指数衰减（直接与sn值负相关）
            sn_decay_rate = 0.5  # 调整衰减速率
            sn_reward = np.exp(-sn_decay_rate * sn)

            # 合成基础奖励（权重可根据实际效果调整）
            reward = 0.6 * co_reward + 0.4 * sn_reward

            # 连续惩罚机制（避免突变）
            co_penalty = 0.0
            # CO超出安全范围的惩罚（假设允许范围0-5000）
            if co > 6000:
                co_penalty += (co - 5000) * 0.0005  # 每超出5000单位惩罚0.002
            elif co < 0:
                co_penalty += abs(co) * 0.01  # 负值严重惩罚

            sn_penalty = 0.0
            # Sn过高惩罚（假设安全阈值6）
            if sn > 6:
                sn_penalty += (sn - 6) * 0.1  # 每超出6单位惩罚0.1

            # 应用惩罚项
            reward -= (co_penalty + sn_penalty)

            # 适度裁剪防止极端值（保持训练稳定性）
            reward = np.clip(reward, -2, 2)

            # 调试输出
            print(f'CO: {co:.1f} → Reward: {co_reward:.3f} (Penalty: {co_penalty:.3f}) | '
                  f'Sn: {sn:.1f} → Reward: {sn_reward:.3f} (Penalty: {sn_penalty:.3f}) | '
                  f'Total Reward: {reward:.3f}')
            return reward
        except Exception as e:
            print(f"Error calculating reward: {str(e)}, State: {state}")
            return -1.0  # 错误时返回负奖励


