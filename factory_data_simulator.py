#!/usr/bin/env python3
"""
工厂数据模拟器
模拟工厂设备向系统发送实时数据
"""

import requests
import json
import time
import random
import numpy as np
from datetime import datetime
import threading
import signal
import sys

class FactoryDataSimulator:
    def __init__(self, api_base_url="http://localhost:8000"):
        self.api_base_url = api_base_url
        self.running = False
        self.data_interval = 5  # 数据发送间隔（秒）
        
        # 基础数据值
        self.base_data = {
            'temperature': 31.5,
            'pressure': 98.7,
            'flow': 496.8,
            'gas': 4327,
            'oxygen': 78.9,
            'cooling': 87.2,
            'Offgas CO Analyzer': 4327,
            'Roof & Ports CW Return': 31.3,
            'Transition CW Return': 33.6,
            'Pb T/Blk2 Iner CW Ret P2': 31.7,
            'Bin 6 PV Feedrate': 3,
            'CW Flw Transition Cool': 98.7,
            'CW Flw Roof & Ports': 65.1,
            'Total Cooling Water Flow': 496.7,
            'Oxygen Purity PV': 78.9,
            'CW Flw Shell Cone': 87.2,
            'Process Mode': 4,
            'Lance Air Flow': 14776,
            'Fuel Coal SV': 6200,
            'Coal Carrier Air Flow': 495,
            'Tip Flow Demand(air eqv)': 38738,
            'Tip Flow Supply(air eqv)': 36465,
            'Total Concentrate': 576704.2,
            'L=150mm H=3500mm': 250.1
        }
        
        # 数据变化范围
        self.variation_ranges = {
            'temperature': 0.2,
            'pressure': 0.1,
            'flow': 2.0,
            'gas': 100.0,
            'oxygen': 0.1,
            'cooling': 0.1,
            'Offgas CO Analyzer': 100.0,
            'Roof & Ports CW Return': 0.2,
            'Transition CW Return': 0.2,
            'Pb T/Blk2 Iner CW Ret P2': 0.2,
            'Bin 6 PV Feedrate': 0.5,
            'CW Flw Transition Cool': 0.2,
            'CW Flw Roof & Ports': 0.2,
            'Total Cooling Water Flow': 2.0,
            'Oxygen Purity PV': 0.1,
            'CW Flw Shell Cone': 0.2,
            'Process Mode': 0,
            'Lance Air Flow': 200.0,
            'Fuel Coal SV': 100.0,
            'Coal Carrier Air Flow': 10.0,
            'Tip Flow Demand(air eqv)': 500.0,
            'Tip Flow Supply(air eqv)': 500.0,
            'Total Concentrate': 10.0,
            'L=150mm H=3500mm': 0.1
        }
        
        # 异常事件配置
        self.anomaly_config = {
            'temperature': {'min': 30, 'max': 35, 'probability': 0.05},
            'pressure': {'min': 95, 'max': 105, 'probability': 0.03},
            'gas': {'min': 0, 'max': 5000, 'probability': 0.08},
            'oxygen': {'min': 75, 'max': 85, 'probability': 0.02}
        }
        
        self.data_count = 0
        self.error_count = 0
        
    def generate_realistic_data(self):
        """生成逼真的工厂数据"""
        data = {}
        
        for key, base_value in self.base_data.items():
            # 基础变化
            variation = self.variation_ranges.get(key, 0.1)
            change = (random.random() - 0.5) * variation
            new_value = base_value + change
            
            # 异常事件
            if key in self.anomaly_config and random.random() < self.anomaly_config[key]['probability']:
                anomaly_range = self.anomaly_config[key]
                if random.random() < 0.5:
                    # 超出上限
                    new_value = anomaly_range['max'] + random.random() * variation
                else:
                    # 低于下限
                    new_value = anomaly_range['min'] - random.random() * variation
                print(f"⚠️  异常事件: {key} = {new_value:.2f}")
            
            # 确保数值合理
            if key == 'Process Mode':
                new_value = int(new_value)  # 工艺模式为整数
            elif 'Flow' in key or 'flow' in key:
                new_value = max(0, new_value)  # 流量不能为负
            elif 'Purity' in key or 'oxygen' in key:
                new_value = max(0, min(100, new_value))  # 纯度在0-100之间
            
            data[key] = round(new_value, 2) if isinstance(new_value, float) else new_value
        
        return data
    
    def send_factory_data(self, data):
        """发送工厂数据到API"""
        try:
            payload = {
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            
            response = requests.post(
                f"{self.api_base_url}/real-time-monitoring/factory-data",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    self.data_count += 1
                    print(f"✅ 数据发送成功 (第{self.data_count}次)")
                    return True
                else:
                    print(f"❌ 数据发送失败: {result.get('message')}")
                    self.error_count += 1
                    return False
            else:
                print(f"❌ HTTP错误: {response.status_code}")
                self.error_count += 1
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 网络错误: {e}")
            self.error_count += 1
            return False
        except Exception as e:
            print(f"❌ 未知错误: {e}")
            self.error_count += 1
            return False
    
    def run_simulation(self):
        """运行数据模拟"""
        print("🏭 工厂数据模拟器启动")
        print(f"📡 API地址: {self.api_base_url}")
        print(f"⏱️  数据间隔: {self.data_interval}秒")
        print("=" * 50)
        
        self.running = True
        start_time = time.time()
        
        while self.running:
            try:
                # 生成数据
                data = self.generate_realistic_data()
                
                # 发送数据
                success = self.send_factory_data(data)
                
                # 显示关键数据
                if success:
                    print(f"📊 温度: {data['temperature']:.1f}°C | "
                          f"压力: {data['pressure']:.1f}MPa | "
                          f"流量: {data['flow']:.1f}m³/h | "
                          f"CO: {data['gas']:.0f}ppm")
                
                # 等待下次发送
                time.sleep(self.data_interval)
                
            except KeyboardInterrupt:
                print("\n🛑 收到停止信号")
                break
            except Exception as e:
                print(f"❌ 模拟器错误: {e}")
                time.sleep(self.data_interval)
        
        # 显示统计信息
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 50)
        print("📈 模拟统计:")
        print(f"   运行时间: {elapsed_time:.1f}秒")
        print(f"   成功发送: {self.data_count}次")
        print(f"   发送失败: {self.error_count}次")
        if self.data_count > 0:
            success_rate = (self.data_count / (self.data_count + self.error_count)) * 100
            print(f"   成功率: {success_rate:.1f}%")
        print("🏭 工厂数据模拟器已停止")
    
    def stop_simulation(self):
        """停止模拟"""
        self.running = False
    
    def set_data_interval(self, interval):
        """设置数据发送间隔"""
        self.data_interval = interval
        print(f"⏱️  数据间隔已设置为 {interval}秒")
    
    def set_anomaly_probability(self, key, probability):
        """设置异常事件概率"""
        if key in self.anomaly_config:
            self.anomaly_config[key]['probability'] = probability
            print(f"⚠️  {key} 异常概率已设置为 {probability:.2%}")
        else:
            print(f"❌ 未知参数: {key}")

def signal_handler(sig, frame):
    """信号处理器"""
    print("\n🛑 收到停止信号，正在关闭模拟器...")
    if simulator:
        simulator.stop_simulation()
    sys.exit(0)

def main():
    """主函数"""
    global simulator
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 创建模拟器
    simulator = FactoryDataSimulator()
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='工厂数据模拟器')
    parser.add_argument('--interval', type=int, default=5, help='数据发送间隔（秒）')
    parser.add_argument('--url', type=str, default='http://localhost:8000', help='API基础URL')
    parser.add_argument('--anomaly-temp', type=float, default=0.05, help='温度异常概率')
    parser.add_argument('--anomaly-pressure', type=float, default=0.03, help='压力异常概率')
    parser.add_argument('--anomaly-gas', type=float, default=0.08, help='气体异常概率')
    
    args = parser.parse_args()
    
    # 配置模拟器
    simulator.api_base_url = args.url
    simulator.set_data_interval(args.interval)
    simulator.set_anomaly_probability('temperature', args.anomaly_temp)
    simulator.set_anomaly_probability('pressure', args.anomaly_pressure)
    simulator.set_anomaly_probability('gas', args.anomaly_gas)
    
    # 运行模拟
    simulator.run_simulation()

if __name__ == "__main__":
    simulator = None
    main() 