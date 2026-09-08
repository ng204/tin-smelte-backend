#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
锡冶炼智能分析系统 - 实时数据监测系统启动脚本
用于快速启动后端服务和数据模拟器
"""

import os
import sys
import time
import subprocess
import threading
import signal
import argparse
from pathlib import Path

class SystemStarter:
    def __init__(self):
        self.processes = []
        self.running = True
        
    def start_backend(self):
        """启动后端API服务"""
        print("🚀 启动后端API服务...")
        try:
            # 检查是否在正确的目录
            if not os.path.exists("main.py"):
                print("❌ 错误: 请在 tin-smelte-backend-master 目录下运行此脚本")
                return False
                
            # 启动后端服务
            process = subprocess.Popen([
                sys.executable, "main.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes.append(("后端API", process))
            print("✅ 后端API服务启动成功")
            return True
            
        except Exception as e:
            print(f"❌ 启动后端服务失败: {e}")
            return False
    
    def start_simulator(self, interval=5):
        """启动工厂数据模拟器"""
        print("🏭 启动工厂数据模拟器...")
        try:
            process = subprocess.Popen([
                sys.executable, "factory_data_simulator.py",
                "--interval", str(interval)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes.append(("数据模拟器", process))
            print("✅ 工厂数据模拟器启动成功")
            return True
            
        except Exception as e:
            print(f"❌ 启动数据模拟器失败: {e}")
            return False
    
    def monitor_processes(self):
        """监控进程状态"""
        while self.running:
            for name, process in self.processes:
                if process.poll() is not None:
                    print(f"⚠️  {name} 进程已退出")
            time.sleep(5)
    
    def stop_all(self):
        """停止所有进程"""
        print("\n🛑 正在停止所有服务...")
        self.running = False
        
        for name, process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ {name} 已停止")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"⚠️  {name} 被强制终止")
            except Exception as e:
                print(f"❌ 停止 {name} 时出错: {e}")
    
    def signal_handler(self, signum, frame):
        """信号处理器"""
        print(f"\n📡 收到信号 {signum}，正在关闭系统...")
        self.stop_all()
        sys.exit(0)
    
    def print_status(self):
        """打印系统状态"""
        print("\n" + "="*60)
        print("🎯 锡冶炼智能分析系统 - 实时数据监测")
        print("="*60)
        print("📊 系统组件:")
        for name, process in self.processes:
            status = "🟢 运行中" if process.poll() is None else "🔴 已停止"
            print(f"   {name}: {status}")
        print("="*60)
        print("🌐 访问地址:")
        print("   后端API: http://localhost:8000")
        print("   前端界面: http://localhost:5666/predict/realTime")
        print("   API文档: http://localhost:8000/docs")
        print("="*60)
        print("💡 提示: 按 Ctrl+C 停止所有服务")
        print("="*60 + "\n")
    
    def run(self, start_simulator=True, simulator_interval=5):
        """运行系统"""
        # 设置信号处理器
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        print("🎯 锡冶炼智能分析系统启动中...")
        
        # 启动后端服务
        if not self.start_backend():
            return False
        
        # 等待后端服务启动
        time.sleep(3)
        
        # 启动数据模拟器
        if start_simulator:
            if not self.start_simulator(simulator_interval):
                print("⚠️  数据模拟器启动失败，但系统仍可运行")
        
        # 启动监控线程
        monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
        monitor_thread.start()
        
        # 打印状态
        self.print_status()
        
        # 主循环
        try:
            while self.running:
                time.sleep(10)
                # 每10秒打印一次状态
                if self.running:
                    self.print_status()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_all()

def main():
    parser = argparse.ArgumentParser(description="锡冶炼智能分析系统启动脚本")
    parser.add_argument("--no-simulator", action="store_true", 
                       help="不启动数据模拟器")
    parser.add_argument("--interval", type=int, default=5,
                       help="数据模拟器发送间隔（秒）")
    
    args = parser.parse_args()
    
    starter = SystemStarter()
    starter.run(
        start_simulator=not args.no_simulator,
        simulator_interval=args.interval
    )

if __name__ == "__main__":
    main() 