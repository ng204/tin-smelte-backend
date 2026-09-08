#!/usr/bin/env python3
"""
锡冶炼智能分析系统服务器启动脚本
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ 错误: 需要Python 3.8或更高版本")
        print(f"当前版本: {sys.version}")
        return False
    print(f"✓ Python版本检查通过: {sys.version}")
    return True

def check_dependencies():
    """检查依赖包"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'pydantic',
        'torch',
        'torchvision',
        'pandas',
        'numpy',
        'Pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} 未安装")
    
    if missing_packages:
        print(f"\n缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True

def check_model_file():
    """检查模型文件"""
    model_path = Path("best_model.pth")
    if not model_path.exists():
        print("⚠️  警告: 模型文件 best_model.pth 不存在")
        print("系统将使用模拟数据进行演示")
        return False
    print("✓ 模型文件检查通过")
    return True

def check_port_availability(port=8000):
    """检查端口是否可用"""
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        
        if result == 0:
            print(f"⚠️  警告: 端口 {port} 已被占用")
            return False
        else:
            print(f"✓ 端口 {port} 可用")
            return True
    except Exception as e:
        print(f"端口检查失败: {e}")
        return False

def start_server(port=8000, reload=True):
    """启动服务器"""
    print(f"\n🚀 启动服务器...")
    print(f"端口: {port}")
    print(f"自动重载: {reload}")
    
    try:
        # 构建启动命令
        cmd = [
            sys.executable, "-m", "uvicorn",
            "main:app",
            "--host", "0.0.0.0",
            "--port", str(port),
            "--timeout-keep-alive", "600",  # 设置保持连接超时为600秒（10分钟）
        ]
        
        if reload:
            cmd.append("--reload")
        
        print(f"启动命令: {' '.join(cmd)}")
        
        # 启动服务器
        process = subprocess.Popen(cmd)
        
        # 等待服务器启动
        print("等待服务器启动...")
        time.sleep(3)
        
        # 检查服务器是否成功启动
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                print("✅ 服务器启动成功!")
                print(f"📖 API文档: http://localhost:{port}/docs")
                print(f"🔗 健康检查: http://localhost:{port}/health")
                print(f"🏠 主页: http://localhost:{port}/")
                print("\n按 Ctrl+C 停止服务器")
                
                # 等待用户中断
                try:
                    process.wait()
                except KeyboardInterrupt:
                    print("\n🛑 正在停止服务器...")
                    process.terminate()
                    process.wait()
                    print("✅ 服务器已停止")
                
            else:
                print(f"❌ 服务器启动失败，状态码: {response.status_code}")
                process.terminate()
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 无法连接到服务器: {e}")
            process.terminate()
            return False
            
    except Exception as e:
        print(f"❌ 启动服务器失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 50)
    print("锡冶炼智能分析系统 - 服务器启动脚本")
    print("=" * 50)
    
    # 环境检查
    print("\n🔍 环境检查...")
    
    if not check_python_version():
        sys.exit(1)
    
    if not check_dependencies():
        print("\n请安装缺少的依赖包后重试")
        sys.exit(1)
    
    check_model_file()
    
    port = 8000
    if not check_port_availability(port):
        print(f"\n请确保端口 {port} 未被占用，或修改端口号")
        sys.exit(1)
    
    print("\n✅ 环境检查通过")
    
    # 启动服务器
    start_server(port)

if __name__ == "__main__":
    main() 