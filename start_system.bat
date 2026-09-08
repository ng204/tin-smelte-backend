@echo off
chcp 65001 >nul
title 锡冶炼智能分析系统 - 实时数据监测

echo.
echo ========================================
echo 🎯 锡冶炼智能分析系统启动脚本
echo ========================================
echo.

:: 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未找到Python，请先安装Python 3.7+
    pause
    exit /b 1
)

:: 检查是否在正确的目录
if not exist "main.py" (
    echo ❌ 错误: 请在 tin-smelte-backend-master 目录下运行此脚本
    pause
    exit /b 1
)

:: 检查依赖是否安装
echo 📦 检查依赖包...
python -c "import fastapi, uvicorn, websockets" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  检测到缺少依赖包，正在安装...
    pip install fastapi uvicorn websockets requests
)

echo.
echo 🚀 启动系统...
echo.

:: 启动Python启动脚本
python start_system.py %*

echo.
echo 🛑 系统已停止
pause 