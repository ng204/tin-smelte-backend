@echo off
setlocal enabledelayedexpansion

REM 锡冶炼智能分析系统部署脚本 (Windows版本)

set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM 打印带颜色的消息
:print_info
echo %BLUE%[INFO]%NC% %~1
goto :eof

:print_success
echo %GREEN%[SUCCESS]%NC% %~1
goto :eof

:print_warning
echo %YELLOW%[WARNING]%NC% %~1
goto :eof

:print_error
echo %RED%[ERROR]%NC% %~1
goto :eof

REM 显示帮助信息
:show_help
echo 锡冶炼智能分析系统部署脚本
echo.
echo 用法: %0 [选项]
echo.
echo 选项:
echo   local     - 本地部署（使用Python虚拟环境）
echo   docker    - Docker部署
echo   docker-compose - Docker Compose部署（推荐）
echo   clean     - 清理部署
echo   help      - 显示此帮助信息
echo.
echo 示例:
echo   %0 local           # 本地部署
echo   %0 docker-compose  # Docker Compose部署
goto :eof

REM 本地部署
:deploy_local
call :print_info "开始本地部署..."

REM 检查Python版本
python --version >nul 2>&1
if errorlevel 1 (
    call :print_error "Python 未安装"
    exit /b 1
)

REM 创建虚拟环境
if not exist "venv" (
    call :print_info "创建虚拟环境..."
    python -m venv venv
)

REM 激活虚拟环境
call :print_info "激活虚拟环境..."
call venv\Scripts\activate.bat

REM 安装依赖
call :print_info "安装依赖包..."
python -m pip install --upgrade pip
pip install -r requirements.txt

REM 检查模型文件
if not exist "best_model.pth" (
    call :print_warning "模型文件 best_model.pth 不存在，将使用模拟数据"
)

call :print_success "本地部署完成！"
call :print_info "运行以下命令启动服务："
echo   venv\Scripts\activate.bat
echo   python start_server.py
echo   或
echo   uvicorn main:app --reload --host 0.0.0.0 --port 8000
goto :eof

REM Docker部署
:deploy_docker
call :print_info "开始Docker部署..."

REM 检查Docker
docker --version >nul 2>&1
if errorlevel 1 (
    call :print_error "Docker 未安装"
    exit /b 1
)

REM 构建镜像
call :print_info "构建Docker镜像..."
docker build -t tin-smelte-api .

REM 运行容器
call :print_info "启动容器..."
docker run -d --name tin-smelte-api -p 8000:8000 -v %cd%\models:/app/models -v %cd%\data:/app/data tin-smelte-api

call :print_success "Docker部署完成！"
call :print_info "服务地址: http://localhost:8000"
call :print_info "API文档: http://localhost:8000/docs"
goto :eof

REM Docker Compose部署
:deploy_docker_compose
call :print_info "开始Docker Compose部署..."

REM 检查Docker Compose
docker-compose --version >nul 2>&1
if errorlevel 1 (
    call :print_error "Docker Compose 未安装"
    exit /b 1
)

REM 创建必要的目录
if not exist "models" mkdir models
if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "ssl" mkdir ssl

REM 启动服务
call :print_info "启动服务..."
docker-compose up -d

REM 等待服务启动
call :print_info "等待服务启动..."
timeout /t 10 /nobreak >nul

REM 检查服务状态
curl -f http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    call :print_error "服务启动失败，请检查日志"
    docker-compose logs
) else (
    call :print_success "Docker Compose部署完成！"
    call :print_info "服务地址: http://localhost:8000"
    call :print_info "API文档: http://localhost:8000/docs"
    call :print_info "健康检查: http://localhost:8000/health"
)
goto :eof

REM 清理部署
:clean_deployment
call :print_info "清理部署..."

REM 停止Docker容器
docker ps -q --filter "name=tin-smelte-api" >nul 2>&1
if not errorlevel 1 (
    call :print_info "停止Docker容器..."
    docker stop tin-smelte-api
    docker rm tin-smelte-api
)

REM 停止Docker Compose服务
if exist "docker-compose.yml" (
    call :print_info "停止Docker Compose服务..."
    docker-compose down
)

REM 删除Docker镜像
docker images | findstr "tin-smelte-api" >nul 2>&1
if not errorlevel 1 (
    call :print_info "删除Docker镜像..."
    docker rmi tin-smelte-api
)

REM 清理虚拟环境
if exist "venv" (
    call :print_info "删除虚拟环境..."
    rmdir /s /q venv
)

call :print_success "清理完成！"
goto :eof

REM 显示服务状态
:show_status
call :print_info "服务状态检查..."

REM 检查本地服务
curl -f http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    call :print_warning "本地服务未运行"
) else (
    call :print_success "本地服务运行正常"
)

REM 检查Docker容器
docker ps | findstr "tin-smelte-api" >nul 2>&1
if errorlevel 1 (
    call :print_warning "Docker容器未运行"
) else (
    call :print_success "Docker容器运行正常"
)

REM 检查Docker Compose服务
docker-compose ps | findstr "Up" >nul 2>&1
if errorlevel 1 (
    call :print_warning "Docker Compose服务未运行"
) else (
    call :print_success "Docker Compose服务运行正常"
)
goto :eof

REM 主函数
:main
if "%1"=="" goto :show_help
if "%1"=="help" goto :show_help
if "%1"=="local" goto :deploy_local
if "%1"=="docker" goto :deploy_docker
if "%1"=="docker-compose" goto :deploy_docker_compose
if "%1"=="clean" goto :clean_deployment
if "%1"=="status" goto :show_status
goto :show_help

REM 执行主函数
call :main %* 