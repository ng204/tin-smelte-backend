#!/bin/bash

# 锡冶炼智能分析系统 - 扩展API部署脚本
# 版本: 1.0.0
# 日期: 2024-01-01

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "$1 未安装，请先安装 $1"
        exit 1
    fi
}

# 检查Python版本
check_python_version() {
    local python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    local required_version="3.8"
    
    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
        log_error "Python版本过低，需要Python 3.8或更高版本，当前版本: $python_version"
        exit 1
    fi
    
    log_success "Python版本检查通过: $python_version"
}

# 安装Python依赖
install_python_dependencies() {
    log_info "安装Python依赖..."
    
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
        log_success "Python依赖安装完成"
    else
        log_warning "未找到requirements.txt文件，跳过Python依赖安装"
    fi
}

# 创建虚拟环境
create_virtual_environment() {
    log_info "创建Python虚拟环境..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        log_success "虚拟环境创建完成"
    else
        log_warning "虚拟环境已存在，跳过创建"
    fi
    
    # 激活虚拟环境
    source venv/bin/activate
    log_success "虚拟环境已激活"
}

# 创建必要的目录
create_directories() {
    log_info "创建必要的目录..."
    
    mkdir -p logs
    mkdir -p data
    mkdir -p config
    mkdir -p temp
    
    log_success "目录创建完成"
}

# 创建配置文件
create_config_files() {
    log_info "创建配置文件..."
    
    # 创建主配置文件
    cat > config/app_config.json << EOF
{
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
        "debug": false
    },
    "database": {
        "url": "sqlite:///data/monitoring.db"
    },
    "websocket": {
        "enabled": true,
        "max_connections": 100
    },
    "data_quality": {
        "enabled": true,
        "check_interval": 60
    },
    "logging": {
        "level": "INFO",
        "file": "logs/app.log",
        "max_size": "10MB",
        "backup_count": 5
    }
}
EOF

    # 创建环境变量文件
    cat > .env << EOF
# 应用配置
APP_NAME=Tin Smelting Intelligent Analysis System
APP_VERSION=1.0.0
DEBUG=false

# 服务器配置
HOST=0.0.0.0
PORT=8000

# 数据库配置
DATABASE_URL=sqlite:///data/monitoring.db

# 安全配置
SECRET_KEY=your-secret-key-here
API_KEY=your-api-key-here

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# WebSocket配置
WEBSOCKET_ENABLED=true
WEBSOCKET_MAX_CONNECTIONS=100

# 数据质量配置
DATA_QUALITY_ENABLED=true
DATA_QUALITY_CHECK_INTERVAL=60
EOF

    log_success "配置文件创建完成"
}

# 创建systemd服务文件
create_systemd_service() {
    log_info "创建systemd服务文件..."
    
    local service_file="/etc/systemd/system/tin-smelting-api.service"
    local current_dir=$(pwd)
    
    sudo tee $service_file > /dev/null << EOF
[Unit]
Description=Tin Smelting Intelligent Analysis System API
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$current_dir
Environment=PATH=$current_dir/venv/bin
ExecStart=$current_dir/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    log_success "systemd服务文件创建完成: $service_file"
}

# 创建nginx配置
create_nginx_config() {
    log_info "创建nginx配置文件..."
    
    local nginx_config="/etc/nginx/sites-available/tin-smelting-api"
    
    sudo tee $nginx_config > /dev/null << EOF
server {
    listen 80;
    server_name your-domain.com;  # 替换为你的域名

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket支持
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # 静态文件缓存
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
EOF

    # 创建软链接
    sudo ln -sf $nginx_config /etc/nginx/sites-enabled/
    
    log_success "nginx配置文件创建完成: $nginx_config"
}

# 启动服务
start_service() {
    log_info "启动API服务..."
    
    # 检查服务是否已经在运行
    if pgrep -f "python main.py" > /dev/null; then
        log_warning "API服务已在运行，正在停止..."
        pkill -f "python main.py"
        sleep 2
    fi
    
    # 启动服务
    nohup python main.py > logs/app.log 2>&1 &
    local pid=$!
    
    # 等待服务启动
    sleep 3
    
    if kill -0 $pid 2>/dev/null; then
        log_success "API服务启动成功，PID: $pid"
        echo $pid > .pid
    else
        log_error "API服务启动失败"
        exit 1
    fi
}

# 停止服务
stop_service() {
    log_info "停止API服务..."
    
    if [ -f ".pid" ]; then
        local pid=$(cat .pid)
        if kill -0 $pid 2>/dev/null; then
            kill $pid
            log_success "API服务已停止"
        else
            log_warning "服务进程不存在"
        fi
        rm -f .pid
    else
        log_warning "未找到PID文件"
    fi
}

# 重启服务
restart_service() {
    log_info "重启API服务..."
    stop_service
    sleep 2
    start_service
}

# 检查服务状态
check_service_status() {
    log_info "检查服务状态..."
    
    if [ -f ".pid" ]; then
        local pid=$(cat .pid)
        if kill -0 $pid 2>/dev/null; then
            log_success "API服务正在运行，PID: $pid"
            
            # 检查端口
            if netstat -tuln | grep ":8000 " > /dev/null; then
                log_success "API服务端口8000正在监听"
            else
                log_warning "API服务端口8000未监听"
            fi
        else
            log_error "API服务未运行"
        fi
    else
        log_error "未找到PID文件，服务可能未启动"
    fi
}

# 查看日志
view_logs() {
    log_info "查看服务日志..."
    
    if [ -f "logs/app.log" ]; then
        tail -f logs/app.log
    else
        log_error "日志文件不存在"
    fi
}

# 清理临时文件
cleanup() {
    log_info "清理临时文件..."
    
    rm -rf temp/*
    rm -f .pid
    
    log_success "清理完成"
}

# 显示帮助信息
show_help() {
    echo "锡冶炼智能分析系统 - 扩展API部署脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  install     安装和配置系统"
    echo "  start       启动API服务"
    echo "  stop        停止API服务"
    echo "  restart     重启API服务"
    echo "  status      检查服务状态"
    echo "  logs        查看服务日志"
    echo "  cleanup     清理临时文件"
    echo "  help        显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 install    # 安装和配置系统"
    echo "  $0 start      # 启动服务"
    echo "  $0 status     # 检查状态"
}

# 主函数
main() {
    case "${1:-help}" in
        install)
            log_info "开始安装锡冶炼智能分析系统扩展API..."
            check_command python3
            check_command pip3
            check_python_version
            create_virtual_environment
            install_python_dependencies
            create_directories
            create_config_files
            create_systemd_service
            create_nginx_config
            log_success "安装完成！"
            log_info "使用 '$0 start' 启动服务"
            ;;
        start)
            start_service
            ;;
        stop)
            stop_service
            ;;
        restart)
            restart_service
            ;;
        status)
            check_service_status
            ;;
        logs)
            view_logs
            ;;
        cleanup)
            cleanup
            ;;
        help|*)
            show_help
            ;;
    esac
}

# 执行主函数
main "$@" 