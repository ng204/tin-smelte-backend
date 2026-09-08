#!/bin/bash

# 锡冶炼智能分析系统启动脚本
# 用于快速启动后端服务和数据模拟器

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 检查Python是否安装
check_python() {
    if ! command_exists python3; then
        if ! command_exists python; then
            print_error "未找到Python，请先安装Python 3.7+"
            exit 1
        else
            PYTHON_CMD="python"
        fi
    else
        PYTHON_CMD="python3"
    fi
    
    print_success "找到Python: $($PYTHON_CMD --version)"
}

# 检查是否在正确的目录
check_directory() {
    if [ ! -f "main.py" ]; then
        print_error "请在 tin-smelte-backend-master 目录下运行此脚本"
        exit 1
    fi
    print_success "目录检查通过"
}

# 检查并安装依赖
check_dependencies() {
    print_info "检查依赖包..."
    
    if ! $PYTHON_CMD -c "import fastapi, uvicorn, websockets" 2>/dev/null; then
        print_warning "检测到缺少依赖包，正在安装..."
        pip install fastapi uvicorn websockets requests
        print_success "依赖包安装完成"
    else
        print_success "依赖包检查通过"
    fi
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --no-simulator    不启动数据模拟器"
    echo "  --interval N      设置数据模拟器发送间隔（秒，默认5）"
    echo "  --help            显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                    # 启动完整系统"
    echo "  $0 --no-simulator     # 只启动后端API"
    echo "  $0 --interval 3       # 设置数据发送间隔为3秒"
}

# 主函数
main() {
    echo ""
    echo "========================================"
    echo "🎯 锡冶炼智能分析系统启动脚本"
    echo "========================================"
    echo ""
    
    # 检查环境
    check_python
    check_directory
    check_dependencies
    
    echo ""
    print_info "启动系统..."
    echo ""
    
    # 启动Python启动脚本
    $PYTHON_CMD start_system.py "$@"
    
    echo ""
    print_info "系统已停止"
}

# 处理命令行参数
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi

# 运行主函数
main "$@" 