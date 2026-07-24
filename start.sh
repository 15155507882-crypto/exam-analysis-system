#!/bin/bash
# 中考成绩分析系统 - 启动脚本
# 自动检测平台并设置所需环境变量

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# macOS 上 WeasyPrint 需要 Homebrew 库路径
if [[ "$(uname)" == "Darwin" ]] && [[ -d "/opt/homebrew/lib" ]]; then
    export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
fi

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "正在创建虚拟环境..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

echo "============================================"
echo "  中考成绩分析系统 v1.0"
echo "  启动地址: http://localhost:8501"
echo "  按 Ctrl+C 停止"
echo "============================================"

streamlit run app.py --server.address 0.0.0.0 --server.port 8501
