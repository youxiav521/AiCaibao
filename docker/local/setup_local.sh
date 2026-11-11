#!/bin/bash
set -e

# setup_local.sh - 跨平台环境初始化脚本 (macOS / Windows)
# 1. 复制 .env.local 到 .env
# 2. 启动 docker-compose-env.yml

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[setup_local] 开始环境初始化..."

# 1. 复制环境变量文件
echo "[setup_local] 复制 docker/local/.env.local -> .env"
if [ -f "$SCRIPT_DIR/.env.local" ]; then
    cp -f "$SCRIPT_DIR/.env.local" "$SCRIPT_DIR/../../.env"
    echo "[setup_local] .env 文件已更新"
else
    echo "[setup_local] 警告: 未找到 .env.local 文件，跳过"
fi

# 2. 复制 llm.json 配置文件
echo "[setup_local] 复制 docker/local/llm.json -> config/llm.json"
if [ -f "$SCRIPT_DIR/llm.json" ]; then
    mkdir -p "$SCRIPT_DIR/../../config"
    cp -f "$SCRIPT_DIR/llm.json" "$SCRIPT_DIR/../../config/llm.json"
    echo "[setup_local] config/llm.json 文件已更新"
else
    echo "[setup_local] 警告: 未找到 docker/local/llm.json 文件，跳过"
fi

# 3. 判断操作系统并安装 MinIO、启动 Docker
OS="$(uname -s)"
if [ "$OS" = "Darwin" ]; then
    # 检查 MinIO 是否已安装
    if command -v minio &> /dev/null; then
        echo "[setup_local] MinIO 已安装，跳过安装步骤"
    else
        echo "[setup_local] 检测到 macOS，安装 MinIO..."
        # 使用 Homebrew 安装 MinIO
        brew install minio/stable/minio
        echo "[setup_local] MinIO 安装完成"
    fi

    # 创建数据目录
    mkdir -p ~/minio-data

    # 检查 MinIO 是否已启动
    if lsof -i :9000 &> /dev/null; then
        echo "[setup_local] MinIO 已在运行，跳过启动步骤"
    else
        echo "[setup_local] 启动 MinIO..."
        # 设置 MinIO 环境变量（与 .env.local 保持一致）
        export MINIO_ROOT_USER=minioadmin
        export MINIO_ROOT_PASSWORD=minioadmin
        # 后台启动 MinIO
        minio server ~/minio-data > ~/minio.log 2>&1 &

        # 检查 MinIO 是否成功启动
        echo "[setup_local] 检查 MinIO 启动状态..."
        sleep 3
        if lsof -i :9000 &> /dev/null; then
            echo "[setup_local] MinIO 启动成功"
        else
            echo "[setup_local] 警告: MinIO 启动可能失败，请检查 ~/minio.log"
        fi
    fi

    echo "[setup_local] 启动 Docker 环境..."
    docker compose -f "$SCRIPT_DIR/docker-compose-env.yml" up -d
    echo "[setup_local] Docker 环境已启动"
elif [ "$OS" = "Linux" ]; then
    echo "[setup_local] 检测到 Linux..."
    # 检查 MinIO 是否已安装
    if command -v minio &> /dev/null; then
        echo "[setup_local] MinIO 已安装，跳过安装步骤"
    else
        echo "[setup_local] 安装 MinIO..."
        # 下载并安装 MinIO
        wget -q https://dl.min.io/server/minio/release/linux-amd64/minio -O /usr/local/bin/minio
        chmod +x /usr/local/bin/minio
        echo "[setup_local] MinIO 安装完成"
    fi

    # 创建数据目录
    mkdir -p ~/minio-data

    # 检查 MinIO 是否已启动
    if lsof -i :9000 &> /dev/null; then
        echo "[setup_local] MinIO 已在运行，跳过启动步骤"
    else
        echo "[setup_local] 启动 MinIO..."
        # 设置 MinIO 环境变量（与 .env.local 保持一致）
        export MINIO_ROOT_USER=minioadmin
        export MINIO_ROOT_PASSWORD=minioadmin
        # 后台启动 MinIO
        minio server ~/minio-data > ~/minio.log 2>&1 &

        # 检查 MinIO 是否成功启动
        echo "[setup_local] 检查 MinIO 启动状态..."
        sleep 3
        if lsof -i :9000 &> /dev/null; then
            echo "[setup_local] MinIO 启动成功"
        else
            echo "[setup_local] 警告: MinIO 启动可能失败，请检查 ~/minio.log"
        fi
    fi

    echo "[setup_local] 启动 Docker 环境..."
    docker compose -f "$SCRIPT_DIR/docker-compose-env.yml" up -d
    echo "[setup_local] Docker 环境已启动"
elif [ "$OS" = "CYGWIN"* ] || [ "$OS" = "MINGW"* ] || [ "$OS" = "MSYS"* ] || [ "$OS" = "Windows" ]; then
    echo "[setup_local] 检测到 Windows..."
    # 检查 MinIO 是否已安装
    if [ -f "C:\\minio.exe" ]; then
        echo "[setup_local] MinIO 已安装，跳过安装步骤"
    else
        echo "[setup_local] 安装 MinIO..."
        # 下载 MinIO 可执行文件到 C 盘根目录
        curl -s -o "C:\\minio.exe" "https://dl.min.io/server/minio/release/windows-amd64/minio.exe"
        # 设置 MinIO 环境变量（与 .env.local 保持一致）
        setx MINIO_ROOT_USER minioadmin
        setx MINIO_ROOT_PASSWORD minioadmin
        # 创建数据目录
        mkdir -p "D:\\Data"
        echo "[setup_local] MinIO 安装完成"
    fi

    # 检查 MinIO 是否已启动
    if netstat -an | grep -q ":9000.*LISTENING"; then
        echo "[setup_local] MinIO 已在运行，跳过启动步骤"
    else
        echo "[setup_local] 启动 MinIO..."
        # 启动 MinIO 服务
        start "MinIO" "C:\\minio.exe" server "D:\\Data"

        # 检查 MinIO 是否成功启动
        echo "[setup_local] 检查 MinIO 启动状态..."
        sleep 3
        if netstat -an | grep -q ":9000.*LISTENING"; then
            echo "[setup_local] MinIO 启动成功"
        else
            echo "[setup_local] 警告: MinIO 启动可能失败，请检查 C:\\minio.log"
        fi
    fi

    echo "[setup_local] 启动 Docker 环境..."
    docker compose -f "$SCRIPT_DIR\docker-compose-env.yml" up -d
    echo "[setup_local] Docker 环境已启动"
else
    echo "[setup_local] 未知操作系统 ($OS)，跳过 Docker 启动"
fi

# ========== 构建本地 .coder_venv 虚拟环境（根目录）==========
echo "[setup_local] 开始构建 .coder_venv 虚拟环境（根目录，隐藏文件）..."

# 定义 .coder_venv 路径（项目根目录下，与 .venv 同级）
CODER_VENV_DIR="$SCRIPT_DIR/../../.coder_venv"
CODER_VENV_DIR=$(cd "$(dirname "$CODER_VENV_DIR")" && pwd)/$(basename "$CODER_VENV_DIR")

# 判断虚拟环境是否已存在（检查 pyvenv.cfg）
if [ -f "$CODER_VENV_DIR/pyvenv.cfg" ]; then
    echo "[setup_local] .coder_venv 已存在，跳过创建"
else
    echo "[setup_local] 创建 .coder_venv 虚拟环境..."
    # 优先使用 python3，兼容 macOS 系统默认没有 python 命令的情况
    if command -v python3 &> /dev/null; then
        python3 -m venv "$CODER_VENV_DIR"
    else
        python -m venv "$CODER_VENV_DIR"
    fi
    echo "[setup_local] .coder_venv 创建完成"
fi

# 确定 pip 路径（setup_local.sh 只在 bash 环境运行：Linux/macOS/WSL/Git Bash）
# 类 Unix 环境下 venv 的 python/pip 都在 bin 目录下
CODER_PIP="$CODER_VENV_DIR/bin/pip"
CODER_PYTHON="$CODER_VENV_DIR/bin/python"

# 安装 coder 分组依赖（版本在 pyproject.toml 中统一管理）
# 使用 uv 的 --python 选项在指定 venv 中安装依赖，确保版本一致性
echo "[setup_local] 安装 .coder_venv coder 分组依赖（版本在 pyproject.toml 中统一管理）..."
$CODER_PIP install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
# 切换到项目根目录执行 uv 命令，确保能找到 pyproject.toml
cd "$SCRIPT_DIR/../../"
uv pip install --python $CODER_PYTHON --group coder -i https://pypi.tuna.tsinghua.edu.cn/simple
echo "[setup_local] .coder_venv 基础包安装完成"

echo "[setup_local] 环境初始化完成"
