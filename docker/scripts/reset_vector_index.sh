#!/bin/bash
# 重置向量库索引的便捷执行脚本

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 检查.env文件是否存在
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "⚠️  未找到.env文件，正在从docker/local/.env.local复制..."
    cp -f "$PROJECT_ROOT/docker/local/.env.local" "$PROJECT_ROOT/.env"
fi

# 执行Python脚本
echo "🚀 执行向量索引重置脚本..."
cd "$PROJECT_ROOT" && .venv/bin/python docker/scripts/reset_vector_index.py
