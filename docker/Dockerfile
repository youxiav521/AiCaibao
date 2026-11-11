# build stage
# syntax=docker/dockerfile:1.4

FROM shanbai/python_libre:1.2

ARG PROXY_URL=http://172.16.40.16:7280
# ARG PROXY_URL
ENV http_proxy=${PROXY_URL} \
    https_proxy=${PROXY_URL} \
    HTTP_PROXY=${PROXY_URL} \
    HTTPS_PROXY=${PROXY_URL}

ENV UV_HTTP_TIMEOUT=600
ENV UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

# 修复 APT 源替换（确保 security 路径正确）
RUN sed -i 's|http://deb.debian.org/debian|https://mirrors.aliyun.com/debian|g' /etc/apt/sources.list && \
    sed -i 's|http://security.debian.org/debian-security|https://mirrors.aliyun.com/debian-security|g' /etc/apt/sources.list

# 安装编译依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libmagic1 \
    graphviz \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 只复制依赖文件（关键：不复制项目代码）
COPY pyproject.toml uv.lock ./

# ========== 新增内容开始 ==========
# 确保python-magic包被安装
RUN --mount=type=cache,target=/root/.cache \
    uv add python-magic || (echo "安装python-magic失败" && exit 1)
# ========== 新增内容结束 ==========


# 使用 BuildKit 缓存 uv 下载的依赖，避免每次重新下载
RUN --mount=type=cache,target=/root/.cache \
    uv sync --frozen --no-group dev || (echo "依赖同步失败" && exit 1)

# 检查依赖锁定状态
RUN --mount=type=cache,target=/root/.cache \
    uv sync --check --no-group dev || (echo "依赖未锁定或有变更，请先运行 'uv lock'" && exit 1)

# 卸载 torch 和 nvidia 相关（不使用GPU，兼容低版本uv）
RUN uv pip uninstall -y torch torchvision torchaudio || true
RUN uv pip uninstall -y "nvidia-*" || true

# ========== .coder_venv 虚拟环境构建（根目录，与 .venv 同级）==========
# 定义 .coder_venv 路径（项目根目录下，隐藏文件）
ENV CODER_VENV_DIR=/app/.coder_venv

# 使用 Python 创建 .coder_venv 虚拟环境（使用 BuildKit 缓存 pip 下载）
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv --copies ${CODER_VENV_DIR}

# 设置 .coder_venv 的 pip 和 python 路径（Linux）
ENV CODER_PIP=${CODER_VENV_DIR}/bin/pip
ENV CODER_PYTHON=${CODER_VENV_DIR}/bin/python

# 升级 pip 并安装 coder 分组依赖（版本在 pyproject.toml 中统一管理）
# 使用 uv 的 --python 选项在指定 venv 中安装依赖，确保版本一致性
RUN --mount=type=cache,target=/root/.cache \
    ${CODER_PIP} install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ && \
    uv pip install --python ${CODER_PYTHON} --group coder -i https://mirrors.aliyun.com/pypi/simple/

# 设定中文/UTF-8 本地化
RUN sed -i 's/# zh_CN.UTF-8 UTF-8/zh_CN.UTF-8 UTF-8/' /etc/locale.gen && locale-gen
ENV LANG=zh_CN.UTF-8 \
    LANGUAGE=zh_CN:zh \
    LC_ALL=zh_CN.UTF-8

    
# 复制项目代码（这一步改变不会影响上面的依赖安装和 venv 构建）
COPY . .

ENV PYTHONPATH=/app
ENV SOFFICE_BIN=/usr/bin/soffice
ENV GRAPHVIZ_DOT_PATH=/usr/bin/dot

ENV http_proxy= \
    https_proxy= \
    HTTP_PROXY= \
    HTTPS_PROXY=

EXPOSE 6300
ENV GUNICORN_RUNNING=true
CMD ["uv", "run","--no-sync", "gunicorn", "-w", "6", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "1200", "src.main:app", "--bind", "0.0.0.0:6300"]

