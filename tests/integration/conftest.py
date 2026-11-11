"""
MinerU API 集成测试公共配置
所有 MinerU 相关测试共享的 fixtures 和常量定义
"""
import os
import sys
import subprocess
import time
from pathlib import Path

import pytest
import httpx
from httpx import ASGITransport
from dotenv import load_dotenv

# ─── 常量 ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.parent
# 确保项目根目录在 Python 路径中
sys.path.insert(0, str(PROJECT_ROOT))

# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def project_root():
    """项目根目录路径"""
    return PROJECT_ROOT

@pytest.fixture(scope="module")
def load_env(project_root):
    """加载环境变量，模块级别只加载一次"""
    load_dotenv(project_root / ".env")
    # 加载测试专用环境变量以覆盖
    load_dotenv(project_root / "tests" / ".env_test", override=True)
    



@pytest.fixture(scope="module")
def arq_worker(project_root, load_env):
    """以子进程方式启动 ARQ Worker，测试模块结束后自动终止。

    模块级别 fixture，整个测试模块共享同一个 Worker 实例
    显式传入 PYTHONPATH 以保证子进程能 import src.*，
    因为测试时 pytest 的 pythonpath 配置对子进程不生效。
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)

    proc = subprocess.Popen(
        [sys.executable, "-m", "arq", "src.worker.WorkerSettings"],
        cwd=project_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # 等待 Worker 连接 Redis 并进入监听状态
    time.sleep(3)

    if proc.poll() is not None:
        output = proc.stdout.read().decode(errors="replace")
        pytest.fail(f"ARQ Worker 启动失败:\n{output}")

    yield proc

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture
async def client(minimal_app):
    """每测试独立的 ASGI 客户端（ASGITransport 无需绑定端口）。"""
    async with httpx.AsyncClient(
        transport=ASGITransport(app=minimal_app),
        base_url="http://testserver",
    ) as ac:
        yield ac


@pytest.fixture(scope="session")
def sample_pdf(project_root):
    """最小合规 PDF 字节，用于上传测试。
    会话级别 fixture，只读取一次文件内容
    """
    file_path = project_root / "tests" / "files" / "test.pdf"
    with open(file_path, "rb") as f:
        return f.read()

