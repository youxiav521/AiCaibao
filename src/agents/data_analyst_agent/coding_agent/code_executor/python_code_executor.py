import asyncio
import threading
import sys
import os
import time
import tempfile
import subprocess
import traceback
from pathlib import Path
import aiofiles

# 兼容相对导入和直接运行
try:
    from ..models.python_code_model import PythonCodeModel
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))
    from src.agents.data_analyst_agent.coding_agent.models.python_code_model import PythonCodeModel

from loguru import logger


# 环境执行错误：环境/系统问题，不应该让 agent 重试
class EnvironmentExecutionError(Exception):
    """执行环境/系统错误，重试无意义甚至有害"""
    pass


# 模块级单例，确保所有请求共享同一个实例
_instance = None
_lock = threading.Lock()


def _get_executor_instance():
    """线程安全的单例获取方法"""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PythonCodeExecutor()
    return _instance


class PythonCodeExecutor:
    """
    基于子进程的Python代码异步执行器，使用 .coder_venv 独立环境执行代码。
    """
    work_dir: str
    venv_dir: Path
    python_path: Path

    def __init__(self):
        # 使用当前文件所在目录作为工作目录
        self.work_dir = os.path.abspath(os.path.dirname(__file__))

        # 定位 .coder_venv 虚拟环境
        project_root = os.path.abspath(os.path.join(self.work_dir, "../../../../../"))
        self.venv_dir = Path(os.path.join(project_root, ".coder_venv"))
        self.project_root = project_root

        # 检查根目录的 venv 是否存在，不存在则回退到代码目录下的旧路径
        venv_cfg_path = self.venv_dir / "pyvenv.cfg"
        if not venv_cfg_path.exists():
            self.venv_dir = Path(os.path.join(self.work_dir, "coder_venv"))
            logger.info(f"代码执行初始化，使用旧路径 coder_venv: {self.venv_dir}")
        else:
            logger.info(f"代码执行初始化，使用新路径 .coder_venv: {self.venv_dir}")

        # 检查环境是否正常
        self._check_environment()

    def _check_environment(self):
        """
        检查执行环境是否正常：
        1. 虚拟环境目录必须存在
        2. Python 解释器可执行文件必须存在
        异常直接抛出 EnvironmentExecutionError
        """
        # 1. 检查虚拟环境配置文件
        venv_cfg_path = self.venv_dir / "pyvenv.cfg"
        if not venv_cfg_path.exists():
            raise EnvironmentExecutionError(
                f"虚拟环境不存在: {self.venv_dir}，请先配置好 .coder_venv 环境"
            )

        # 2. 获取虚拟环境的 python 路径（兼容 Windows 和 Linux/Mac）
        if os.name == "nt":
            self.python_path = self.venv_dir / "Scripts" / "python.exe"
        else:
            self.python_path = self.venv_dir / "bin" / "python"

        # 3. 检查 Python 解释器是否存在
        if not self.python_path.exists():
            raise EnvironmentExecutionError(
                f".coder_venv 的 Python 解释器不存在: {self.python_path}"
            )

    def _run_code_sync(self, code_path: str, timeout: int) -> dict:
        """
        同步方式执行代码，在线程池中调用以避免阻塞 event loop
        此方式兼容所有平台的 event loop（Windows/Linux/Mac）
        """
        try:
            # 设置环境变量，强制子进程使用 UTF-8 编码输出
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            result = subprocess.run(
                [str(self.python_path), code_path],
                cwd=self.work_dir,
                capture_output=True,
                text=False,  # 先获取 bytes，再统一用 UTF-8 解码
                timeout=timeout,
                env=env
            )

            # 统一使用 UTF-8 解码输出
            output = result.stdout.decode("utf-8", errors="replace")
            error = result.stderr.decode("utf-8", errors="replace")

            return {
                "success": result.returncode == 0,
                "output": output,
                "error": error
            }

        except subprocess.TimeoutExpired:
            raise asyncio.TimeoutError(f"执行超时 (timeout={timeout}s)")

    async def execute(self, code_model: PythonCodeModel, timeout=600) -> dict:
        """
        异步执行传入的PythonCodeModel中的代码。
        使用 .coder_venv 独立环境的 Python 解释器执行代码。
        参数:
            code_model (PythonCodeModel): 包含要执行的代码、导入和描述。
            timeout (int): 执行超时时间，默认10分钟。
        返回:
            dict: 包含执行结果或错误信息。
        """
        # 拼接完整代码
        full_code = ""
        if code_model.imports:
            full_code += code_model.imports + "\n"
        full_code += code_model.code

        # 添加路径配置代码，确保代码能找到当前目录的模块
        setup_code = f"""
import sys
import os
# 添加当前执行器目录到路径
executor_dir = r"{self.work_dir}"
sys.path.insert(0, executor_dir)
# 添加项目根目录到路径
sys.path.insert(0, r"{self.project_root}")

"""
        full_code = setup_code + full_code

        # 在代码开头添加编码声明
        if not full_code.startswith('# -*- coding:'):
            full_code = "# -*- coding: utf-8 -*-\n" + full_code

        # 记录开始时间
        start_time = time.time()

        # 创建临时文件存放代码
        temp_dir = os.path.join(self.project_root, "src/agents/data_analyst_agent/coding_agent/code_executor/tmp")
        os.makedirs(temp_dir, exist_ok=True)

        # 生成临时文件名，用 aiofiles 异步写入
        temp_code_path = tempfile.mktemp(suffix='.py', dir=temp_dir)
        async with aiofiles.open(temp_code_path, 'w', encoding='utf-8') as f:
            await f.write(full_code)

        try:
            # 使用 .coder_venv 的 Python 解释器执行代码
            # 通过 to_thread 在线程池中执行同步 subprocess，兼容所有平台的 event loop
            result = await asyncio.to_thread(
                self._run_code_sync,
                temp_code_path,
                timeout
            )

            output = result["output"]
            error = result["error"]
            success = result["success"]

            # 计算执行时间
            elapsed_time = time.time() - start_time

            # 错误分类处理：判断是否为需要 raise 的环境错误
            if not success:
                logger.info(f"执行代码失败，判断代码执行错误的原因：{error}")
                if self._is_raise_error(error):
                    # 环境错误直接 raise，不返回给 agent（重试无意义）
                    raise EnvironmentExecutionError("代码执行出现环境错误：" + error)

            return {
                "success": success,
                "output": output,
                "error": error,
                "elapsed_time": elapsed_time
            }

        except EnvironmentExecutionError:
            # _is_raise_error 错误直接向上抛出
            raise
        except asyncio.TimeoutError:
            logger.error(f"--code_executor--：执行超时 (timeout={timeout}s)")
            return {
                "success": False,
                "error": f"执行超时 (timeout={timeout}s)",
                "output": "",
                "elapsed_time": time.time() - start_time,
            }
        except Exception as e:
            logger.error(f"--code_executor--：代码执行异常，异常堆栈：{traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "elapsed_time": 0,
            }
        finally:
            if os.path.exists(temp_code_path):
                os.remove(temp_code_path)

    # 需要直接 raise 的错误关键词集合（环境/系统问题，重试无意义）
    _RAISE_ERROR_KEYWORDS = {
        # 缺包相关
        "modulenotfounderror", "no module named", "importerror",
        "cannot find module", "模块未找到", "找不到模块",
        # 权限问题
        "permission denied", "permissionerror", "权限不足",
        "权限被拒绝", "access is denied", "没有权限",
        # OOM
        "memoryerror", "out of memory", "oom",
        "内存不足", "内存溢出", "cannot allocate memory",
        # 网络问题
        "connectionerror", "network", "connection error"
        "连接超时", "unreachable", "连接被拒绝",
    }

    def _is_raise_error(self, error_msg: str) -> bool:
        """
        判断错误是否为需要直接 raise 的类型（环境/系统问题）。
        这类错误让 agent 重试是无意义的，应该直接抛出 EnvironmentExecutionError。
        """
        if not error_msg:
            return False

        error_lower = error_msg.lower()
        return any(keyword in error_lower for keyword in self._RAISE_ERROR_KEYWORDS)


if __name__ == "__main__":
    # __main__ 执行时使用绝对导入
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))
    from src.agents.data_analyst_agent.coding_agent.models.python_code_model import PythonCodeModel

    # 构造一个简单的PythonCodeModel实例
    code_model = PythonCodeModel(
        description="计算1+1的结果",
        imports="""
# 导入需要的库
import pandas as pd
import numpy as np
        """,
        code="""
print("1+1=", 1+1)
print("pandas 版本:", pd.__version__)
import sys
print("Python 路径:", sys.executable)
import os
print("工作目录:", os.getcwd())
        """
    )

    async def main():
        executor = PythonCodeExecutor()
        result = await executor.execute(code_model)
        print("execute执行结果：", result)

    asyncio.run(main())
