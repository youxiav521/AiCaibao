"""
Policy Check Agent - 提示词工具模块

提供提示词模板加载、渲染功能以及图片读取功能
"""
import os
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from jinja2 import Environment, FileSystemLoader
from loguru import logger

# 缓存 Jinja2 Environment 对象
_jinja_env: Environment | None = None

# 用于异步读取文件的线程池
_read_file_executor = ThreadPoolExecutor(max_workers=4)


def read_image_as_base64(relative_path: str, base_dir: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """
    读取项目指定相对路径的图片，返回 base64 编码和 mime_type。

    适用于 FastAPI 异步环境，支持同步文件 IO 操作。

    Args:
        relative_path: 图片相对路径（相对于项目根目录）
        base_dir: 可选的基准目录，默认为项目根目录

    Returns:
        (base64编码字符串, mime_type) 元组，失败返回 None
    """
    try:
        # 处理 Windows 路径分隔符
        relative_path = relative_path.replace('\\', '/')

        if base_dir is None:
            # 获取项目根目录（向上两级）
            current_dir = Path(__file__).parent
            base_dir = str(current_dir.parent.parent.parent)
        else:
            # 处理 Unix 风格路径（以 / 开头）在 Windows 上被当作绝对路径的问题
            base_dir = base_dir.replace('\\', '/').lstrip('/')

        # 标准化路径 - 使用 Path 处理跨平台问题
        file_path = Path(base_dir) / relative_path

        # 安全检查：确保路径在基准目录内
        file_path = file_path.resolve()
        base_path = Path(base_dir).resolve()
        if not str(file_path).startswith(str(base_path)):
            logger.warning("路径安全检查失败: {} 不在基准目录 {} 内", file_path, base_path)
            return None

        if not file_path.exists():
            logger.warning("图片文件不存在: {}", file_path)
            return None

        # 检测图片类型
        suffix = file_path.suffix.lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp',
        }
        mime_type = mime_types.get(suffix, 'image/jpeg')

        # 读取并编码
        with open(file_path, 'rb') as f:
            image_data = f.read()

        base64_data = base64.b64encode(image_data).decode('utf-8')

        logger.info("成功读取图片: {}, 大小: {} bytes, 类型: {}", file_path, len(image_data), mime_type)
        return (base64_data, mime_type)

    except Exception as e:
        logger.error("读取图片失败 {}: {}", relative_path, e)
        return None


async def read_image_async(relative_path: str, base_dir: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    异步读取图片，返回 (base64, mime_type) 元组。

    Args:
        relative_path: 图片相对路径
        base_dir: 可选的基准目录

    Returns:
        (base64编码字符串, mime_type) 元组，失败时 base64 为 None
    """
    import asyncio

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            _read_file_executor,
            read_image_as_base64,
            relative_path,
            base_dir
        )
        if result is None:
            return None, None
        return result
    except Exception as e:
        logger.error("异步读取图片失败 {}: {}", relative_path, e)
        return None, None


def _get_jinja_env() -> Environment:
    """
    获取 Jinja2 环境，加载当前目录下的 prompts 文件夹。
    """
    global _jinja_env
    if _jinja_env is None:
        current_dir = Path(__file__).parent
        prompts_dir = current_dir / "prompts"
        _jinja_env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            autoescape=False,
            keep_trailing_newline=True,
        )
    return _jinja_env


def apply_prompt_template(prompt_name: str, state: Dict[str, Any] | None = None) -> str:
    """
    使用 Jinja2 渲染 prompts 目录下的 .md 模板，返回渲染后的字符串。

    Args:
        prompt_name: 模板文件名（不含 .md）
        state: 用于渲染模板的变量字典（可选）

    Returns:
        渲染后的提示词字符串

    Raises:
        FileNotFoundError: 模板文件不存在
        ValueError: 模板渲染失败
    """
    try:
        env = _get_jinja_env()
        template = env.get_template(f"{prompt_name}.md")
        return template.render(**(state or {}))
    except FileNotFoundError:
        current_dir = Path(__file__).parent
        raise FileNotFoundError(
            f"Prompt模板文件不存在: {current_dir / 'prompts' / (prompt_name + '.md')}"
        )
    except Exception as e:
        raise ValueError(f"模板渲染失败: {e}")


# 便捷函数 - 用于直接获取各个子智能体的提示词
def get_system_prompt(state: Dict[str, Any] | None = None) -> str:
    """获取主智能体系统提示词"""
    return apply_prompt_template("system", state)


def get_document_checker_prompt(
    resource_repository: Dict[str, Any] | None = None,
    controls: List[Dict[str, Any]] | None = None,
) -> str:
    """获取文档检查子智能体提示词"""
    return apply_prompt_template("document_checker", {
        "resource_repository": resource_repository,
        "controls": controls,
    })


def get_multimodal_checker_prompt(
    resource_repository: Dict[str, Any] | None = None,
    controls: List[Dict[str, Any]] | None = None,
) -> str:
    """获取多模态检查子智能体提示词"""
    return apply_prompt_template("multimodal_checker", {
        "resource_repository": resource_repository,
        "controls": controls,
    })

def __main__():
    result = read_image_as_base64(
        "00af6699b3064e8b821c926371a2a34dqlOlvj5WSX/52b2ce0b6d32c27b9b5c949194e8f0391edf3fd38750026e97e097ea7063c4d7.jpg",
        "storage/mineru_output/images"
    )
    print(result[:100] + "..." if result else result)

if __name__ == "__main__":
    __main__()
