"""
Policy Check Agent - 子智能体定义

定义用于文档检查和多模态检查的子智能体
"""
from typing import Any, List, Optional
from langchain.agents import create_agent
from loguru import logger

from ..utils import (
    get_document_checker_prompt,
    get_multimodal_checker_prompt,
)


def create_document_checker_subagent(
    model: Any,
    base_tools: List[Any],
    middlewares: Optional[List[Any]] = None,
) -> Any:
    """
    创建文档内容检查子智能体

    负责检查文档内容是否符合规则要求：
    1. 读取材料库中的文件内容
    2. 根据控制项的 logicDescription 检查相关内容是否存在
    3. 识别缺失或不符的内容
    4. 提取相关证据

    Args:
        model: 语言模型实例
        base_tools: 工具列表
        middlewares: 可选的中间件列表

    Returns:
        配置好的子智能体实例
    """
    system_prompt = get_document_checker_prompt()

    agent = create_agent(
        model=model,
        tools=base_tools,
        system_prompt=system_prompt,
        middleware=middlewares
    )

    return agent


def create_multimodal_checker_subagent(
    model: Any,
    base_tools: List[Any],
    middlewares: Optional[List[Any]] = None,
) -> Any:
    """
    创建多模态检查子智能体（图片内容分析）

    负责检查图片内容是否符合规则要求：
    1. 识别材料库中的图片内容
    2. 分析图片中的关键信息
    3. 判断图片内容是否符合控制项要求

    Args:
        model: 语言模型实例
        base_tools: 工具列表
        middlewares: 可选的中间件列表

    Returns:
        配置好的子智能体实例
    """
    system_prompt = get_multimodal_checker_prompt()

    agent = create_agent(
        model=model,
        tools=base_tools,
        system_prompt=system_prompt,
        middleware=middlewares
    )

    return agent
