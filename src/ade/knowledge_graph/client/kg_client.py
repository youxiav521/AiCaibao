from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from src.utils.model_factory import ModelFactory
from src.ade.knowledge_graph.client.prompt import (
    ENTITY_EXTRACTION_SYSTEM_PROMPT,
    ENTITY_EXTRACT_USER_PROMPT,
)
from src.ade.knowledge_graph.client.schema.extraction_schema import KGExtractionResult
from src.utils.json_utils import dumps as json_dumps


class KGClient:
    """知识图谱LLM客户端，负责调用模型提取实体和关系"""

    def __init__(self):
        # function_calling 强制结构化输出：该模型（及部分大模型）不支持 response_format(json_schema)，
        # 必须显式指定 method="function_calling" 走 tools/tool_choice。
        model = ModelFactory.get_model(key="model_deepseek_v4_flash")
        self.model = model.with_structured_output(
            KGExtractionResult,
            method="function_calling",
            include_raw=True,
        )

    async def extract(
        self,
        content: str,
        project_name: str = "",
        document_metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """调用LLM获取 function-calling 原始包络，由服务层逐条校验。

        Args:
            content: 文档文本内容
            project_name: 项目节点名称，用于替换prompt中的{project_name}占位符
            document_metadata: 已过滤的文件夹名和当前piece来源文件名

        Returns:
            dict: 包含 raw、parsed 和 parsing_error 的结构化输出包络
        """
        # 系统提示中含示例 JSON 的花括号，不能用 .format()，仅替换项目名称占位符。
        system_content = ENTITY_EXTRACTION_SYSTEM_PROMPT.replace(
            "{project_name}", project_name
        )
        user_prompt = ENTITY_EXTRACT_USER_PROMPT.format(
            document_metadata=json_dumps(document_metadata or {}),
            input_text=content,
        )
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_prompt),
        ]
        result = await self.model.ainvoke(messages)
        if not isinstance(result, dict):
            return {
                "raw": result,
                "parsed": None,
                "parsing_error": TypeError("结构化输出包络不是对象"),
            }
        logger.info(
            "KGClient | LLM调用完成 | has_raw={} parsed={} parsing_error={}",
            result.get("raw") is not None,
            result.get("parsed") is not None,
            result.get("parsing_error") is not None,
        )
        return result
