from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from src.agents.content_generate_agent.utils import _retry_with_backoff, count_tokens
from typing import Callable
from langgraph.runtime import Runtime
from typing import Any
from loguru import logger


class SubagentMiddleware(AgentMiddleware):
    """
    用于处理subagent（子智能体）调用后的输出增强/更改。
    例如：对LLM结果输出做格式化或补充字段，增强结构化输出，修正不规范内容等。
    """
    def __init__(self, max_retries: int = 5, model_id: str = "model_bailian_policy_check"):
        super().__init__()
        self.max_retries = max_retries
        self.model_id = model_id

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        # 可以在这里做调用模型前的token检查
        messages_str = str(state['messages'])
        if count_tokens(messages_str) > 220000:
            logger.warning(f"模型调用token超过220000")
            # 可以在此处对消息进行截断或重新生成
            return None
        return None

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """
        调用subagent模型，并在模型调用后对输出做规范化处理。
        """
        # 先进行模型调用，获取原始输出
        async def call_handler():
            return await handler(request)

        # 控制模型调用重试次数
        response: ModelResponse = await _retry_with_backoff(call_handler, max_retries=self.max_retries, model_id=self.model_id)

        # 对subagent的输出内容进行后处理（示例：格式化、增加结构化信息等）
        # 这里只做演示，可根据具体业务需求定制
        try:
            if hasattr(response, 'content') and isinstance(response.content, str):
                # 例如：对输出内容做json结构检查，也可以修正常见格式错误
                import json
                try:
                    # 测试是否为有效json格式，如果不是，可做适当修正处理
                    parsed = json.loads(response.content)
                    # 可根据业务定制进一步的检查或结构补全
                    # 例如：确保必要字段存在
                    required_fields = ["result", "reason"]
                    for field in required_fields:
                        if field not in parsed:
                            parsed[field] = ""
                    # 如需补充文件标识等信息可在此修改
                    response.content = json.dumps(parsed, ensure_ascii=False)
                except Exception as e:
                    logger.warning(f"子智能体输出非标准JSON，可按需修正。错误信息：{e}")
                    # 可选择添加一定的格式修正逻辑
        except Exception as e:
            logger.error(f"SubagentMiddleware处理输出异常: {e}")

        return response