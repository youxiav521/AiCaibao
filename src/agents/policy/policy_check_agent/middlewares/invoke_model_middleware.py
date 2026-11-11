from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from src.agents.content_generate_agent.utils import _retry_with_backoff, count_tokens
from typing import Callable
from langgraph.runtime import Runtime
from typing import Any
from loguru import logger


class InvokeModelMiddleware(AgentMiddleware):
    def __init__(self, max_retries: int = 5,model_id: str = "model_bailian_policy_check"):
        super().__init__()
        self.max_retries = max_retries
        self.model_id = model_id

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        # TODO 可以在这里做调用模型前的token检查
        messages_str = str(state['messages'])
        if count_tokens(messages_str) > 220000:
            logger.warning(f"模型调用token超过220000")
            # 截断消息？或者重新生成消息？
            return None
        # logger.info(f"About to call model with {len(state['messages'])} messages")
        return None

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # handler 是异步的，需要包装成 async 函数
        async def call_handler():
            return await handler(request)
        # 控制模型调用的retrys
        return await _retry_with_backoff(call_handler, max_retries=self.max_retries, model_id=self.model_id)