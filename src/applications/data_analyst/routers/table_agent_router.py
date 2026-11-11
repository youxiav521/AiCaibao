from email import message
import os
import time
from langchain_core import documents
from typing_extensions import Any, Dict, List, Optional
from fastapi import APIRouter, Form, Header
from src.agents.content_generate_agent.input_models import DocumentItems, Item
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse, StreamingResponse
import json

from ....agents.data_analyst_agent.coding_agent.graph.table_agent.fill_table_graph.fill_table_agent import FillTableAgentService
from src.agents.content_generate_agent.table_answer_components.answer_graph.answer_table_agent import AnswerTableAgentService
from src.agents.content_generate_agent.utils import get_localized_message, resolve_answer_locale
from loguru import logger
table_agent_router = APIRouter(prefix="/api/test/table_agent", tags=["表格相关(测试使用接口)"]) 

# 模块级Service单例，避免每次请求重新实例化
_fill_table_service = FillTableAgentService()
_answer_table_service = AnswerTableAgentService() 

class FillTableRequest(BaseModel):
    question: str = Field(..., description="问题描述")
    prompt: Optional[str]=Field(None, description="用户提示词")
    documents: Optional[List[dict]] = Field(default=[], description="与问题相关的数据片段列表")
    model_id: Optional[str] = Field(default="model_bailian_80b", description="模型ID")

class FillTableResponse(BaseModel):
    code: int = Field(..., description="返回码")
    error_message: Optional[str] = Field(None, description="错误信息")
    data: Optional[Any] = Field(None, description="数据")

@table_agent_router.post(
    "/fill", 
    response_model=FillTableResponse,
    summary="填充问题表格测试使用接口",
    description="填充问题表格测试使用接口",
    responses={
        200: {
            "description": "执行成功",
            "model": FillTableResponse
        },
    }
)
async def call_fill_table(
    request: FillTableRequest,
    accept_language: Optional[str] = Header(None, alias="Accept-Language"),
):
    answer_lang = resolve_answer_locale(accept_language)
    try:
        result = await _fill_table_service.run_fill_table_with_token_limit(
            question=request.question,
            prompt=request.prompt,
            documents=request.documents,
            model_id=request.model_id,
            answer_lang=answer_lang,
        )
        
        return FillTableResponse(
            code=0,
            error_message=get_localized_message("api_operation_success", answer_lang),
            data=result,
        )
    except Exception as e:
        logger.error(f"调用fill_table_agent时发生异常: {e}")
        return FillTableResponse(
            code=1,
            error_message=get_localized_message("api_operation_failed", answer_lang, error=str(e)),
            data=None,
        )

class Document(BaseModel):
    file_name: str = Field(alias="fileName")
    content: List[Dict[str, Any]]

class AnswerTableRequest(BaseModel):
    content_list: List[Document] = Field(..., description="需要处理的内容列表")
    question_tasks: List[Dict[str, Any]] = Field(..., description="问题任务描述")
    model_id: Optional[str] = Field("model_bailian_80b", description="模型ID")

class AnswerTableResponse(BaseModel):
    code: int = Field(..., description="返回码")
    error_message: Optional[str] = Field(None, description="错误信息")
    data: Optional[Any] = Field(None, description="数据")

@table_agent_router.post(
    "/answer_table",
    response_model=AnswerTableResponse,
    summary="表格问题自动回答接口",
    description="表格 Agent 用于自动回答复杂表格数据分析任务",
    responses={
        200: {
            "description": "执行成功",
            "model": AnswerTableResponse
        },
    }
)
async def answer_table(
    request: AnswerTableRequest,
    accept_language: Optional[str] = Header(None, alias="Accept-Language"),
):
    answer_lang = resolve_answer_locale(accept_language)

    async def event_stream_generator(merged_content, question_tasks, model_id, answer_lang):
        try:
            typed_items: List[Item] = DocumentItems.model_validate(merged_content).root
        except Exception as e:
            logger.error(f"内容列表校验失败：{e}")
            error_payload = {
                "code": 1,
                "error_message": get_localized_message("api_stream_error", answer_lang, error=str(e)),
                "data": None,
            }
            yield f"data: {json.dumps(error_payload)}\n\n"
            return
        try:
            async for event in _answer_table_service.table_answering(
                typed_items=typed_items,
                question_tasks=question_tasks,
                model_id=model_id,
                answer_lang=answer_lang,
            ):
                # 每次返回一个事件（假设event可dict化）
                yield f"data: {json.dumps(event.model_dump() if hasattr(event, 'model_dump') else event.__dict__)}\n\n"
        except Exception as e:
            logger.error(f"调用answer_table_agent时发生异常（流式生成）: {e}")
            import traceback
            logger.error(f"调用answer_table_agent时发生异常: {e}\n堆栈信息:\n{traceback.format_exc()}")
            # 发生错误也向前端推送一条异常事件
            error_payload = {
                "code": 1,
                "error_message": get_localized_message("api_stream_error", answer_lang, error=str(e)),
                "data": None,
            }
            yield f"data: {json.dumps(error_payload)}\n\n"

    merged_content: List[Dict[str, Any]] = []
    if request.content_list:
        for doc in request.content_list:
            if doc.content:
                merged_content.extend(doc.content)
                logger.info(f"添加文档 {doc.file_name} 的 {len(doc.content)} 个内容片段")
    
    return StreamingResponse(
        event_stream_generator(merged_content, request.question_tasks, request.model_id, answer_lang),
        media_type="text/event-stream"
    )
