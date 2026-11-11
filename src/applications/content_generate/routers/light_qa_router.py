"""轻量溯源问答路由模块

目标：
- 只新增接口，不影响现有 content_generate 功能与路由
- 仅做“回答 + 溯源(citations)”：不走 planner/coder，不启用滑动窗口/上下文压缩
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from pydantic.alias_generators import to_camel

from src.applications.rag.schemas import BatchQueryItemHttp
from src.agents.content_generate_agent.models import FinalAnswer
from src.agents.content_generate_agent.nodes import reporter_core
from src.agents.content_generate_agent.utils import (
    apply_prompt_template,
    build_prompt_state,
    count_tokens,
    get_answer_language_instruction,
    get_localized_message,
    judge_simple_question,
    resolve_answer_locale,
)
from src.agents.content_generate_agent.config import create_default_configs
from loguru import logger


router = APIRouter(prefix="/api/content_generate_light", tags=["内容生成"])


class LightCitedQaRequest(BaseModel):
    """轻量溯源问答请求（单问题 + RAG 直出）"""

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    question_description: str = Field(..., description="问题文本（QuestionTask.description）")
    question_type: Literal["text", "text-json", "text-json-citation"] = Field(..., description="问题类型，支持 text/text-json/text-json-citation（QuestionTask.type）")
    question_prompt: str = Field("", description="可选提示信息（QuestionTask.prompt），将注入 reporter_instruction")

    rag_items: List[BatchQueryItemHttp] = Field(default_factory=list, description="RAG 批量查询结果项列表（HTTP 形态）")

    answer_lang: str = Field("zh_CN", description="回答语言 locale，如 zh_CN / en_US")
    model_id: Optional[str] = Field(None, description="模型ID；不传则根据问题类型自动选择：text类型默认使用微调模型，其他类型使用系统默认模型")
    enable_sliding_windows: bool = Field(
        default=True,
        description="是否允许 reporter_core 在超长上下文时启用滑动窗口策略；关闭时超阈值将直接拒绝",
    )


def _build_reporter_inputs_from_rag_items(
    rag_items: List[BatchQueryItemHttp],
) -> tuple[List[dict], Dict[str, Dict[str, Any]]]:
    """将 BatchQueryItemHttp 转换为 reporter_core 所需的 content 与 docid_to_source。"""
    content: List[dict] = []
    docid_to_source: Dict[str, Dict[str, Any]] = {}

    for idx, item in enumerate(rag_items, start=1):
        doc_id = f"doc_{idx}"
        text = (item.text or "").strip()
        line_str = f"<doc id={doc_id}>\n{text}\n</doc>"
        content.append(
            {
                "doc_id": doc_id,
                "text": text,
                "line_str": line_str,
            }
        )
        docid_to_source[doc_id] = {
            "element_ids": list(item.elementIds or []),
            "file_id": str(item.fileId or ""),
        }

    return content, docid_to_source


def _validate_no_sliding_window(
    *,
    question: str,
    content: List[dict],
    reporter_instruction: str,
    answer_lang: str,
) -> None:
    """轻量接口约束：不启用滑动窗口；超过阈值直接拒绝。"""
    cfg = create_default_configs()
    ctx_limit = cfg.context_token_limit

    content_full = "\n\n".join([c.get("line_str", "") for c in content])
    simple_answer = judge_simple_question(reporter_instruction)

    answer_language_instruction = get_answer_language_instruction(answer_lang)
    content_generation_example_detailed_answer = get_localized_message(
        "content_generation_example_detailed_answer",
        answer_lang,
    )
    content_generation_example_simple_answer = get_localized_message(
        "content_generation_example_simple_answer",
        answer_lang,
    )
    unanswerable_example_answer = get_localized_message("unanswerable_example_answer", answer_lang)

    system_prompt = apply_prompt_template(
        "content_generation_system",
        build_prompt_state(
            "content_generation_system",
            answer_lang,
            simple_answer=simple_answer,
            answer_language_instruction=answer_language_instruction,
            content_generation_example_detailed_answer=content_generation_example_detailed_answer,
            content_generation_example_simple_answer=content_generation_example_simple_answer,
            unanswerable_example_answer=unanswerable_example_answer,
        ),
        locale=answer_lang,
    )
    human_prompt = apply_prompt_template(
        "content_generation_human",
        build_prompt_state(
            "content_generation_human",
            answer_lang,
            question=question,
            context_text=content_full,
            extra_instruction=reporter_instruction,
        ),
        locale=answer_lang,
    )

    total_tokens = count_tokens(system_prompt) + count_tokens(human_prompt)
    if total_tokens > ctx_limit:
        raise HTTPException(
            status_code=400,
            detail=(
                f"上下文过长（约 {total_tokens} tokens，阈值 {ctx_limit}）。"
                "轻量溯源问答接口不启用滑动窗口/上下文压缩，请减少 ragItems 数量或截断 text。"
            ),
        )


@router.post("/answer_with_citations", response_model=FinalAnswer)
async def answer_with_citations(req: LightCitedQaRequest) -> FinalAnswer:
    """轻量溯源问答：单问题 + RAG 直出 ragItems -> FinalAnswer(citations)。"""
    answer_lang = resolve_answer_locale(req.answer_lang)

    question = (req.question_description or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="questionDescription 不能为空")

    if req.question_type not in ("text", "text-json", "text-json-citation"):
        raise HTTPException(status_code=400, detail="questionType 仅支持 text/text-json/text-json-citation")

    reporter_instruction = (req.question_prompt or "").strip()
    content, docid_to_source = _build_reporter_inputs_from_rag_items(req.rag_items or [])

    # 轻量接口：可选禁用滑动窗口（禁用时超阈值直接拒绝；启用时由 reporter_core 自行处理）
    if not req.enable_sliding_windows:
        _validate_no_sliding_window(
            question=question,
            content=content,
            reporter_instruction=reporter_instruction,
            answer_lang=answer_lang,
        )

    if req.question_type == "text-json-citation":
        reporter_template = "text_json_citation"
    elif req.question_type == "text-json":
        reporter_template = "text_json"
    else:
        reporter_template = "default"

    cfg = create_default_configs()

    if req.model_id is not None:
        if req.question_type == "text" and cfg.reporter_text_model_id:
            logger.warning("light_qa: text类型问题传入了model_id={}，默认微调模型({})将被覆盖", req.model_id, cfg.reporter_text_model_id)
        elif req.question_type == "text-json-citation" and cfg.reporter_text_json_citation_model_id:
            logger.warning("light_qa: text-json-citation类型问题传入了model_id={}，默认微调模型({})将被覆盖", req.model_id, cfg.reporter_text_json_citation_model_id)
        effective_model_id = req.model_id
    elif req.question_type == "text" and cfg.reporter_text_model_id:
        effective_model_id = cfg.reporter_text_model_id
    elif req.question_type == "text-json-citation" and cfg.reporter_text_json_citation_model_id:
        effective_model_id = cfg.reporter_text_json_citation_model_id
    else:
        effective_model_id = "" # 默认值 应会使用default model

    logger.info(
        "light_qa: question_len={}, rag_items={}, answer_lang={}, reporter_template={}, effective_model_id={}",
        len(question), len(content), answer_lang, reporter_template, effective_model_id,
    )

    return await reporter_core(
        question=question,
        content=content,
        coder_output=[],
        docid_to_source=docid_to_source,
        model_id=effective_model_id,
        reporter_instruction=reporter_instruction,
        reporter_template=reporter_template,
        history_messages=None,
        answer_lang=answer_lang,
        dependency_context="",
    )

