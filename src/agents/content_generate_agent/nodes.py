from __future__ import annotations

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from langgraph.types import Command
from src.base.langfuse_observe import trace_span

from .utils import (
    apply_prompt_template,
    ainvoke_llm_with_messages,
    ainvoke_llm_with_messages_structured_output,
    ainvoke_llm_with_prompt,
    ainvoke_llm_with_structured_output,
    build_prompt_state,
    count_tokens,
    ensure_within_token_limit,
    get_answer_language_instruction,
    get_localized_message,
    judge_simple_question,
    truncate_to_token_limit,
    log_error_type_and_message,
)
from .config import create_default_configs

_CFG = create_default_configs()
_CTX_LIMIT = _CFG.context_token_limit
from pydantic import ValidationError
from .state import ContentGenState, PlannerResponse, QuotedAnswer
from .models import FinalAnswer, Citation
from typing import Tuple
from loguru import logger
from src.utils import StopWatch

# ============= 滑动窗口工具函数 =============

def _create_sliding_windows(
    content: List[dict],
    coder_output: List[str],
    question: str,
    max_tokens_per_window: int,
    answer_lang: str = "zh_CN",
    overlap_ratio: float = 0.0
) -> List[Dict[str, Any]]:
    """
    创建滑动窗口，将content和coder_output分成多个窗口。
    
    Args:
        content: 内容列表，每个元素包含 doc_id, text, line_str
        coder_output: coder输出列表
        question: 问题文本（用于计算prompt模板的token开销）
        max_tokens_per_window: 每个窗口的最大token数
        overlap_ratio: 窗口重叠比例（0.0-1.0），默认0.0（0%重叠）
        
    Returns:
        窗口列表，每个窗口包含：
        - window_id: 窗口编号
        - content_items: 该窗口包含的content项
        - coder_output_items: 该窗口包含的coder_output项
        - content_text: 窗口的content文本
        - coder_output_text: 窗口的coder_output文本
    """
    # 估算prompt模板的token开销（question + 模板固定部分）
    template_overhead = count_tokens(apply_prompt_template(
        "planner_new",
        {
            "question": question,
            "content": "",
            "coder_output": ""
        },
        locale=answer_lang,
    ))
    
    # 实际可用于内容的token数
    available_tokens = max_tokens_per_window - template_overhead
    logger.info(f"可用于内容的token数: {available_tokens}")
    
    # 将所有内容项合并成一个列表，保留原始索引和类型信息
    all_items = []
    for item in content:
        all_items.append({
            "type": "content",
            "item": item,
            "text": item.get("line_str", "")
        })
    for coder_text in coder_output:
        all_items.append({
            "type": "coder_output",
            "item": coder_text,
            "text": coder_text
        })
    
    if not all_items:
        return []
    
    windows = []
    window_id = 0
    start_idx = 0
    
    while start_idx < len(all_items):
        window_items = []
        current_tokens = 0
        idx = start_idx
        
        # 构建当前窗口
        while idx < len(all_items):
            item = all_items[idx]
            item_tokens = count_tokens(item["text"])
            
            # 如果添加这个item会超过限制，且窗口不为空，则停止
            if current_tokens + item_tokens > available_tokens and window_items:
                break
            
            window_items.append(item)
            current_tokens += item_tokens
            idx += 1
        
        if not window_items:
            # 如果单个item就超过限制，忽略这个item并警告
            logger.error(f"窗口 {window_id} 忽略单个item: {all_items[start_idx]['text']}，因为单个item超过上下文长度限制:{available_tokens} tokens")
            start_idx += 1
        else:
            # 计算下一个窗口的起始位置（考虑可能重叠）
            overlap_size = max(0, int(len(window_items) * overlap_ratio)) # 重叠大小为0时，不重叠
            start_idx = max(start_idx + 1, idx - overlap_size)
        
        # 分离content和coder_output
        content_items = [item["item"] for item in window_items if item["type"] == "content"]
        coder_output_items = [item["item"] for item in window_items if item["type"] == "coder_output"]
        content_text = "\n\n".join([item["text"] for item in window_items if item["type"] == "content"])
        coder_output_text = "\n\n".join([item["text"] for item in window_items if item["type"] == "coder_output"])
        
        windows.append({
            "window_id": window_id,
            "content_items": content_items,
            "coder_output_items": coder_output_items,
            "content_text": content_text,
            "coder_output_text": coder_output_text,
        })
        
        window_id += 1
        
        # 防止无限循环
        if start_idx >= len(all_items):
            break
    
    logger.info(f"创建了 {len(windows)} 个滑动窗口（每个窗口最多 {max_tokens_per_window} tokens，重叠比例 {overlap_ratio}）")
    return windows


async def _planner_with_sliding_windows(
    question: str,
    content: List[dict],
    coder_output: List[str],
    model_id: str,
    answer_lang: str = "zh_CN",
    max_tokens_per_window: int = None
) -> PlannerResponse:
    """
    使用滑动窗口策略调用planner。
    
    Args:
        question: 问题文本
        content: 内容列表
        coder_output: coder输出列表
        model_id: 模型ID
        max_tokens_per_window: 每个窗口的最大token数，如果为None则使用_CTX_LIMIT
        
    Returns:
        合并后的PlannerResponse
    """
    if max_tokens_per_window is None:
        max_tokens_per_window = _CTX_LIMIT
    
    # 创建滑动窗口
    windows = _create_sliding_windows(
        content,
        coder_output,
        question,
        max_tokens_per_window,
        answer_lang=answer_lang,
    )
    
    if not windows:
        # 如果没有窗口，返回默认响应
        logger.error("无法创建滑动窗口，上下文为空")
        return PlannerResponse(
            needs_computation=False,
            reasoning="Unable to create sliding windows because the context is empty."
            if answer_lang == "en_US"
            else "无法创建滑动窗口，上下文为空",
            computation_prompt="",
            computation_input=[]
        )
    
    # 对每个窗口调用planner TODO?: 并行
    window_responses = []
    for window in windows:
        prompt = apply_prompt_template(
            "planner_new",
            {
                "question": question,
                "content": window["content_text"],
                "coder_output": window["coder_output_text"]
            },
            locale=answer_lang,
        )
        
        try:
            response = await ainvoke_llm_with_structured_output(
                prompt, 
                model_id=model_id, 
                response_type=PlannerResponse
            )
            window_responses.append(response)
            logger.info(f"窗口 {window['window_id']} planner响应: needs_computation={response.needs_computation}")
        except Exception as e:
            log_error_type_and_message(
                logger=logger,
                context=f"窗口 {window['window_id']} planner调用失败",
                e=e,
                exc_info=False,
                max_message_chars=200,
            )
            # 失败时使用默认响应
            window_responses.append(PlannerResponse(
                needs_computation=False,
                reasoning=(
                    f"Window {window['window_id']} processing failed: {str(e)}"
                    if answer_lang == "en_US"
                    else f"窗口 {window['window_id']} 处理失败: {str(e)}"
                ),
                computation_prompt="",
                computation_input=[]
            ))
    
    # 合并所有窗口的响应
    # needs_computation: 如果任何一个窗口需要，则整体需要
    merged_needs_computation = any(r.needs_computation for r in window_responses)
    
    # 情况1：所有窗口都认为不需要计算时 → 简单合并所有内容（包括reasoning）
    # 情况2：有一部分窗口认为需要计算时 → 仅合并认为需要计算的窗口
    if not merged_needs_computation:
        # 情况1：所有窗口都不需要计算
        # 合并所有窗口的 reasoning 作为"整体解释"
        merged_reasoning = "\n\n".join(
            [
                f"[窗口 {i+1}]: {r.reasoning}"
                for i, r in enumerate(window_responses)
                if r.reasoning
            ]
        )
        # 不需要计算时，computation_prompt 和 computation_input 为空
        merged_computation_prompt = ""
        merged_computation_input = []
    else:
        # 情况2：有一部分窗口认为需要计算
        # 仅合并需要计算的窗口
        computation_responses = [r for r in window_responses if r.needs_computation]
        
        # reasoning: 合并所有需要计算窗口的 reasoning
        merged_reasoning = "\n\n".join(
            [
                f"[窗口 {i+1}]: {r.reasoning}"
                for i, r in enumerate(computation_responses)
                if r.reasoning
            ]
        )
        
        # computation_prompt: 合并所有需要计算窗口的 prompt
        prompts = [r.computation_prompt for r in computation_responses if r.computation_prompt]
        merged_computation_prompt = "\n\n".join(prompts) if prompts else ""
        
        # computation_input: 合并所有需要计算窗口的 input（去重）
        seen_inputs = set()
        merged_computation_input = []
        for response in computation_responses:
            for inp in response.computation_input:
                # inp 现在是字符串格式的 doc_id
                if inp not in seen_inputs:
                    seen_inputs.add(inp)
                    merged_computation_input.append(inp)
    
    logger.info(f"合并了 {len(window_responses)} 个窗口的planner响应: needs_computation={merged_needs_computation}, "
                f"computation_input数量={len(merged_computation_input)}")
    
    return PlannerResponse(
        needs_computation=merged_needs_computation,
        reasoning=merged_reasoning,
        computation_prompt=merged_computation_prompt,
        computation_input=merged_computation_input
    )


def _coerce_quoted_answer_from_validation_error(e: ValidationError) -> Optional[QuotedAnswer]:
    """
    当模型在「请简单回答」场景下只返回数字/短句时，结构化输出会解析为 int/str 导致 QuotedAnswer 校验失败。
    本函数提供兜底逻辑：若错误为单次 model_type（期望对象、得到基本类型），则将原始值包装为 QuotedAnswer 并返回。
    """
    errs = e.errors()
    if len(errs) != 1:
        return None
    err = errs[0]
    if err.get("type") != "model_type":
        return None
    inp = err.get("input")
    if inp is None:
        return None
    if isinstance(inp, (str, int, float, bool)):
        logger.info(f"简单回答时模型只返回数字/短句，导致校验失败，手动包装为 QuotedAnswer: {str(inp)}")
        return QuotedAnswer(is_answerable=True, answer=str(inp))
    return None


def _parse_llm_json_answer(raw: Optional[str]) -> tuple[bool, str]:
    """
    从 text-json 模型的纯文本响应中提取并解析 JSON。
    成功返回 (True, 规范化后的 JSON 字符串)；失败返回 (False, 原始文本截断供日志/回显)。
    兼容偶发的 Markdown ```json``` 围栏或前缀噪声：尝试从首个可解码的 `{` / `[` 起 raw_decode。
    """

    def _try_decode_one(text: str) -> Optional[Any]:
        t = text.strip()
        if not t:
            return None
        try:
            return json.loads(t)
        except Exception:
            pass
        for _starter in ("{", "["):
            idx = 0
            while True:
                i = t.find(_starter, idx)
                if i < 0:
                    break
                try:
                    obj, _end = json.JSONDecoder().raw_decode(t, i)
                    return obj
                except Exception:
                    idx = i + 1
        return None

    if raw is None:
        return False, ""
    s = str(raw).strip()
    if not s:
        return False, ""

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
    if fence:
        obj = _try_decode_one(fence.group(1))
        if obj is not None:
            return True, json.dumps(obj, ensure_ascii=False)

    obj = _try_decode_one(s)
    if obj is not None:
        return True, json.dumps(obj, ensure_ascii=False)

    max_echo = 4000
    echo = s if len(s) <= max_echo else s[:max_echo] + "…(truncated)"
    return False, echo


async def _reporter_core_internal(
    question: str,
    content: List[dict],
    coder_output: List[str],
    docid_to_source: Dict[str, Dict[str, Any]],
    model_id: str,
    reporter_instruction: str = "",
    reporter_template: str = "default",
    history_messages: Optional[List[tuple[str, str]]] = None,
    answer_lang: str = "zh_CN",
    dependency_context: str = "",
) -> FinalAnswer:
    """
    reporter的核心逻辑内部实现，生成最终答案（不进行滑动窗口检查）。
    
    Args:
        question: 问题文本
        content: 内容列表，每个元素包含 doc_id, text, line_str
        coder_output: coder输出列表
        docid_to_source: doc_id到source信息的映射
        model_id: 模型ID
        reporter_instruction: 额外的报告指令
        history_messages: 可选的对话历史，格式为 [("human", ...), ("ai", ...), ...]
        answer_lang: 回答语言 locale，如 zh_CN / en_US
        
    Returns:
        FinalAnswer: 最终答案对象
    """
    # 将coder_output拼到content后面以统一作为输入
    content_text = "\n\n".join([c.get("line_str", "") for c in content])
    content_full = content_text + "\n\n" + "\n\n".join(coder_output)

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
    # Render prompt并调用模型（简单回答由 questionTask.prompt 中的关键词判定）
    simple_answer = judge_simple_question(reporter_instruction)
    
    if reporter_template == "text_json":
        system_prompt, human_prompt = "", apply_prompt_template(
            "content_generate_json",
            {
                "simple_answer": simple_answer,
                "question": question,
                "context_text": content_full,
                "extra_instruction": reporter_instruction,
                "dependency_context": dependency_context,
            },
        )
    elif reporter_template == "text_json_citation":
        system_prompt, human_prompt = "", apply_prompt_template(
            "content_generate_json_with_citation",
            {
                "simple_answer": simple_answer,
                "question": question,
                "context_text": content_full,
                "extra_instruction": reporter_instruction,
                "dependency_context": dependency_context,
            },
        )
    else:   
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

    try:
        # 构建消息列表：system + history_messages + current human
        # 如果用json模式，system_prompt为空
        if system_prompt and system_prompt.strip():
            messages = [("system", system_prompt)]
        else:
            messages = []
        
        # 如果有历史消息，添加到消息列表中
        if history_messages:
            messages.extend(history_messages)
        
        # 添加当前问题的 human prompt
        messages.append(("human", human_prompt))

        if reporter_template == "text_json":
            raw_llm = await ainvoke_llm_with_messages(messages=messages, model_id=model_id)
            ok, parsed_or_echo = _parse_llm_json_answer(raw_llm)
            if ok:
                final_answer = FinalAnswer(
                    is_answerable=True,
                    answer_text=parsed_or_echo,
                    citations=[],
                )
            else:
                final_answer = FinalAnswer(
                    is_answerable=False,
                    answer_text=f"模型输出非 JSON：{parsed_or_echo}",
                    citations=[],
                )
        elif reporter_template == "text_json_citation":
            raw_llm = await ainvoke_llm_with_messages(messages=messages, model_id=model_id)
            ok, parsed_or_echo = _parse_llm_json_answer(raw_llm)
            if ok:
                updated_answer_text, resolved_citations = _process_citations(
                    parsed_or_echo, docid_to_source, content, coder_output
                )
                final_answer = FinalAnswer(
                    is_answerable=True,
                    answer_text=updated_answer_text,
                    citations=resolved_citations,
                )
            else:
                final_answer = FinalAnswer(
                    is_answerable=False,
                    answer_text=f"模型输出非 JSON：{parsed_or_echo}",
                    citations=[],
                )
        else:
            try:
                raw_answer = await ainvoke_llm_with_messages_structured_output(
                    messages=messages,
                    model_id=model_id,
                    response_type=QuotedAnswer,
                )
            except ValidationError as e:
                # 简单回答时模型可能只返回数字/短句，导致校验失败，试图手动包装为 QuotedAnswer
                logger.info(f"简单回答时模型只返回数字/短句，导致校验失败，试图手动包装为 QuotedAnswer: {e}")
                raw_answer = _coerce_quoted_answer_from_validation_error(e)
                if raw_answer is None:
                    # DeepSeek json_mode 下模型可能返回 list（如 [15]）而非对象，
                    # 回退到普通文本调用，直接取模型返回的文本作为答案
                    logger.info(
                        f"Structured output 失败且无法兜底，回退到普通文本调用: model_id={model_id}"
                    )
                    raw_text = await ainvoke_llm_with_messages(
                        messages=messages, model_id=model_id
                    )
                    raw_answer = QuotedAnswer(
                        is_answerable=True,
                        answer=raw_text.strip() if raw_text else "",
                    )

            answer_text = raw_answer.answer
            # 处理答案中的citation引用
            updated_answer_text, resolved_citations = _process_citations(
                answer_text, docid_to_source, content, coder_output
            )
            final_answer = FinalAnswer(
                is_answerable=raw_answer.is_answerable,
                answer_text=updated_answer_text,
                citations=resolved_citations,
            )

    except Exception as e:
        log_error_type_and_message(
            logger=logger,
            context="Reporter Core 调用失败",
            e=e,
            exc_info=False,
            max_message_chars=200,
        )
        final_answer = FinalAnswer(is_answerable=False, answer_text=get_localized_message(
                "reporter_generation_error",
                answer_lang,
                error=str(e),
            ), citations=[])

    return final_answer


async def _reporter_with_sliding_windows(
    question: str,
    content: List[dict],
    coder_output: List[str],
    docid_to_source: Dict[str, Dict[str, Any]],
    model_id: str,
    reporter_instruction: str = "",
    reporter_template: str = "default",
    history_messages: Optional[List[tuple[str, str]]] = None,
    max_tokens_per_window: int = None,
    answer_lang: str = "zh_CN",
    dependency_context: str = "",
) -> FinalAnswer:
    """
    使用滑动窗口策略调用reporter。
    
    Args:
        question: 问题文本
        content: 内容列表
        coder_output: coder输出列表
        docid_to_source: doc_id到source信息的映射
        model_id: 模型ID
        reporter_instruction: 额外的报告指令
        history_messages: 可选的对话历史
        max_tokens_per_window: 每个窗口的最大token数，如果为None则使用_CTX_LIMIT
        answer_lang: 回答语言 locale，如 zh_CN / en_US
        
    Returns:
        合并后的FinalAnswer
    """
    if max_tokens_per_window is None:
        max_tokens_per_window = _CTX_LIMIT

    sw = StopWatch()
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
    # 估算prompt模板的token开销（简单回答由 questionTask.prompt 中的关键词判定）
    simple_answer = judge_simple_question(reporter_instruction)
    if reporter_template == "text_json":
        human_prompt_overhead = apply_prompt_template(
            "content_generate_json",
            {
                "simple_answer": simple_answer,
                "question": question,
                "context_text": "",
                "extra_instruction": reporter_instruction,
                "dependency_context": dependency_context,
            },
        )
        template_overhead = count_tokens(human_prompt_overhead)
    elif reporter_template == "text_json_citation":
        human_prompt_overhead = apply_prompt_template(
            "content_generate_json_with_citation",
            {
                "simple_answer": simple_answer,
                "question": question,
                "context_text": "",
                "extra_instruction": reporter_instruction,
                "dependency_context": dependency_context,
            },
        )
        template_overhead = count_tokens(human_prompt_overhead)
    else:
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
        template_overhead = count_tokens(system_prompt) + count_tokens(question) + count_tokens(reporter_instruction or "")
    if history_messages:
        template_overhead += sum(count_tokens(msg[1]) for msg in history_messages)
    
    # 实际可用于内容的token数
    available_tokens = max_tokens_per_window - template_overhead
    logger.info(f"可用于内容的token数: {available_tokens}")
    
    # 创建滑动窗口
    windows = _create_sliding_windows(content, coder_output, question, available_tokens)
    
    if not windows:
        # 如果没有窗口，返回默认响应
        logger.error("无法创建滑动窗口，上下文为空")
        return FinalAnswer(
            is_answerable=False,
            answer_text=get_localized_message("sliding_windows_empty", answer_lang),
            citations=[]
        )
    
    # 对每个窗口调用reporter（使用内部实现，跳过滑动窗口检查） TODO?: 并行
    window_answers = []
    for i, window in enumerate(windows):
        # 构建该窗口的content和coder_output
        window_content = window["content_items"]
        window_coder_output = window["coder_output_items"]
        
        try:
            with sw.task(f"_reporter_core_internal_{i}") as t:
                answer = await _reporter_core_internal(
                    question=question,
                    content=window_content,
                    coder_output=window_coder_output,
                    docid_to_source=docid_to_source,
                    model_id=model_id,
                    reporter_instruction=reporter_instruction,
                    reporter_template=reporter_template,
                    history_messages=history_messages,
                    answer_lang=answer_lang,
                    dependency_context=dependency_context,
                )
                t.detail(str(answer.is_answerable))
            window_answers.append(answer)
            logger.info(f"窗口 {window['window_id']} reporter响应: is_answerable={answer.is_answerable}，答案文本: {answer.answer_text}")
        except Exception as e:
            log_error_type_and_message(
                logger=logger,
                context=f"窗口 {window['window_id']} reporter调用失败",
                e=e,
                exc_info=False,
                max_message_chars=200,
            )
            # 失败时使用默认响应
            window_answers.append(FinalAnswer(
                is_answerable=False,
                answer_text=get_localized_message(
                    "sliding_window_processing_error",
                    answer_lang,
                    window_id=window["window_id"],
                    error=str(e),
                ),
                citations=[]
            ))
    
    # text-json：不需要溯源，也不应该做任何基于 [n] 的重编号/过滤（JSON 内可能自然出现 []）
    if reporter_template == "text_json":
        # 兜底：merge 失败时，返回第一段非空答案（尽量保证仍是 JSON）
        fallback_json = next((a.answer_text for a in window_answers if a.answer_text), "")
        final_answer_text = fallback_json
        
        try:
            window_answer_blocks = []
            for i, ans in enumerate(window_answers, start=1):
                text = ans.answer_text or ""
                if not text:
                    continue
                window_answer_blocks.append(f"【窗口 {i} 答案】\n{text}")
            window_answers_text = "\n\n".join(window_answer_blocks)

            merge_prompt = apply_prompt_template(
                "content_generate_json_merge_windows",
                {
                    "question": question,
                    "window_answers": window_answers_text,
                },
            )
            logger.info("正在调用大模型对窗口JSON答案进行二次整合（merge windows, text-json）")
            with sw.task("summarize_final_answer"):
                summarized = await ainvoke_llm_with_prompt(merge_prompt, model_id=model_id)
            if summarized and isinstance(summarized, str):
                final_answer_text = summarized
            else:
                logger.warning("text-json merge windows 结果为空或类型异常，回退到第一段窗口答案")
        except Exception as e:
            log_error_type_and_message(
                logger=logger,
                context="text-json merge windows 执行失败，将使用第一段窗口答案",
                e=e,
                exc_info=False,
                max_message_chars=200,
            )

        ok, normalized = _parse_llm_json_answer(final_answer_text)
        
        sw.print()
        if not ok:
            return FinalAnswer(
                is_answerable=False,
                answer_text=f"模型输出非 JSON：{final_answer_text}",
                citations=[],
            )
        return FinalAnswer(
            is_answerable=True,
            answer_text=normalized,
            citations=[],
        )

    # text-json-citation：需要做全局引用重编号后再合并（与 text 类类似），但使用 JSON merge 提示词
    if reporter_template == "text_json_citation":
        # --- 第一步：为所有窗口的引用分配全局唯一的编号 ---
        citation_key_to_gid: Dict[Tuple[Tuple[int, ...], str, str], int] = {}
        window_local_to_gid: List[Dict[int, int]] = []
        global_citations_map: Dict[int, Citation] = {}
        next_gid = 1

        for w_idx, ans in enumerate(window_answers):
            local_map: Dict[int, int] = {}
            for cit in ans.citations:
                key = (
                    tuple(sorted(cit.element_ids)) if cit.element_ids else (),
                    cit.file_id or "",
                )
                gid = citation_key_to_gid.get(key)
                if gid is None:
                    gid = next_gid
                    next_gid += 1
                    citation_key_to_gid[key] = gid
                    global_citations_map[gid] = Citation(
                        id=gid,
                        element_ids=cit.element_ids,
                        file_id=cit.file_id,
                        quote=cit.quote,
                    )
                local_map[cit.id] = gid
            window_local_to_gid.append(local_map)

        # --- 第二步：重写各窗口答案文本中的本地引用编号为全局编号 ---
        renumbered_texts: List[str] = []
        for w_idx, ans in enumerate(window_answers):
            text = ans.answer_text or ""
            local_map = window_local_to_gid[w_idx]

            def _replace_local_id(m, _local_map=local_map):
                try:
                    lid = int(m.group(1))
                except Exception:
                    return m.group(0)
                gid = _local_map.get(lid)
                if gid is None:
                    return m.group(0)
                return f"[{gid}]"

            renumbered = re.sub(r"\[(\d+)\]", _replace_local_id, text)
            renumbered_texts.append(renumbered)

        # 兜底：简单拼接
        fallback_json = next((t for t in renumbered_texts if t), "")

        # --- 第三步：用 JSON merge 提示词合并 ---
        final_answer_text = fallback_json
        try:
            window_answer_blocks = []
            for i, text in enumerate(renumbered_texts, start=1):
                if not text:
                    continue
                window_answer_blocks.append(f"【窗口 {i} 答案】\n{text}")
            window_answers_text = "\n\n".join(window_answer_blocks)

            merge_prompt = apply_prompt_template(
                "content_generate_json_with_citation_merge_windows",
                {
                    "question": question,
                    "window_answers": window_answers_text,
                },
            )
            logger.info("正在调用大模型对窗口JSON答案进行二次整合（merge windows, text-json-citation）")
            with sw.task("summarize_final_answer"):
                summarized = await ainvoke_llm_with_prompt(merge_prompt, model_id=model_id)
            if summarized and isinstance(summarized, str):
                final_answer_text = summarized
            else:
                logger.warning("text-json-citation merge windows 结果为空或类型异常，回退到第一段窗口答案")
        except Exception as e:
            log_error_type_and_message(
                logger=logger,
                context="text-json-citation merge windows 执行失败，将使用第一段窗口答案",
                e=e,
                exc_info=False,
                max_message_chars=200,
            )

        ok, normalized = _parse_llm_json_answer(final_answer_text)

        sw.print()
        if not ok:
            return FinalAnswer(
                is_answerable=False,
                answer_text=f"模型输出非 JSON：{final_answer_text}",
                citations=[],
            )

        # --- 第四步：基于最终答案文本过滤未使用的引用编号 ---
        merged_citations = [
            global_citations_map[gid] for gid in sorted(global_citations_map.keys())
        ]
        used_ids: set[int] = set()
        try:
            id_matches = re.findall(r"\[(\d+)\]", normalized)
            used_ids = {int(m) for m in id_matches}
        except Exception as e:
            log_error_type_and_message(
                logger=logger,
                context="解析最终答案中的引用编号失败",
                e=e,
                exc_info=False,
                max_message_chars=200,
            )

        if used_ids:
            filtered_citations = [c for c in merged_citations if c.id in used_ids]
        else:
            logger.warning("text-json-citation 合并答案中没有任何 [id]，保留全部 citations 以避免信息丢失")
            filtered_citations = merged_citations

        return FinalAnswer(
            is_answerable=True,
            answer_text=normalized,
            citations=filtered_citations,
        )

    # 合并所有窗口的答案（非 text-json）
    merged_is_answerable = any(a.is_answerable for a in window_answers)

    # --- 第一步：为所有窗口的引用分配全局唯一的编号 ---
    citation_key_to_gid: Dict[Tuple[Tuple[int, ...], str, str], int] = {}
    window_local_to_gid: List[Dict[int, int]] = []
    global_citations_map: Dict[int, Citation] = {}
    next_gid = 1

    for w_idx, ans in enumerate(window_answers):
        local_map: Dict[int, int] = {}
        for cit in ans.citations:
            # 使用 (element_ids, file_id) 作为来源去重的 key，quote 不参与 key 比较
            key = (
                tuple(sorted(cit.element_ids)) if cit.element_ids else (),
                cit.file_id or "",
            )
            gid = citation_key_to_gid.get(key)
            if gid is None:
                gid = next_gid
                next_gid += 1
                citation_key_to_gid[key] = gid
                global_citations_map[gid] = Citation(
                    id=gid,
                    element_ids=cit.element_ids,
                    file_id=cit.file_id,
                    quote=cit.quote,
                )
            local_map[cit.id] = gid
        window_local_to_gid.append(local_map)

    # --- 第二步：重写各窗口答案文本中的本地引用编号为全局编号 ---
    renumbered_texts: List[str] = []
    for w_idx, ans in enumerate(window_answers):
        text = ans.answer_text or ""
        local_map = window_local_to_gid[w_idx]

        def _replace_local_id(m):
            try:
                lid = int(m.group(1))
            except Exception:
                return m.group(0)
            gid = local_map.get(lid)
            if gid is None:
                return m.group(0)
            return f"[{gid}]"

        renumbered = re.sub(r"\[(\d+)\]", _replace_local_id, text)
        renumbered_texts.append(renumbered)

    # 先简单合并所有窗口的答案文本，作为兜底结果
    fallback_merged_answer_text = "\n\n".join(
        [
            f"{get_localized_message('window_answer_label', answer_lang, window_id=i + 1)}\n{text}"
            for i, text in enumerate(renumbered_texts)
            if text
        ]
    )
    logger.info(f"简单合并所有窗口的答案文本: {fallback_merged_answer_text}")

    # 全局 citations 列表（按 id 排序），稍后再基于最终答案文本过滤未使用的编号
    merged_citations = [
        global_citations_map[gid] for gid in sorted(global_citations_map.keys())
    ]

    logger.info(
        f"合并了 {len(window_answers)} 个窗口的reporter响应: "
        f"is_answerable={merged_is_answerable}, citations数量={len(merged_citations)}"
    )

    # ============= 使用大模型对窗口答案做二次总结（保留引用编号） =============
    final_answer_text = fallback_merged_answer_text
    try:
        # 构造用于合并的窗口答案文本（使用已重写引用编号后的文本）
        window_answer_blocks = []
        for i, text in enumerate(renumbered_texts, start=1):
            if not text:
                continue
            window_answer_blocks.append(
                f"{get_localized_message('window_answer_label', answer_lang, window_id=i)}\n{text}"
            )
        window_answers_text = "\n\n".join(window_answer_blocks)

        merge_prompt = apply_prompt_template(
            "content_generation_merge_windows",
            build_prompt_state(
                "content_generation_merge_windows",
                answer_lang,
                question=question,
                window_answers=window_answers_text,
                answer_language_instruction=answer_language_instruction,
            ),
            locale=answer_lang,
        )

        logger.info("正在调用大模型对窗口答案进行二次总结（merge windows）")
        summarized = await ainvoke_llm_with_prompt(merge_prompt, model_id=model_id)
        logger.info(f"大模型的二次总结结果: {summarized}")
        if summarized and isinstance(summarized, str):
            final_answer_text = summarized
        else:
            logger.warning("merge windows 总结结果为空或类型异常，回退到简单拼接结果")
    except Exception as e:
        log_error_type_and_message(
            logger=logger,
            context="merge windows 总结失败，将使用简单拼接结果",
            e=e,
            exc_info=False,
            max_message_chars=200,
        )

    # ============= 基于最终答案文本过滤未使用的引用编号 =============
    # 与重写引用编号时使用的格式一致，匹配 [1]、[2] 等
    used_ids: set[int] = set()
    try:
        id_matches = re.findall(r"\[(\d+)\]", final_answer_text)
        used_ids = {int(m) for m in id_matches}
    except Exception as e:
        log_error_type_and_message(
            logger=logger,
            context="解析最终答案中的引用编号失败",
            e=e,
            exc_info=False,
            max_message_chars=200,
        )

    if used_ids:  # TODO?: 现在仅仅是简单的筛选，没有重排序，可能需要优化
        filtered_citations = [c for c in merged_citations if c.id in used_ids]
        logger.info(f"过滤后的 citations : {used_ids}")
    else:
        # 如果答案中没有任何 [id]，保留全部 citations 以避免信息丢失
        logger.error("答案中没有任何 [id]，保留全部 citations 以避免信息丢失")
        filtered_citations = merged_citations

    return FinalAnswer(
        is_answerable=merged_is_answerable,
        answer_text=final_answer_text,
        citations=filtered_citations,
    )


# Planner：基于 prompts/planner.md 判断是否需要进一步计算
# - 渲染模板并调用 LLM
# - 使用 with_structured_output 直接获取结构化响应
# - 通过 Command(update, goto) 显式路由到 coder 或 reporter
async def planner(state: ContentGenState) -> Command[str]:
    plan_iter = state["plan_iteration"] + 1
    answer_lang = state.get("answer_lang", "zh_CN") or "zh_CN"
    if plan_iter > state["max_iterations"]: # 卫语句
        max_iter = state["max_iterations"]
        logger.warning(f"达到Planner最大循环次数 {max_iter} 次，强制进入reporter")
        return Command(update=None, goto="reporter")
    

    logger.info(f"Planner | answer_lang={answer_lang}")
    question = state["question"]
    # 这里从List[dict]对象中中取每一个dict的line_str
    content_text = "\n\n".join([c.get("line_str", "") for c in state["content"]])
    # content_text = "\n\n".join(state["content"])
    model_id = state["model_id"]
    coder_output = state["coder_output"]

    # 使用新的planner提示词
    prompt = apply_prompt_template(
        "planner_new",
        {
            "question": question,
            "content": content_text,
            "coder_output": coder_output
        },
        locale=answer_lang,
    )
    
    # 检查是否需要使用滑动窗口策略
    prompt_tokens = count_tokens(prompt)
    if prompt_tokens > _CTX_LIMIT:
        logger.warning(f"Planner prompt长度超过阈值({_CTX_LIMIT} tokens，实际{prompt_tokens} tokens)，将使用滑动窗口策略")
        try:
            planner_response = await _planner_with_sliding_windows(
                question=question,
                content=state["content"],
                coder_output=coder_output,
                model_id=model_id,
                answer_lang=answer_lang,
                max_tokens_per_window=_CTX_LIMIT
            )
            
            needs_computation = planner_response.needs_computation
            reasoning = planner_response.reasoning
            computation_prompt = planner_response.computation_prompt
            computation_input = planner_response.computation_input
        except Exception as e:
            log_error_type_and_message(
                logger=logger,
                context="滑动窗口planner处理失败",
                e=e,
                exc_info=False,
                max_message_chars=200,
            )
            needs_computation = False
            reasoning = (
                f"Sliding-window processing failed: {str(e)}"
                if answer_lang == "en_US"
                else f"滑动窗口处理失败: {str(e)}"
            )
            computation_prompt = ""
            computation_input = []
    else:
        # 正常处理流程
        try:
            # 使用结构化输出直接获取 PlannerResponse 对象
            planner_response = await ainvoke_llm_with_structured_output(prompt, model_id=model_id, response_type=PlannerResponse)
            logger.info(f"Planner Output:{planner_response}")
            # 从结构化响应中提取字段
            needs_computation = planner_response.needs_computation
            reasoning = planner_response.reasoning
            computation_prompt = planner_response.computation_prompt
            computation_input = planner_response.computation_input
            
        except Exception as e:
            log_error_type_and_message(
                logger=logger,
                context="结构化输出失败",
                e=e,
                exc_info=False,
                max_message_chars=200,
            )
            needs_computation = False
            reasoning = ""
            computation_prompt = ""
            computation_input = []


    # 直接在节点中添加planner步骤到answer_agent_steps
    from .models import AnswerAgentStepItem
    answer_agent_steps = state.get("answer_agent_steps", [])
    logger.debug(f"[planner] 当前answer_agent_steps数量: {len(answer_agent_steps)}")
    
    if reasoning:
        answer_agent_steps.append(AnswerAgentStepItem(
            type="planner",
            text=reasoning
        ))
        logger.debug(f"[planner] 成功添加planner步骤，当前answer_agent_steps数量: {len(answer_agent_steps)}")
    else:
        logger.error(f"[planner] reasoning为空，写入空thinking_steps")
        answer_agent_steps.append(AnswerAgentStepItem(
            type="planner",
            text=""
        ))
    
    update = {
        "computation_prompt": computation_prompt,
        "computation_input": computation_input,
        "plan_iteration": plan_iter,
        "answer_agent_steps": answer_agent_steps,
    }

    goto = "reporter"
    if needs_computation:
        goto="coder"

    return Command(update=update, goto=goto)


# Coder Node
async def coder(state: ContentGenState) -> Command[str]:
    logger.info("Coder")
    answer_lang = state.get("answer_lang", "zh_CN") or "zh_CN"

    try:
        # 准备参数
        question = state["question"]
        computation_prompt = state["computation_prompt"]
        computation_input = state["computation_input"]

        # 若 computation_input 为空，无可用计算输入，不调用计算智能体，直接进入 reporter
        if not computation_input:
            logger.info(
                "[coder] computation_input 为空，跳过计算智能体调用，直接进入 reporter 节点"
            )
            return Command(update=None, goto="reporter")

        content = state["content"]
        # 准备docid_to_source映射
        docid_to_source = state["docid_to_source"]
        # coder调用
        documents = []
        # 准备coder使用的文档的file_id 和 element_ids
        coder_citations = {
            "element_ids": [],
            "file_id": ""
        }
        
        for idx, docid in enumerate(computation_input):
            content_text = "\n\n".join([c.get("text", "") for c in content if c.get("doc_id", "") == docid])
            doc = docid_to_source.get(docid, {})
            element_ids = doc.get("element_ids", [])
            file_id = doc.get("file_id", "")
            if doc:
                documents.append(
                    {
                        "id": idx + 1,
                        "title": docid,
                        "text": content_text,
                        "element_ids": element_ids,
                        "file_id": file_id,
                    }
                )
                # 合并element_ids
                coder_citations["element_ids"].extend([eid for eid in element_ids if eid not in coder_citations["element_ids"]])
                # 收集file_id，如果已有则不再更改 TODO: 多个文件时citations可能还有问题
                if file_id and not coder_citations["file_id"]:
                    coder_citations["file_id"] = file_id

        # 使用CodingAgentService调用Coding Agent来获取结果
        # 修复：将 asyncio.run() 改为 await，避免阻塞事件循环
        from ..data_analyst_agent.coding_agent.graph.agent_full import AgentFullService
        coding_agent_service = AgentFullService()
        if _CFG.coding_model_id is not None and _CFG.coding_model_id != "":
            coding_model_id=_CFG.coding_model_id
        else:
            coding_model_id = state["model_id"]
            logger.warning(f"coding_model_id set to none fallback to default model_id:{coding_model_id}")
        if _CFG.coding_context_token_limit is not None:
            coding_context_token_limit = _CFG.coding_context_token_limit
        else:
            logger.warning("coding_context_token_limit not set, using default.")
            coding_context_token_limit = None
        coding_max_retry_limit = _CFG.coding_max_retry_limit
        result = await coding_agent_service.run_full_analysis_with_token_limit(
            question=question, 
            prompt=computation_prompt,
            documents=documents,
            disable_planning=True,
            disable_extract=True,
            token_limit=coding_context_token_limit,
            max_iterations=coding_max_retry_limit, 
            model_id = coding_model_id, # 传入model_id以免其使用
            answer_language = answer_lang
        )
        coder_generation_result = result["result"]
        trace_task = result.get("trace_task")  # 获取trace_task用于填充answer_agent_steps
        logger.info(f"coder_generation_result: {coder_generation_result}")
        logger.info(f"[coder] trace_task值: {trace_task}")
        # coder_generation_result = "Test" 

        # 将这次coder的输出添加coder_output中
        coder_output = state["coder_output"]
        content_cnt = state["content_cnt"]
        content = state["content"]
        # doc_id 归一化：coder 生成的输出也使用 doc1, doc2, ... 格式
        # 编号从 content 数量 + 已有 coder_output 数量 + 1 开始，确保在整个上下文中唯一连续
        coder_normalized_id = f"doc{len(content) + len(coder_output) + 1}"
        # 将归一化后的 coder output 也注册到 docid_to_source 中
        # _original_doc_id 保存旧格式 ID 用于调试和日志溯源
        docid_to_source[coder_normalized_id] = {**coder_citations, "_original_doc_id": f"coder_output_{len(coder_output)}"}

        # 将 coder 生成结果包装为 <doc id=docN>...</doc> 格式，与 content 中其他文档的格式统一
        new_coder_output = f"<doc id={coder_normalized_id}>\n{coder_generation_result}\n</doc>"
        content_cnt += 1
        coder_output.append(new_coder_output)
        
        # 直接在节点中添加coder步骤到answer_agent_steps
        answer_agent_steps = state.get("answer_agent_steps", [])
        logger.debug(f"[coder] 当前answer_agent_steps数量: {len(answer_agent_steps)}")
        
        try:
            from .models import AnswerAgentStepItem
            
            if trace_task is None:
                logger.error(f"[coder] trace_task为None，无法添加coder步骤")
            else:
                # trace_task是TraceTask对象，使用getattr获取trace_steps
                trace_steps = getattr(trace_task, "trace_steps", []) if trace_task else []
                logger.debug(f"[coder] trace_steps数量: {len(trace_steps) if trace_steps else 0}")
               
                answer_agent_steps.append(AnswerAgentStepItem(
                    type="coder",
                    items=trace_steps if trace_steps else None
                ))
                logger.debug(f"[coder] 成功添加coder步骤，当前answer_agent_steps数量: {len(answer_agent_steps)}")
        except Exception as e:
            log_error_type_and_message(
                logger=logger,
                context="[coder] 添加coder步骤到answer_agent_steps失败",
                e=e,
                exc_info=True,
                max_message_chars=200,
            )
            # 不要重新获取answer_agent_steps，保持当前状态
        
        update_dict = {
            "coder_output": coder_output,
            "docid_to_source": docid_to_source,
            "answer_agent_steps": answer_agent_steps
        }
        
        return Command(update=update_dict, goto="planner")
    
    except Exception as e:
        log_error_type_and_message(
            logger=logger,
            context="[coder] 处理过程中发生未知错误",
            e=e,
            exc_info=True,
            max_message_chars=200,
        )
        # 创建包含错误信息的FinalAnswer
        error_answer = FinalAnswer(
            is_answerable=False,
            answer_text=get_localized_message(
                "coder_processing_error",
                answer_lang,
                error=str(e),
            ),
            citations=[],
        )
        # 获取answer_agent_steps并添加错误信息
        answer_agent_steps = state.get("answer_agent_steps", [])
        try:
            from .models import AnswerAgentStepItem, ThinkingSteps
            answer_agent_steps.append(AnswerAgentStepItem(
                type="coder",
                text=get_localized_message(
                    "coder_processing_error",
                    answer_lang,
                    error=str(e),
                )
            ))
            # 设置answer_agent_steps到thinking_steps
            if error_answer.thinking_steps is None:
                error_answer.thinking_steps = ThinkingSteps()
            error_answer.thinking_steps.answer_agent_steps = answer_agent_steps
        except Exception as inner_e:
            logger.error(f"[coder] 添加错误步骤到answer_agent_steps失败: {inner_e}")
        
        # 直接跳转到__end__，设置错误答案，避免reporter重新生成答案
        return Command(update={"answer": error_answer, "answer_agent_steps": answer_agent_steps}, goto="__end__")

@trace_span(name="reporter_core")
async def reporter_core(
    question: str,
    content: List[dict],
    coder_output: List[str],
    docid_to_source: Dict[str, Dict[str, Any]],
    model_id: str,
    reporter_instruction: str = "",
    reporter_template: str = "default",
    history_messages: Optional[List[tuple[str, str]]] = None,
    answer_lang: str = "zh_CN",
    dependency_context: str = "",
) -> FinalAnswer:
    """reporter的核心逻辑，生成最终答案。
    
    Args:
        question: 问题文本
        content: 内容列表，每个元素包含 doc_id, text, line_str
        coder_output: coder输出列表
        docid_to_source: doc_id到source信息的映射
        model_id: 模型ID
        reporter_instruction: 额外的报告指令
        history_messages: 可选的对话历史，格式为 [("human", ...), ("ai", ...), ...]
        answer_lang: 回答语言 locale，如 zh_CN / en_US
        
    Returns:
        FinalAnswer: 最终答案对象
    """
    logger.info("Reporter Core | model_id={}", model_id)
    logger.info(
        f"[i18n] reporter_core收到语言参数: "
        f"question={question[:100]}, answer_lang={answer_lang}"
    )
    
    # 将coder_output拼到content后面以统一作为输入
    content_text = "\n\n".join([c.get("line_str", "") for c in content])
    content_full = content_text + "\n\n" + "\n\n".join(coder_output)

    # 使用独立的logger将上下文文本存储到独立的log文件中
    context_logger = logger.bind(request_tag="context_text")
    context_logger.info(f"问题: {question}\n\n上下文文本:\n{content_full}")

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
    # Render prompt并调用模型（简单回答由 questionTask.prompt 中的关键词判定）
    simple_answer = judge_simple_question(reporter_instruction)
    logger.info(f"问题'{question}'是否简单回答: {simple_answer}")
    if reporter_template == "text_json":
        system_prompt = ""
        human_prompt = apply_prompt_template(
            "content_generate_json",
            {
                "simple_answer": simple_answer,
                "question": question,
                "context_text": content_full,
                "extra_instruction": reporter_instruction,
                "dependency_context": dependency_context,
            },
        )
    elif reporter_template == "text_json_citation":
        system_prompt = ""
        human_prompt = apply_prompt_template(
            "content_generate_json_with_citation",
            {
                "simple_answer": simple_answer,
                "question": question,
                "context_text": content_full,
                "extra_instruction": reporter_instruction,
                "dependency_context": dependency_context,
            },
        )
    else:
        # 使用拆分后的 system / human 模板，避免在代码中基于字符串启发式拆分
        # system_prompt 只需要 simple_answer（控制是否展示示例/简短回答说明）
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
        # human_prompt 使用具体的问题、上下文和额外指示
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

    # 检查是否需要使用滑动窗口策略
    human_prompt_tokens = count_tokens(human_prompt)
    system_prompt_tokens = count_tokens(system_prompt)
    history_tokens = sum(count_tokens(msg[1]) for msg in history_messages) if history_messages else 0
    total_tokens = human_prompt_tokens + system_prompt_tokens + history_tokens
    
    if total_tokens > _CTX_LIMIT:
        logger.warning(f"Reporter上下文长度超过阈值({_CTX_LIMIT} tokens，实际{total_tokens} tokens)，将使用滑动窗口策略")
        try:
            return await _reporter_with_sliding_windows(
                question=question,
                content=content,
                coder_output=coder_output,
                docid_to_source=docid_to_source,
                model_id=model_id,
                reporter_instruction=reporter_instruction,
                reporter_template=reporter_template,
                history_messages=history_messages,
                max_tokens_per_window=_CTX_LIMIT,
                answer_lang=answer_lang,
                dependency_context=dependency_context,
            )
        except Exception as e:
            log_error_type_and_message(
                logger=logger,
                context="滑动窗口reporter处理失败",
                e=e,
                exc_info=False,
                max_message_chars=200,
            )
            return FinalAnswer(
                is_answerable=False,
                answer_text=get_localized_message(
                    "sliding_windows_processing_error",
                    answer_lang,
                    error=str(e),
                ),
                citations=[]
            )

    # 正常处理流程：直接调用内部实现
    return await _reporter_core_internal(
        question=question,
        content=content,
        coder_output=coder_output,
        docid_to_source=docid_to_source,
        model_id=model_id,
        reporter_instruction=reporter_instruction,
        reporter_template=reporter_template,
        history_messages=history_messages,
        answer_lang=answer_lang,
    )


def _match_doc_id(
    citation_text: str,
    available_doc_ids: set[str]
) -> str | None:
    """将 citation 文本匹配到可用的 doc_id。

    归一化 doc_id 格式为 doc1, doc2, ... 后，匹配策略：
    1. 精确匹配（如模型输出 [doc1] → doc1）
    2. 去掉 doc_ 前缀后匹配（如模型错误输出 [doc_doc1] → doc1）
    3. 纯数字匹配（如模型偷懒只写 [1] → doc1）

    注意：不采用模糊/子串匹配，避免放大模型幻觉
    
    Args:
        citation_text: 从答案中提取的citation文本
        available_doc_ids: 所有可用的doc_id集合
        
    Returns:
        匹配到的doc_id，如果无法匹配则返回None
    """
    # 策略1: 精确匹配 —— 模型正确输出 [doc1] 等归一化格式
    if citation_text in available_doc_ids:
        return citation_text
    
    # 策略2: 去掉 doc_ 前缀后匹配
    # 模型可能将 "doc" 和 doc_id 拼在一起输出 [doc_doc1]，需要修正为 doc1
    stripped = citation_text
    if stripped.startswith("doc_"):
        stripped = stripped[4:]
    if stripped.startswith("doc"):
        candidate = "doc" + stripped[3:]
        if candidate in available_doc_ids:
            logger.warning(f"前缀修正匹配: '{citation_text}' -> '{candidate}'")
            return candidate
    
    # 策略3: 纯数字匹配 [1] → doc1
    # 模型可能偷懒只输出数字，需要补全 doc 前缀
    if citation_text.isdigit():
        candidate = f"doc{citation_text}"
        if candidate in available_doc_ids:
            logger.warning(f"纯数字匹配: '{citation_text}' -> '{candidate}'")
            return candidate
    
    # 以上策略均未命中，说明模型产生了幻觉引用或格式完全无法识别
    # 不做模糊匹配，避免错误地"校验通过"不存在的引用
    logger.warning(f"无法匹配citation '{citation_text}'到任何doc_id，将视为普通文本内容")
    return None


def _extract_quote_from_doc_id(
    doc_id: str,
    content: List[dict],
    coder_output: List[str]
) -> str:
    """从doc_id提取quote文本（取前200字，跳过<doc id=...>中的内容）。

    归一化 doc_id 后，content 和 coder_output 统一使用 doc1, doc2, ... 格式，
    不再按 "coder_output_" 前缀区分来源。查找顺序：先在 content 中查，再在 coder_output 中查。
    这是因为 content 通常包含原始检索文档，coder_output 是计算生成的补充内容。
    
    Args:
        doc_id: 文档ID（归一化格式，如 doc1, doc2）
        content: 内容列表，每个元素包含 doc_id, text, line_str
        coder_output: coder输出列表
        
    Returns:
        提取的quote文本（前200字）
    """
    try:
        # 先在 content 中查找 —— content 是原始检索文档，优先级更高
        for content_item in content:
            if content_item.get("doc_id") == doc_id:
                line_str = content_item.get("line_str", "")
                quote_text = re.sub(r'<doc id=[^>]*>', '', line_str)
                quote_text = re.sub(r'</doc>', '', quote_text)
                quote_text = quote_text.strip()
                return quote_text[:200] if len(quote_text) > 200 else quote_text

        # 再在 coder_output 中查找 —— coder_output 是计算生成的补充内容
        # coder output 格式为 <doc id=docN>...</doc>，通过标签中的 id 匹配
        for coder_text in coder_output:
            if f'<doc id={doc_id}>' in coder_text:
                quote_text = re.sub(r'<doc id=[^>]*>', '', coder_text)
                quote_text = re.sub(r'</doc>', '', quote_text)
                quote_text = quote_text.strip()
                return quote_text[:200] if len(quote_text) > 200 else quote_text

        return ""
    except Exception as e:
        # quote 提取失败不应中断整体的 citation 处理流程，降级返回空字符串
        log_error_type_and_message(
            logger=logger,
            context=f"提取 doc_id quote 失败 | doc_id={doc_id}",
            e=e,
            exc_info=False,
            max_message_chars=200,
        )
        return ""


def _process_citations(
    answer_text: str,
    docid_to_source: Dict[str, Dict[str, Any]],
    content: List[dict],
    coder_output: List[str]
) -> Tuple[str, List[Citation]]:
    """
    处理答案中的citation引用，包括匹配、替换和构建citations数组。
    
    Args:
        answer_text: 原始答案文本（包含[doc_id]格式的引用）
        docid_to_source: doc_id到source信息的映射
        content: 内容列表，每个元素包含 doc_id, text, line_str
        coder_output: coder输出列表
        
    Returns:
        tuple: (更新后的答案文本, citations数组)
    """
    # 使用正则表达式提取所有 [doc_id] 格式的引用
    citation_pattern = r'\[(doc\d+)\]'
    matches = re.findall(citation_pattern, answer_text)
    logger.info(f"从答案中提取到 {len(matches)} 个引用标记: {matches}")
    
    # 获取所有可用的doc_id
    mapping = docid_to_source or {}
    available_doc_ids = set(mapping.keys())
    for content_item in content:
        doc_id = content_item.get("doc_id")
        if doc_id:
            available_doc_ids.add(doc_id)
    # doc_id 归一化后，coder_output 中每项格式为 <doc id=docN>...</doc>
    # 从标签中提取归一化后的 doc_id，确保 citation 匹配能覆盖 coder 生成的文档
    for coder_text in coder_output:
        m = re.search(r'<doc id=([^>]+)>', coder_text)
        if m:
            available_doc_ids.add(m.group(1))
    
    logger.info(f"可用的doc_id列表: {sorted(available_doc_ids)}")
    
    # 按照在答案中出现的顺序，为每个唯一的 doc_id 分配从1开始的连续 id
    doc_id_to_id = {}
    citation_to_doc_id = {}  # 记录citation文本到实际doc_id的映射（只记录成功匹配的）
    raw_text_only_doc_ids: set[str] = set()  # 需要忽略的纯文本片段（无 file_id/element_ids 溯源）
    current_id = 1
    ordered_doc_ids = []
    
    for citation_text in matches:
        # 将citation文本匹配到实际的doc_id
        actual_doc_id = _match_doc_id(citation_text, available_doc_ids)
        
        # 只有成功匹配的citation才进行处理
        if actual_doc_id is not None:
            citation_to_doc_id[citation_text] = actual_doc_id

            # raw_text_only：无真实文档来源，忽略该引用（不编号、不产出 citations；答案中也会移除该引用标记）
            src = mapping.get(actual_doc_id)
            if src.get("raw_text_only") is True:
                if actual_doc_id not in raw_text_only_doc_ids:
                    logger.warning(
                        f"检测到 raw_text_only 引用，将忽略并从答案中移除: citation='{citation_text}', doc_id='{actual_doc_id}'"
                    )
                raw_text_only_doc_ids.add(actual_doc_id)
                continue
            
            # 如果这个doc_id还没有分配id，则分配
            if actual_doc_id not in doc_id_to_id:
                doc_id_to_id[actual_doc_id] = current_id
                ordered_doc_ids.append(actual_doc_id)
                logger.info(f"为 doc_id '{actual_doc_id}' (来自citation '{citation_text}') 分配 id={current_id}")
                current_id += 1
    
    # 将答案中的 [citation_text] 替换为 [id]（只替换成功匹配的citation）
    def replace_citation(match):
        citation_text = match.group(1)
        # 检查是否成功匹配到doc_id
        if citation_text not in citation_to_doc_id:
            # 无法匹配，保持原样（视为普通文本内容）
            return f"[{citation_text}]"
        
        # 找到对应的实际doc_id
        actual_doc_id = citation_to_doc_id[citation_text]
        # raw_text_only：删除该引用标记
        if actual_doc_id in raw_text_only_doc_ids:
            return ""
        # 获取对应的id
        citation_id = doc_id_to_id.get(actual_doc_id)
        if citation_id is None:
            logger.error(f"citation '{citation_text}' (映射到doc_id '{actual_doc_id}') 没有找到对应的 id")
            return f"[{citation_text}]"
        return f"[{citation_id}]"
    
    updated_answer_text = re.sub(citation_pattern, replace_citation, answer_text)
    
    # 将 doc_id 解析为 element_ids/file_id，并重新构建 citations 数组
    resolved_citations: list[Citation] = []
    
    # 按照 ordered_doc_ids 的顺序构建 citations，确保 id 从1开始连续递增
    for idx, doc_id in enumerate(ordered_doc_ids, start=1):
        src = mapping.get(doc_id, {})
        # 从 content 或 coder_output 中提取对应的 quote
        quote = _extract_quote_from_doc_id(doc_id, content, coder_output)
        
        resolved_citations.append(
            Citation(
                id=idx,  # 从1开始连续递增
                element_ids=src.get("element_ids", []),
                file_id=src.get("file_id", ""),
                quote=quote,
            )
        )
    
    return updated_answer_text, resolved_citations


async def reporter(state: ContentGenState) -> Command[str]:
    """Reporter节点，调用core函数并处理状态更新。
    
    这是原有流程的wrapper函数，调用core函数并处理状态更新。
    """
    logger.info("Reporter")
    answer_lang = state.get("answer_lang", "zh_CN") or "zh_CN"
    logger.info(
        f"[i18n] reporter节点读取到状态语言: "
        f"question={state.get('question', '')[:100]}, answer_lang={answer_lang}"
    )
    reporter_model_id = state.get("reporter_model_id")
    effective_model_id = reporter_model_id or state["model_id"]
    final_answer = await reporter_core(
        question=state["question"],
        content=state["content"],
        coder_output=state["coder_output"],
        docid_to_source=state.get("docid_to_source", {}),
        model_id=effective_model_id,
        reporter_instruction=state.get("reporter_instruction", ""),
        reporter_template=state.get("reporter_template", "default"),
        history_messages=state.get("history_messages"),
        answer_lang=answer_lang,
        dependency_context=state.get("dependency_context", ""),
    )
    
    # 添加reporter步骤到answer_agent_steps
    from .models import AnswerAgentStepItem, ThinkingSteps
    answer_agent_steps = state.get("answer_agent_steps", [])
    logger.debug(f"[reporter] 当前answer_agent_steps数量: {len(answer_agent_steps)}")
    logger.debug(f"[reporter] answer_agent_steps内容: {[step.type for step in answer_agent_steps]}")
    
    if final_answer and final_answer.answer_text:
        answer_agent_steps.append(AnswerAgentStepItem(
            type="reporter",
            text=final_answer.answer_text
        ))
        logger.debug(f"[reporter] 成功添加reporter步骤，当前answer_agent_steps数量: {len(answer_agent_steps)}")
    else:
        logger.warning(f"[reporter] final_answer或answer_text为空，无法添加reporter步骤")
    
    # 将answer_agent_steps设置到final_answer.thinking_steps中
    if answer_agent_steps:
        if final_answer.thinking_steps is None:
            logger.error(f"[reporter] final_answer.thinking_steps为空，写入空thinking_steps")
            final_answer.thinking_steps = ThinkingSteps()
        final_answer.thinking_steps.answer_agent_steps = answer_agent_steps
        logger.info(f"[reporter] 成功设置answer_agent_steps到final_answer.thinking_steps，数量: {len(final_answer.thinking_steps.answer_agent_steps)}")
    else:
        logger.warning(f"[reporter] answer_agent_steps为空，无法设置到final_answer.thinking_steps")

    return Command(update={"answer": final_answer}, goto="__end__")
