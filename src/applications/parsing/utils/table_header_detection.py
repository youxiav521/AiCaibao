from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar
import asyncio

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator
from loguru import logger

from src.agents.content_generate_agent.utils import _retry_with_backoff
from src.utils.llm_invoke import DEFAULT_INITIAL_DELAY_S, DEFAULT_MAX_DELAY_S, _retry_model_id_for_gate, _with_timeout
from src.utils.model_factory import ModelFactory
from src.utils.tokenizer import Tokenizer

T = TypeVar("T")


async def _ainvoke_table_header_json_schema(
    prompt: str,
    *,
    model_id: str,
    response_type: Type[T],
    timeout_s: Optional[float] = None,
    max_retries: int = 3,
    initial_delay: float = DEFAULT_INITIAL_DELAY_S,
    max_delay: float = DEFAULT_MAX_DELAY_S,
    rate_limited: bool = True,
    **model_overrides,
) -> T:
    async def _call_llm():
        model = ModelFactory.get_model(key=model_id, rate_limited=rate_limited, **(model_overrides or {}))
        prompt_template = ChatPromptTemplate.from_messages([("human", "{prompt}")])
        structured = model.with_structured_output(response_type, method="json_schema")
        chain = prompt_template | structured
        return await chain.ainvoke({"prompt": prompt})

    coro = _retry_with_backoff(
        _call_llm,
        max_retries=max_retries,
        initial_delay=initial_delay,
        max_delay=max_delay,
        model_id=_retry_model_id_for_gate(model_id),
    )
    return await _with_timeout(coro, timeout_s=timeout_s)


class HeaderPresence(BaseModel):
    hasHeader: bool = Field(description="该表格是否包含表头（列标题）")

class HeaderRowCount(BaseModel):
    # 表头行数：0 表示无表头；正整数表示顶部连续表头的行数
    numRows: int = Field(description="表头占用的顶部行数，0 表示无表头")

    @field_validator("numRows", mode="before")
    @classmethod
    def _coerce_num_rows(cls, v: Any) -> int:
        try:
            n = int(v)
        except Exception:
            return 0
        if n < 0:
            return 0
        # 上限保护，避免异常大数；通常表头不超过 6 行
        return min(n, 6)


class BatchHeaderItem(BaseModel):
    """批量模式下每张表的结果。"""
    id: int = Field(description="表格编号（从 prompt 输入的 1 开始）")
    numRows: int = Field(description="表头占用的顶部行数，0 表示无表头")

    @field_validator("numRows", mode="before")
    @classmethod
    def _coerce_num_rows(cls, v: Any) -> int:
        try:
            n = int(v)
        except Exception:
            return 0
        if n < 0:
            return 0
        return min(n, 6)


class BatchHeaderResult(BaseModel):
    """批量模式的完整响应。"""
    items: List[BatchHeaderItem] = Field(description="每张表的表头行数")


# 批量模式的上下文预算：模型上下文上限 64K token，留出足够空间给 prompt 指令/输出
_BATCH_CONTEXT_TOKEN_LIMIT: int = 64000
_BATCH_PROMPT_INSTRUCTION_RESERVE: int = 4000  # 指令/输出预留
_BATCH_PER_TABLE_HTML_TR_LIMIT: int = 10
_BATCH_PER_TABLE_ROWS_PREVIEW: int = 7
# 每组最多处理的表格数量，避免 prompt 过长导致 LLM 推理超时或漏判
_BATCH_MAX_TABLES_PER_GROUP: int = 40


def _compute_is_standard_2d_from_rows(rows: Any) -> bool:
    try:
        if not isinstance(rows, list):
            return False
        for row in rows:
            if not isinstance(row, list):
                return False
            for cell in row:
                if not isinstance(cell, dict):
                    # 兼容已展开为纯文本二维数组的情况
                    continue
                rs = cell.get("rowspan")
                cs = cell.get("colspan")
                # 兼容 row_span/col_span 命名
                if rs is None:
                    rs = cell.get("row_span", 1)
                if cs is None:
                    cs = cell.get("col_span", 1)
                try:
                    if int(rs or 1) > 1 or int(cs or 1) > 1:
                        return False
                except Exception:
                    # 无法解析则当作默认 1
                    continue
        return True
    except Exception:
        return False


def _extract_text_from_cell(cell: Any) -> str:
    if isinstance(cell, dict):
        return str(cell.get("text", "")).strip()
    return str(cell or "").strip()


def _trim_table_html(html: str, limit: int = 6000) -> str:
    text = (html or "").strip()
    if len(text) > limit:
        return text[:limit] + "\n..."
    return text


def _build_rows_preview(rows_obj: Any, max_rows: int = 10) -> List[List[str]]:
    preview: List[List[str]] = []
    try:
        for r in (rows_obj or [])[:max_rows]:
            if isinstance(r, list):
                preview.append([_extract_text_from_cell(c) for c in r])
    except Exception:
        return []
    return preview


def _rows_to_simple_html(rows: List[List[str]], max_rows: int = 10) -> str:
    try:
        if not rows:
            return ""
        rows = rows[:max_rows]
        def _tds(arr: List[str]) -> str:
            return "".join(f"<td>{(c or '').strip()}</td>" for c in arr)
        trs = "".join(f"<tr>{_tds(r)}</tr>" for r in rows)
        return f"<table>{trs}</table>"
    except Exception:
        return ""


def _render_rows_text_preview(rows_preview: List[List[str]]) -> str:
    try:
        if not rows_preview:
            return ""
        def _join(line: List[str]) -> str:
            return " | ".join((str(v) if v is not None else "") for v in line)
        if len(rows_preview) == 1:
            return _join(rows_preview[0])
        header = _join(rows_preview[0])
        body = [_join(r) for r in rows_preview[1:]]
        return "\n".join([header, *body])
    except Exception:
        return ""


def _first_row_is_fullwidth_title(rows: Any) -> bool:
    try:
        if not rows or not isinstance(rows, list):
            return False
        r0 = rows[0]
        if not isinstance(r0, list) or len(r0) != 1:
            return False
        c0 = r0[0]
        if not isinstance(c0, dict):
            return False
        cs = c0.get("colspan")
        if cs is None:
            cs = c0.get("col_span", 1)
        try:
            return int(cs or 1) >= 2
        except Exception:
            return False
    except Exception:
        return False


def _summarize_top_rows(rows: Any, max_rows: int = 3) -> str:
    try:
        out: List[str] = []
        for i, r in enumerate((rows or [])[:max_rows]):
            if not isinstance(r, list):
                break
            cells = len(r)
            colspans: List[int] = []
            for c in r:
                if isinstance(c, dict):
                    cs = c.get("colspan")
                    if cs is None:
                        cs = c.get("col_span", 1)
                    try:
                        colspans.append(int(cs or 1))
                    except Exception:
                        colspans.append(1)
            if colspans:
                out.append(f"row{i+1}: cells={cells}, colspan={colspans}")
            else:
                out.append(f"row{i+1}: cells={cells}")
        return "\n".join(out)
    except Exception:
        return ""


_TR_RE = re.compile(r"<tr\b[^>]*>.*?</tr>", re.IGNORECASE | re.DOTALL)


def _trim_html_first_n_tr(html: str, max_tr: int = _BATCH_PER_TABLE_HTML_TR_LIMIT) -> str:
    """从 html 中提取前 max_tr 个 <tr>...</tr>，保留 colspan/rowspan 结构信息。"""
    if not isinstance(html, str) or not html.strip():
        return ""
    try:
        trs = _TR_RE.findall(html)
        if not trs:
            return html.strip()
        picked = trs[:max_tr]
        return "<table>" + "".join(picked) + "</table>"
    except Exception:
        return html.strip()


_tokenizer_instance: Optional[Tokenizer] = None


def _get_tokenizer() -> Tokenizer:
    global _tokenizer_instance
    if _tokenizer_instance is None:
        _tokenizer_instance = Tokenizer()
    return _tokenizer_instance


def _build_batch_entry_body(entry: Dict[str, Any]) -> str:
    """构造批量模式下单张表的正文（不含编号前缀）。包含 HTML 截取、rows 预览、standard2D、结构提示。"""
    rows_local = entry.get("rows") or []
    html_trimmed = _trim_html_first_n_tr(entry.get("html") or "")
    rows_preview = _build_rows_preview(rows_local, max_rows=_BATCH_PER_TABLE_ROWS_PREVIEW)
    snippet = html_trimmed or _rows_to_simple_html(rows_preview, max_rows=_BATCH_PER_TABLE_ROWS_PREVIEW) or "(无 html)"
    rows_preview_text = _render_rows_text_preview(rows_preview) or "(无 rows 预览)"
    is_std = "是" if bool(entry.get("standard2D")) else "否"
    structure_hint = _summarize_top_rows(rows_local, max_rows=3) or "(无结构提示)"
    return (
        f"standard2D={is_std}\n"
        f"HTML 截取(前{_BATCH_PER_TABLE_HTML_TR_LIMIT}行):\n{snippet}\n"
        f"rows 前{_BATCH_PER_TABLE_ROWS_PREVIEW}行预览:\n{rows_preview_text}\n"
        f"顶部结构提示(前3行):\n{structure_hint}\n"
    )


def _build_batch_entry_text(entry: Dict[str, Any], idx_in_batch: int) -> str:
    """构造批量模式下单张表的文本块（含编号前缀，组内从 1 开始）。"""
    body = _build_batch_entry_body(entry)
    return f"[表格{idx_in_batch}]\n{body}"


_BATCH_SYSTEM_PROMPT = "你是一个精确的表格解析助手，只返回结构化结果。"


def _build_batch_prompt(entries_text: List[str]) -> str:
    """构造批量 prompt：多张表拼入同一个请求。"""
    joined = "\n---\n".join(entries_text)
    n = len(entries_text)
    instructions = (
        f"你的任务是为下面 {n} 张表分别判断“顶部连续多少行属于表头/标题”（numRows，0-6）。\n"
        "判定规则（对每张表独立适用）：\n"
        "- 若第一行仅1个单元格且 colspan≥2（整表/分节标题），该行计入 numRows。\n"
        "- 若顶部出现“分组行”（通常含 colspan>1）+ 其下一行为具体列名/单位/指标（叶子列），两者都计入。\n"
        "- 若顶部同时存在 表名 + 分组行 + 叶子列名行，三者全部计入。\n"
        "- 左侧行标题（分行业/分产品/分地区/分销售模式等）不属于列表头。\n"
        "- 如顶部几行都是数据行（业务科目/数值），返回 0。\n"
        "- 数据分节标题行（Assets/Liabilities/资产/负债/收入 等）不属于表头，不计入。\n\n"
        "输出要求：严格按 JSON 格式返回，items 数组长度必须等于输入表格数，id 从 1 开始与输入顺序一致。\n"
        '例如：{"items":[{"id":1,"numRows":1},{"id":2,"numRows":2}]}\n\n'
        f"--- 输入的 {n} 张表 ---\n"
    )
    return instructions + joined


def _group_tables_by_token_budget(
    entry_texts: List[str],
    instruction_reserve: int = _BATCH_PROMPT_INSTRUCTION_RESERVE,
    context_limit: int = _BATCH_CONTEXT_TOKEN_LIMIT,
    max_per_group: int = _BATCH_MAX_TABLES_PER_GROUP,
) -> List[List[int]]:
    """按 token 预算和数量上限将表格索引分组，每组拼出的 prompt 不超过上下文上限且不超过 max_per_group 张。

    返回：每个元素是一个索引列表（对 entry_texts/table_indices 的位置索引）。
    """
    tok = _get_tokenizer()
    budget = max(1, context_limit - instruction_reserve)
    groups: List[List[int]] = []
    current: List[int] = []
    current_tokens = 0
    for i, text in enumerate(entry_texts):
        try:
            n_tokens = tok.count(text)
        except Exception:
            n_tokens = max(1, len(text) // 2)
        # 单张表自身就超预算：单独一组（后续会兜底用单表模式）
        if n_tokens >= budget:
            if current:
                groups.append(current)
                current = []
                current_tokens = 0
            groups.append([i])
            continue
        # token 超预算 或 数量达上限 → 开新组
        if (current_tokens + n_tokens > budget or len(current) >= max_per_group) and current:
            groups.append(current)
            current = [i]
            current_tokens = n_tokens
        else:
            current.append(i)
            current_tokens += n_tokens
    if current:
        groups.append(current)
    return groups


async def _run_batch_extract(
    prompt: str,
    expected_count: int,
    model_id: str,
) -> Optional[List[BatchHeaderItem]]:
    """执行一次批量 LLM 调用，返回结构化结果列表；失败返回 None。"""
    try:
        result = await _ainvoke_table_header_json_schema(
            prompt=prompt,
            model_id=model_id,
            response_type=BatchHeaderResult,
            timeout_s=180.0,
            max_retries=2,
        )
        if not result or not isinstance(result.items, list):
            return None
        return result.items
    except Exception as e:
        logger.warning("表头识别(批量)调用失败: expected={}, error={}", expected_count, e)
        return None


async def adetect_headers_for_items(
    items: List[Dict[str, Any]],
    model_id: str = "model_for_table_header_aided",
    max_concurrency: int = 10,
    batch_mode: bool = False,
) -> Tuple[int, int]:
    """
    异步批量并行：单次 LLM 仅返回“顶部连续行数”（numRows），包含表名/分节标题行在内。
    - 对每个表：LLM 输出 numRows（0-6）；hasHeader = numRows > 0。
    - 基于 numRows 从 rows 顶部直接切片（包含整表/分节标题），构造 headerCells。

    参数：
    - batch_mode: 是否启用批量模式。开启后将多张表按 token 预算分组，一次 LLM 调用返回多表结果。
      失败的组会自动回退到单表调用。

    返回：(处理的表格总数, 检测到表头的表数量)
    """
    if not isinstance(items, list):
        return 0, 0

    # 预处理：仅表格项，计算 standard2D
    table_indices: List[int] = []
    for idx, entry in enumerate(items):
        if not isinstance(entry, dict):
            continue
        t = str(entry.get("type") or "").lower()
        if t in {"table", "excel_table"}:
            rows = entry.get("rows")
            is_standard = _compute_is_standard_2d_from_rows(rows)
            entry["standard2D"] = bool(is_standard)
            table_indices.append(idx)

    if not table_indices:
        return 0, 0

    # 统一构建上下文（精简版：HTML 截前 10 个 <tr>，rows 预览 7 行，与批量模式一致）
    def _build_prompt_context(entry: Dict[str, Any]) -> Tuple[str, str, bool, str]:
        rows_local = entry.get("rows") or []
        table_html_trimmed = _trim_html_first_n_tr(
            entry.get("html") or "", max_tr=_BATCH_PER_TABLE_HTML_TR_LIMIT
        )
        rows_preview = _build_rows_preview(rows_local, max_rows=_BATCH_PER_TABLE_ROWS_PREVIEW)
        snippet = (
            table_html_trimmed
            or _rows_to_simple_html(rows_preview, max_rows=_BATCH_PER_TABLE_ROWS_PREVIEW)
            or "(无 html)"
        )
        rows_preview_text = _render_rows_text_preview(rows_preview) or "(无 rows 预览)"
        is_std_local = bool(entry.get("standard2D"))
        structure_hint = _summarize_top_rows(rows_local, max_rows=3)
        return snippet, rows_preview_text, is_std_local, structure_hint
    system_prompt = "你是一个精确的表格解析助手，只返回结构化结果。"
    extract_prompts: List[str] = []
    for idx in table_indices:
        entry = items[idx]
        snippet, rows_preview_text, is_std, structure_hint = _build_prompt_context(entry)
        if is_std:
            prompt = (
                "你的任务是判断一个【标准表格】顶部连续多少行属于“标题/表头”（包含整表/分节标题与用于标注各列含义的行）。\n\n"
                "规则：\n"
                "- 若第一行仅1个单元格且 colspan≥2（整表/分节标题），它也计入 numRows。\n"
                "- 标准表通常 1 行；若包含表名或确有多级列头，可返回 2 或更大。\n"
                "- 若出现“分组行”（包含多个列分组，通常有 colspan>1），且其下一行为具体列名/单位/指标（叶子列），则二者均计入。\n"
                "- 若顶部同时存在表名/分节标题 + 分组行 + 叶子列名行，应将三者全部计入（例如：numRows=3）。\n"
                "- 左侧行标题（如“分行业/分产品/分地区/分销售模式”）不属于列表头，但其存在不否定表头。\n"
                "- 如果顶部几行都是数据而非列名，返回 0。\n\n"
                "【重要排除】以下行不属于表头，不计入 numRows：\n"
                "- 数据分节标题行，例如 'Assets'、'Liabilities'、'Revenue'、'资产'、'负债'、'收入' 等。\n"
                "  这类行是数据的分类标签，不是列标题，即使出现在表格顶部也不计入。\n"
                "- 判断方法：若某行内容是业务科目/概念（而非年份/单位/指标名），且其下方紧跟数值行，则它是数据行。\n\n"
                "--- [表格数据开始] ---\n"
                "表格（HTML 截取）：\n"
                f"{snippet}\n\n"
                "rows 前几行（预览）：\n"
                f"{rows_preview_text}\n\n"
                "结构提示（来自 rows 顶部）：\n"
                f"{structure_hint}\n"
                "--- [表格数据结束] ---\n\n"
                "请仅返回一个合法 JSON 对象：\n"
                "{\"numRows\": <int>}"
            )
        else:
            prompt = (
                "你的任务是判断一个【复杂/多级表格】顶部连续多少行属于“标题/表头”。\n\n"
                "严格规则：\n"
                "- 若第一行仅1个单元格且 colspan≥2（整表/分节标题），它也计入 numRows。\n"
                "- 复杂表通常包含多级列头与分节标题，优先考虑返回 ≥2；若仅观察到单层且为清晰列名，可返回 1。\n"
                "- 如存在“分组行”（包含多个列分组，常见有 colspan>1），其下一行通常为具体列名/单位/指标（叶子列），二者均计入。\n"
                "- 若顶部同时存在表名/分节标题 + 分组行 + 叶子列名行，应将三者全部计入（例如：numRows=3）。\n"
                "- 左侧的行标题（如“分行业/分产品/分地区/分销售模式”）不属于列表头；多级表头各层级均计入。\n"
                "- 顶部若为数据而非列名，返回 0。\n\n"
                "【重要排除】以下行不属于表头，不计入 numRows：\n"
                "- 数据分节标题行，例如 'Assets'、'Liabilities'、'Revenue'、'资产'、'负债'、'收入' 等。\n"
                "  这类行是数据的分类标签，不是列标题，即使出现在表格顶部也不计入。\n"
                "- 判断方法：若某行内容是业务科目/概念（而非年份/单位/指标名），且其下方紧跟数值行，则它是数据行。\n\n"
                "--- [表格数据开始] ---\n"
                "表格（HTML 截取）：\n"
                f"{snippet}\n\n"
                "rows 前几行（预览）：\n"
                f"{rows_preview_text}\n\n"
                "结构提示（来自 rows 顶部）：\n"
                f"{structure_hint}\n"
                "--- [表格数据结束] ---\n\n"
                "请仅返回一个合法 JSON 对象：\n"
                "{\"numRows\": <int>}"
            )
        extract_prompts.append(f"{system_prompt}\n\n{prompt}")

    extract_results_raw: List[Optional[HeaderRowCount]] = [None] * len(table_indices)
    total_tables = len(table_indices)
    llm_call_count = 0
    t_llm_start = time.monotonic()

    async def _run_one_extract(full_prompt: str, table_idx: int) -> Optional[HeaderRowCount]:
        sem2 = _extract_sem  # type: ignore[name-defined]
        async with sem2:
            try:
                result = await _ainvoke_table_header_json_schema(
                    prompt=full_prompt,
                    model_id=model_id,
                    response_type=HeaderRowCount,
                    timeout_s=120.0,
                    max_retries=2,
                    temperature=0.01,
                )
                return result
            except Exception as e:
                logger.warning("表头识别(LLM)失败: table_idx={}, error={}", table_idx, e)
                return None

    try:
        _extract_sem = asyncio.Semaphore(max(1, int(max_concurrency or 1)))

        if batch_mode and total_tables > 1:
            # 先构造 body（不含编号），用于 token 估算和分组
            entry_bodies: List[str] = [
                _build_batch_entry_body(items[idx]) for idx in table_indices
            ]
            groups = _group_tables_by_token_budget(entry_bodies)
            logger.info(
                "表头识别开始(批量): 共{}张表, 分组数={}, model_id={}",
                total_tables, len(groups), model_id,
            )

            # 每个组 1 次 LLM 调用；并发执行多个组
            batch_sem = asyncio.Semaphore(max(1, int(max_concurrency or 1)))

            async def _run_one_group(pos_list: List[int]) -> Tuple[List[int], Optional[List[BatchHeaderItem]]]:
                async with batch_sem:
                    # 组内只有 1 张表时仍然走单表模式（prompt 更精确）
                    if len(pos_list) == 1:
                        p = pos_list[0]
                        r = await _run_one_extract(extract_prompts[p], table_indices[p])
                        if r is None:
                            return pos_list, None
                        return pos_list, [BatchHeaderItem(id=1, numRows=int(getattr(r, "numRows", 0) or 0))]

                    # 组内按 1..N 重新编号，保证与模型期望的 id 对齐
                    sub_entry_texts = [
                        f"[表格{local_id}]\n{entry_bodies[p]}"
                        for local_id, p in enumerate(pos_list, start=1)
                    ]
                    prompt_text = _build_batch_prompt(sub_entry_texts)
                    full_prompt = f"{_BATCH_SYSTEM_PROMPT}\n\n{prompt_text}"
                    items_result = await _run_batch_extract(full_prompt, len(pos_list), model_id)
                    return pos_list, items_result

            group_results = await asyncio.gather(
                *(_run_one_group(g) for g in groups),
                return_exceptions=False,
            )
            llm_call_count = len(groups)

            # 写回结果；失败的组自动回退到单表调用
            fallback_pos: List[int] = []
            for pos_list, items_result in group_results:
                if items_result is None:
                    fallback_pos.extend(pos_list)
                    continue
                # 按 id (从 1 开始) 对齐到组内位置
                id_to_num: Dict[int, int] = {int(it.id): int(it.numRows) for it in items_result}
                missing_positions: List[int] = []
                for local_idx, p in enumerate(pos_list):
                    local_id = local_idx + 1
                    if local_id in id_to_num:
                        extract_results_raw[p] = HeaderRowCount(numRows=id_to_num[local_id])
                    else:
                        missing_positions.append(p)
                if missing_positions:
                    logger.warning(
                        "表头识别(批量)部分缺失结果: missing={}, group_size={}",
                        len(missing_positions), len(pos_list),
                    )
                    fallback_pos.extend(missing_positions)

            if fallback_pos:
                logger.info("表头识别(批量)回退单表: fallback={}", len(fallback_pos))
                fb_results = await asyncio.gather(
                    *(_run_one_extract(extract_prompts[p], table_indices[p]) for p in fallback_pos),
                    return_exceptions=False,
                )
                for p, r in zip(fallback_pos, fb_results):
                    extract_results_raw[p] = r
                llm_call_count += len(fallback_pos)
        else:
            logger.info(
                "表头识别开始: 共{}张表, LLM调用={}, model_id={}",
                total_tables, total_tables, model_id,
            )
            extract_results_raw = list(
                await asyncio.gather(
                    *(_run_one_extract(p, ti) for p, ti in zip(extract_prompts, table_indices)),
                    return_exceptions=False,
                )
            )
            llm_call_count = total_tables

        t_llm_elapsed = time.monotonic() - t_llm_start
    except Exception as e:
        logger.warning("表头识别批量调用失败: {}", e)
        return len(table_indices), 0

    # 写回 hasHeader；headerCells 的构造延迟到跨页合并之后完成，避免基于合并前的 rows 出错
    detected = 0
    for idx, res in zip(table_indices, extract_results_raw):
        try:
            num_rows = int(getattr(res, "numRows", 0) or 0) if res is not None else 0
            items[idx]["hasHeader"] = bool(num_rows > 0)
            # 将 numRows 暂存到 item 上，待跨页合并完成后由 build_header_cells_from_pending 统一构造 headerCells
            if num_rows > 0:
                items[idx]["_pendingHeaderNumRows"] = num_rows
                # 如存在旧的 headerCells（例如重跑场景），先清掉，避免和最终结果混淆
                items[idx].pop("headerCells", None)
                detected += 1
            else:
                items[idx].pop("_pendingHeaderNumRows", None)
                items[idx].pop("headerCells", None)
        except Exception:
            items[idx].pop("_pendingHeaderNumRows", None)
            items[idx].pop("headerCells", None)

    logger.info(
        "表头识别汇总: 共{}张表, LLM调用={}, 检测到表头={}, LLM总耗时={:.2f}s | 批量={}",
        len(table_indices), llm_call_count, detected, t_llm_elapsed,
        "开启" if (batch_mode and total_tables > 1) else "关闭",
    )
    return len(table_indices), detected


def _first_non_empty_cell_text(row: Any) -> str:
    if not isinstance(row, list):
        return ""
    for cell in row:
        if not isinstance(cell, dict):
            text = str(cell or "").strip()
        else:
            text = str(cell.get("text") or cell.get("value") or cell.get("md") or "").strip()
        if text:
            return text
    return ""


def _is_number_like_text(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    if compact in {"-", "—", "–"}:
        return True
    return bool(re.fullmatch(r"[()\-+\d,.'’]+", compact))


def _row_has_amount_after_subject(row: Any) -> bool:
    if not isinstance(row, list) or len(row) < 2:
        return False
    first_text = _first_non_empty_cell_text(row)
    if not first_text or _is_number_like_text(first_text):
        return False
    amount_like = 0
    for cell in row[1:]:
        text = str(cell.get("text") or "").strip() if isinstance(cell, dict) else str(cell or "").strip()
        if _is_number_like_text(text):
            amount_like += 1
    return amount_like >= 1


def _correct_header_num_rows(num_rows: int, rows: Any) -> int:
    if num_rows <= 0 or not isinstance(rows, list) or not rows:
        return num_rows
    first_text = _first_non_empty_cell_text(rows[0])
    data_section_labels = {
        "assets",
        "liabilities",
        "equity",
        "revenue",
        "income",
        "expenses",
        "cash flows from operating activities",
        "cash flows from investing activities",
        "cash flows from financing activities",
    }
    normalized_first = re.sub(r"\s+", " ", first_text).strip().lower()
    if _row_has_amount_after_subject(rows[0]) or normalized_first in data_section_labels:
        return 0
    return num_rows


def build_header_cells_from_pending(items: List[Dict[str, Any]]) -> int:
    """基于各 item 上暂存的 `_pendingHeaderNumRows`，从最终 rows 顶部切片构造 headerCells。

    用法：在 `adetect_headers_for_items` 之后、且跨页表合并/其他会修改 rows 的流程之后调用，
    以保证 headerCells 与最终 rows 保持一致。

    行为：
    - 对 type 为 table/excel_table 且 `_pendingHeaderNumRows` > 0 的 item，从当前 rows 顶部取 numRows 行构造 headerCells；
    - 若当前 rows 行数少于 numRows，则使用实际行数（向下截断）；
    - 未命中条件的 item，清除 `_pendingHeaderNumRows` / `headerCells`，并根据情况回写 `hasHeader=False`；
    - 返回成功构造 headerCells 的表格数量。
    """
    if not isinstance(items, list):
        return 0

    built_count = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") not in ("table", "excel_table"):
            continue

        num_rows_raw = item.pop("_pendingHeaderNumRows", None)
        try:
            num_rows = int(num_rows_raw) if num_rows_raw is not None else 0
        except Exception:
            num_rows = 0

        if num_rows <= 0:
            # 没有暂存 numRows 时，保持现有 hasHeader/headerCells 不变
            continue

        src_rows = item.get("rows") or []
        if not isinstance(src_rows, list) or not src_rows:
            item["hasHeader"] = False
            item.pop("headerCells", None)
            continue

        effective_num_rows = num_rows

        header_cells: List[List[Dict[str, Any]]] = []
        for r in src_rows[:effective_num_rows]:
            if not isinstance(r, list):
                continue
            row_out: List[Dict[str, Any]] = []
            for c in r:
                if isinstance(c, dict):
                    text = str(c.get("text", "")).strip()
                    rs = c.get("rowspan")
                    cs = c.get("colspan")
                    if rs is None:
                        rs = c.get("row_span", 1)
                    if cs is None:
                        cs = c.get("col_span", 1)
                    try:
                        rs_i = int(rs or 1)
                    except Exception:
                        rs_i = 1
                    try:
                        cs_i = int(cs or 1)
                    except Exception:
                        cs_i = 1
                    cell_obj: Dict[str, Any] = {"text": text, "rowspan": rs_i, "colspan": cs_i}
                    # 透传 rowstart/colstart（兼容不同命名）
                    rstart = c.get("rowstart")
                    if rstart is None:
                        rstart = c.get("rowStart")
                    if rstart is None:
                        rstart = c.get("row_start")
                    cstart = c.get("colstart")
                    if cstart is None:
                        cstart = c.get("colStart")
                    if cstart is None:
                        cstart = c.get("col_start")
                    if rstart is not None:
                        try:
                            cell_obj["rowstart"] = int(rstart)
                        except Exception:
                            pass
                    if cstart is not None:
                        try:
                            cell_obj["colstart"] = int(cstart)
                        except Exception:
                            pass
                    row_out.append(cell_obj)
                else:
                    row_out.append({"text": str(c or "").strip(), "rowspan": 1, "colspan": 1})
            if row_out:
                header_cells.append(row_out)

        if header_cells:
            item["headerCells"] = header_cells
            item["hasHeader"] = True
            built_count += 1
        else:
            item["hasHeader"] = False
            item.pop("headerCells", None)

    return built_count



__all__ = [
    "adetect_headers_for_items",
    "build_header_cells_from_pending",
]

