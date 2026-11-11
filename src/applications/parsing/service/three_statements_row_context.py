"""三大表行级结构字段生成模块。

从三大表的原始行列表中提取行级结构信息：行类型、缩进层级、编号前缀、
前后行名称、推断父级等字段。

- lineType HEADER 由上游 lineHeaderRowCount 字段判定（不使用规则）
- indentLevel 由大模型结合表格截图、行 bbox、金额列和文本特征综合判定
"""

from __future__ import annotations

import base64
import json
import re
import unicodedata
from typing import Any, Optional

from loguru import logger
from pydantic import RootModel

_WHITESPACE_RE = re.compile(r"\s+")
_CJK_PUNCT_TRANS = str.maketrans(
    {k: v for k, v in {"。": ".", "【": "[", "】": "]", "《": "<", "》": ">", "、": ","}.items()}
)

_NUMBERING_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*([一二三四五六七八九十]+[、.])"),
    re.compile(r"^\s*(（[一二三四五六七八九十]+）)"),
    re.compile(r"^\s*(\([一二三四五六七八九十]+\))"),
    re.compile(r"^\s*(\d+[.)、])"),
    re.compile(r"^\s*(\(\d+\))"),
    re.compile(r"^\s*([A-Za-z][.)])"),
    re.compile(r"^\s*([\-–—]+)"),
]

_SUBTOTAL_RE = re.compile(r"(小计|subtotal)", re.IGNORECASE)
_TOTAL_RE = re.compile(r"(合计|总计|total)", re.IGNORECASE)
_NOTE_RE = re.compile(r"^(附注|notes?|\*|#|注：)", re.IGNORECASE)
_BOLD_FONT_RE = re.compile(r"(bold|black|heavy|semibold|demibold|extrabold)", re.IGNORECASE)
_ITALIC_FONT_RE = re.compile(r"(italic|oblique)", re.IGNORECASE)

_VLM_MODEL_ID = "three_statements_indent_vlm"
_STYLE_OVERLAP_THRESHOLD = 0.95


class IndentVlmResponse(RootModel[dict[str, int]]):
    pass


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

async def build_lines_context(
    raw_rows: list,
    table_el: dict[str, Any],
    *,
    file_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """三大表行级结构字段生成主入口。

    依次执行：行构建 → 前后行名称 → VLM 缩进预测 → 父级兜底推断。

    Args:
        raw_rows: 三大表提取接口返回中的rows
        table_el: 表格元素字典（含 lineHeaderRowCount、table_type、bBox 等字段）。
        file_id: 文件 ID，传入后将尝试从 MinIO 下载表格图片并用 VLM 判断层级。

    Returns:
        list[dict]: 行级字段字典列表。
    """
    line_header_row_count = _get_line_header_row_count(table_el)
    page_at = _get_table_page(table_el)

    lines = [_build_line(row_index, cells, page_at, line_header_row_count) for row_index, cells in enumerate(raw_rows)]
    _fill_preceding_following(lines)
    await _fill_pdf_styles(lines, file_id)

    if file_id:
        await _predict_indent_levels_with_vlm(lines, table_el, file_id)
        _backfill_empty_line_types(lines)
    _fill_inferred_parents(lines)

    for line in lines:
        line.pop("_bbox", None)
    return lines


# ---------------------------------------------------------------------------
# 辅助：表格级字段
# ---------------------------------------------------------------------------

def _get_line_header_row_count(table_el: dict[str, Any]) -> int:
    # 只信上游对“当前 rows”的显式标记；原始表 hasHeader 不能代表清洗后的 rows[0] 仍是表头。
    if not isinstance(table_el, dict):
        return 0
    try:
        return max(0, int(table_el.get("lineHeaderRowCount") or 0))
    except (TypeError, ValueError):
        return 0


def _get_table_page(table_el: dict[str, Any]) -> Optional[int]:
    v = table_el.get("pageAt")
    if v is not None:
        try:
            return int(v)
        except (TypeError, ValueError):
            pass
    return None


# ---------------------------------------------------------------------------
# 行构建
# ---------------------------------------------------------------------------

def _build_line(
    row_index: int,
    cells: list[dict],
    fallback_page: Optional[int],
    line_header_row_count: int,
) -> dict[str, Any]:
    """从单行单元格列表构建行级字段字典。

    Args:
        row_index: 行序号，0-based。
        cells: 当前行的单元格字典列表。
        fallback_page: 兜底页码。
        line_header_row_count: 当前 rows 中明确属于表头的行数。

    Returns:
        dict: 行级字段字典，其中 precedingName/followingName/inferredParent 等后续填充。
    """
    subject_cell = _find_subject_cell(cells)
    raw_name = (subject_cell.get("text") or "").strip() if subject_cell else ""
    amount_texts = _amount_texts(cells)
    numbering_prefix = _extract_numbering_prefix(raw_name)
    cleaned_name = _clean_name(raw_name)

    line_type, signals = _infer_line_type(cells, raw_name, row_index, line_header_row_count)
    page_number = subject_cell.get("pageAt") if subject_cell else fallback_page
    bbox = subject_cell.get("bBox") if subject_cell else None
    indent_level = -1 if line_type in ("BLANK", "HEADER") else 1

    return {
        "rowIndex": row_index,
        "pageNumber": page_number,
        "rawName": raw_name,
        "cleanedName": cleaned_name,
        "rawAmounts": amount_texts,
        "numberingPrefix": numbering_prefix,
        "lineType": line_type,
        "lineTypeSignals": signals,
        "precedingName": None,
        "followingName": None,
        "fontSize": None,
        "isBold": None,
        "isItalic": None,
        "styleSource": "UNAVAILABLE_IMAGE_ONLY",
        "indentLevel": indent_level,
        "indentSource": "",
        "inferredParent": None,
        "_bbox": bbox,
    }


# ---------------------------------------------------------------------------
# 单元格查找
# ---------------------------------------------------------------------------

def _find_subject_cell(cells: list[dict]) -> Optional[dict]:
    for cell in cells:
        if cell.get("colStart") == 0:
            return cell
    return cells[0] if cells else None


def _amount_texts(cells: list[dict]) -> list[str]:
    return [cell.get("text", "").strip() for cell in _amount_cells(cells)]


def _amount_cells(cells: list[dict]) -> list[dict]:
    data_cells: list[tuple[int, dict]] = []
    for fallback_col_index, cell in enumerate(cells):
        col_key = cell.get("colStart")
        if col_key is None:
            col_key = fallback_col_index
        if col_key is not None and col_key > 0:
            data_cells.append((col_key, cell))

    # 上游三大表提取已删除辅助列（行次/序号/Note），无需再猜测附注引用编号
    return [cell for _, cell in data_cells]


def _is_note_reference_text(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and stripped.isdigit() and len(stripped) <= 3


# ---------------------------------------------------------------------------
# PDF 文本层样式
# ---------------------------------------------------------------------------

async def _fill_pdf_styles(lines: list[dict[str, Any]], file_id: Optional[str]) -> None:
    if not file_id:
        return
    try:
        pdf_bytes = await _load_original_pdf_bytes(file_id)
    except Exception:
        logger.exception("加载原始 PDF 样式失败 | file_id={}", file_id)
        return
    if not pdf_bytes:
        return

    try:
        spans_by_page = _extract_pdf_spans(pdf_bytes)
    except Exception:
        logger.exception("提取 PDF 文本层样式失败 | file_id={}", file_id)
        return
    if not spans_by_page:
        return

    for line in lines:
        line["styleSource"] = "UNMATCHED"
        match = _match_subject_style(line, spans_by_page)
        if not match:
            continue
        line["fontSize"] = match["fontSize"]
        line["isBold"] = match["isBold"]
        line["isItalic"] = match["isItalic"]
        line["styleSource"] = "PDF_SPAN"


async def _load_original_pdf_bytes(file_id: str) -> Optional[bytes]:
    from src.applications.parsing.utils.mineru_storage import load_original_pdf_bytes

    return await load_original_pdf_bytes(file_id)


def _extract_pdf_spans(pdf_bytes: bytes) -> dict[int, dict[str, Any]]:
    import fitz  # type: ignore

    pages: dict[int, dict[str, Any]] = {}
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_index, page in enumerate(doc, start=1):
            spans: list[dict[str, Any]] = []
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        bbox = span.get("bbox")
                        text = str(span.get("text") or "").strip()
                        if text and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                            spans.append(span)
            pages[page_index] = {"height": float(page.rect.height), "spans": spans}
    return pages


def _match_subject_style(
    line: dict[str, Any],
    spans_by_page: dict[int, dict[str, Any]],
) -> Optional[dict[str, Any]]:
    page_number = line.get("pageNumber")
    if page_number is None:
        return None
    try:
        page_number = int(page_number)
    except (TypeError, ValueError):
        return None

    page = spans_by_page.get(page_number)
    if not page:
        return None
    bbox = _normalize_bbox(line.get("_bbox"), page.get("height"))
    if not bbox:
        return None

    best: Optional[dict[str, Any]] = None
    best_ratio = 0.0
    for span in page.get("spans", []):
        span_bbox = _normalize_bbox(span.get("bbox"), None)
        if not span_bbox:
            continue
        ratio = _overlap_ratio(span_bbox, bbox)
        if ratio >= _STYLE_OVERLAP_THRESHOLD and ratio > best_ratio:
            best = span
            best_ratio = ratio
    return _span_style(best) if best else None


def _span_style(span: dict[str, Any]) -> dict[str, Any]:
    import fitz  # type: ignore

    flags = int(span.get("flags") or 0)
    font_name = str(span.get("font") or "")
    return {
        "fontSize": span.get("size"),
        "isBold": bool(flags & fitz.TEXT_FONT_BOLD) or bool(_BOLD_FONT_RE.search(font_name)),
        "isItalic": bool(flags & fitz.TEXT_FONT_ITALIC) or bool(_ITALIC_FONT_RE.search(font_name)),
    }


def _normalize_bbox(value: Any, page_height: Optional[float]) -> Optional[tuple[float, float, float, float]]:
    try:
        if isinstance(value, dict):
            l, t, r, b = value["l"], value["t"], value["r"], value["b"]
            coord_origin = value.get("coordOrigin") or value.get("coord_origin")
        elif isinstance(value, (list, tuple)) and len(value) == 4:
            l, t, r, b = value
            coord_origin = None
        else:
            return None
        left, top, right, bottom = float(l), float(t), float(r), float(b)
    except (KeyError, TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    if str(coord_origin).upper() == "BOTTOMLEFT" and page_height:
        top, bottom = page_height - bottom, page_height - top
    return left, top, right, bottom


def _overlap_ratio(
    span_bbox: tuple[float, float, float, float],
    cell_bbox: tuple[float, float, float, float],
) -> float:
    left = max(span_bbox[0], cell_bbox[0])
    top = max(span_bbox[1], cell_bbox[1])
    right = min(span_bbox[2], cell_bbox[2])
    bottom = min(span_bbox[3], cell_bbox[3])
    if right <= left or bottom <= top:
        return 0.0
    span_area = (span_bbox[2] - span_bbox[0]) * (span_bbox[3] - span_bbox[1])
    return ((right - left) * (bottom - top)) / span_area if span_area > 0 else 0.0


# ---------------------------------------------------------------------------
# 名称清理
# ---------------------------------------------------------------------------

def _extract_numbering_prefix(text: str) -> str:
    """从科目名开头提取编号前缀。

    支持 "一、"、"（一）"、"1."、"(1)"、"A." 等格式。

    Args:
        text: 原始科目名文本。

    Returns:
        str: 编号前缀字符串，无匹配返回空字符串。
    """
    for pattern in _NUMBERING_PATTERNS:
        m = pattern.match(text)
        if m:
            return m.group(1)
    return ""


def _clean_name(text: str) -> str:
    """清洗科目名：去空白 → Unicode NFKC 全角归一化 → CJK 标点映射 → 小写。

    NFKC 覆盖全角 ASCII 标点（，！？；：（）等→半角）、全角字母数字（Ａａ０→Aa0）、
    弯引号（""''→\"\"''），仅剩余【】《》、等 CJK 专属标点需自维护。

    Args:
        text: 原始科目名文本。

    Returns:
        str: 清洗后的科目名。
    """
    if not text:
        return ""
    text = _WHITESPACE_RE.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_CJK_PUNCT_TRANS)
    return text.lower()


# ---------------------------------------------------------------------------
# 行类型判定
# ---------------------------------------------------------------------------

def _infer_line_type(
    cells: list[dict],
    raw_name: str,
    row_index: int,
    line_header_row_count: int,
) -> tuple[str, dict[str, Any]]:
    """按优先级判定行类型。

    判定顺序：BLANK → HEADER(依 lineHeaderRowCount) → SUBTOTAL → TOTAL → NOTE → ITEM。

    Args:
        cells: 当前行的单元格字典列表。
        raw_name: 原始科目名。
        row_index: 行序号。
        line_header_row_count: 当前 rows 中明确属于表头的行数。

    Returns:
        tuple[str, dict]: (lineType, lineTypeSignals)。
    """
    signals: dict[str, Any] = {}

    if all(not cell.get("text") for cell in cells):
        return "BLANK", {"is_blank_row": True}

    # HEADER 判定不做本地猜测，避免把 Assets/Liabilities 这类分节行误当表头。
    if row_index < line_header_row_count:
        return "HEADER", {"from_line_header_row_count": True}

    signals["has_amount"] = any(c.get("text") for c in _amount_cells(cells))

    if _SUBTOTAL_RE.search(raw_name):
        return "SUBTOTAL", {"has_subtotal_keyword": True, **signals}

    if _TOTAL_RE.search(raw_name):
        return "TOTAL", {"has_total_keyword": True, **signals}

    if _NOTE_RE.match(raw_name):
        return "NOTE", {"is_note_like": True, **signals}

    return "ITEM", signals


# ---------------------------------------------------------------------------
# 前后行名称
# ---------------------------------------------------------------------------

def _fill_preceding_following(lines: list[dict[str, Any]]) -> None:
    """为每行填充 precedingName 和 followingName。

    从有效行（非 BLANK、非 HEADER）中取最近的前后 rawName，原地修改。

    Args:
        lines: 行级字段字典列表。
    """
    valid_names = [
        (i, line["rawName"]) for i, line in enumerate(lines)
        if line["lineType"] not in ("BLANK", "HEADER")
    ]
    for i, line in enumerate(lines):
        pred = _nearest(valid_names, i, -1)
        foll = _nearest(valid_names, i, 1)
        line["precedingName"] = pred
        line["followingName"] = foll


def _nearest(pairs: list[tuple[int, str]], idx: int, step: int) -> Optional[str]:
    """在有序 pair 列表中查找距 idx 最近的前或后项。

    Args:
        pairs: (行号, 名称) 有序列表。
        idx: 目标行号。
        step: -1 向前，1 向后。

    Returns:
        Optional[str]: 最近的名称，无匹配返回 None。
    """
    if step == -1:
        for origin_idx, name in reversed(pairs):
            if origin_idx < idx:
                return name
    else:
        for origin_idx, name in pairs:
            if origin_idx > idx:
                return name
    return None


# ---------------------------------------------------------------------------
# VLM 缩进层级预测
# ---------------------------------------------------------------------------

async def _predict_indent_levels_with_vlm(
    lines: list[dict[str, Any]],
    table_el: dict[str, Any],
    file_id: str,
) -> None:
    """使用 VLM 基于表格截图和结构化行数据预测 indentLevel。"""
    candidates = [
        line for line in lines
        if line["lineType"] not in ("BLANK", "HEADER", "SUBTOTAL")
    ]
    if not candidates:
        return

    image_data_url = await _load_table_image_data_url(table_el, file_id)
    if not image_data_url:
        logger.warning("VLM 缩进预测跳过：表格图片不存在 | file_id={} img_path={}", file_id, table_el.get("img_path"))
        return

    statement_title = str(table_el.get("table_type") or table_el.get("text") or "")
    prompt = _build_vlm_indent_prompt(candidates, statement_title)

    try:
        response = await _invoke_indent_vlm(prompt, image_data_url)

        logger.info(
    "VLM 缩进预测结果 | file_id={} img_path={} table_type={} rows={} response={}",
    file_id,
    table_el.get("img_path"),
    table_el.get("table_type"),
    len(candidates),
    response.model_dump(),
)
        prediction = {
            str(row_index): {"indentLevel": indent_level, "inferredParent": None}
            for row_index, indent_level in response.root.items()
        }
    except Exception:
        logger.exception("VLM 缩进预测失败 | file_id={} img_path={}", file_id, table_el.get("img_path"))
        return

    if not prediction:
        return

    _apply_vlm_indent_prediction(candidates, prediction)


async def _load_table_image_data_url(table_el: dict[str, Any], file_id: str) -> Optional[str]:
    img_path = table_el.get("img_path")
    if not isinstance(img_path, str) or not img_path.strip():
        return None

    from src.applications.parsing.utils.mineru_storage import download_mineru_image

    try:
        image_bytes = await download_mineru_image(file_id, img_path.strip())
    except Exception:
        logger.exception("下载 MinerU 表格图片失败 | file_id={} img_path={}", file_id, img_path)
        return None
    if not image_bytes:
        return None
    mime_type = _detect_image_mime_type(image_bytes, img_path)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _detect_image_mime_type(image_bytes: bytes, img_path: str) -> str:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    lower = img_path.lower()
    if lower.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lower.endswith(".webp"):
        return "image/webp"
    return "image/png"


async def _invoke_indent_vlm(prompt: str, image_data_url: str) -> "IndentVlmResponse":
    from src.agents.content_generate_agent.utils import ainvoke_llm_with_messages_structured_output

    return await ainvoke_llm_with_messages_structured_output(
        messages=[
            ("user", [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]),
        ],
        model_id=_VLM_MODEL_ID,
        response_type=IndentVlmResponse,
    )


def _apply_vlm_indent_prediction(lines: list[dict[str, Any]], prediction: dict[str, Any]) -> None:
    for line in lines:
        row_index = line.get("rowIndex")
        row_prediction = prediction.get(str(row_index))
        if not isinstance(row_prediction, dict):
            continue
        level = row_prediction.get("indentLevel")
        try:
            level = int(level)
        except (TypeError, ValueError):
            continue
        level = max(1, min(4, level))
        line["indentLevel"] = level
        line["indentSource"] = "VLM"
        line["inferredParent"] = None


def _build_vlm_indent_prompt(rows: list[dict[str, Any]], statement_title: str) -> str:
    """构建 VLM 缩进层级预测 prompt。

    输入每行的 rowIndex、rawName、numberingPrefix、lineType、precedingName、followingName、
    subjectBBox、amountColumnTexts、amountColumnsAllEmpty；表格图片作为多模态消息附带。

    Args:
        rows: 待预测的行列表（非 BLANK/HEADER）。
        statement_title: 报表标题。

    Returns:
        str: 完整 prompt 文本。
    """
    llm_rows = [
        {
            "rowIndex": row["rowIndex"],
            "rawName": row["rawName"],
            "numberingPrefix": row["numberingPrefix"],
            "lineType": row["lineType"],
            "precedingName": row.get("precedingName"),
            "followingName": row.get("followingName"),
            "subjectBBox": row.get("_bbox"),
            "amountColumnTexts": row.get("rawAmounts") if isinstance(row.get("rawAmounts"), list) else [],
            "amountColumnsAllEmpty": _amount_columns_all_empty(row.get("rawAmounts")),
        }
        for row in rows
    ]
    payload = {"statementTitle": statement_title, "rows": llm_rows}
    return f"""
# 角色
你是专业的财务报表结构分析助手。你将收到一张三大表的表格截图，以及该表中"科目列"的结构化行列表。
你的任务是结合表格截图、行 bbox、行名、视觉缩进、编号前缀、金额列是否彻底为空和上下文顺序，只判断每一行的 indentLevel。

# 层级定义
- indentLevel=1 表示最顶层科目或分组。
- indentLevel=2 表示比上一层更细一级的科目或明细。
- indentLevel=3 表示更细的明细科目。
- 数字越大，层级越深。

# 判断规则（按顺序执行）
1. 分组父级：金额列全空且语义是财报分组的行，才可作为后续行父级；其后连续从属项层级加一。金额列非空的行通常是实际科目，不作为后续行父级。
2. 视觉优先：科目列缩进是主要证据（total行除外）；字号加粗、横线、空行只作辅助信号，不能单独改变层级。
3. Total/合计行：先判断它汇总的连续块。若汇总某个分组下的明细，层级与该分组下的主要明细同级；若汇总整张表或当前最高层分区，层级与该最高层分区同级。不要仅因含 Total/合计、加粗或有横线就判为 1。
4. 同名前缀：不要仅因后一行 rawName 以前一行开头、包含破折号后缀、类别编号或证券类型，就判为子级；若这些行左边界接近且金额列非空，应判为同级。
5. 冒号分组：以冒号结尾或语义为“attributable to / per share / dividends per share”等分组时，其下明显从属项为下一层。
6. 层级不能无证据跳跃；相邻行层级数最多加一。

# Few-shot
- Section total: Assets=1, Cash=2, Other assets=2, Total assets=2；ASSETS=1, Non-current assets=2, Property plant and equipment=3, Current assets=2, Cash and bank balances=3, TOTAL ASSETS=1。
- Balance sheet group: EQUITY AND LIABILITIES=1, Equity=2, Share capital=3, Share capital - RCPS-i A=3, Share capital - RCPS-i C=3, Reserve=3, Retained earnings=3, Non-current liabilities=2, Long-term borrowings=3, Current liabilities=2, Trade payables=3, Total liabilities=2。
- Flat income list: Operating revenue=1, Interest income=1, Net interest income=1, Total operating income=1。
- Colon group: Attributable to:=1, Owners=2, Non-controlling interests=2。
- Detail group: Earnings per share=1, Basic=2, Diluted=2；Dividends per share=1, Paid/Declared 明细=2。

# 输出要求
- 只输出 JSON 对象，不要解释，不要 Markdown。
- 输出格式为 rowIndex 到 indentLevel 的 JSON 对象：
  {{"1": 1, "2": 1, "3": 2, "4": 2}}
- key 必须是输入中的 rowIndex 转成字符串，value 必须是该行的 indentLevel 整数。
- 必须覆盖所有输入 rowIndex，不得新增、删除或改名。
- indentLevel 必须是整数，禁止 null。
- 层级建议控制在 1-4 之间。

# 输入：
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def _amount_columns_all_empty(amount_column_texts: Any) -> bool:
    """判断金额列是否彻底为空。

    只把 None 或去空白后的空字符串视为空；"-" 等占位符是有效单元格内容。
    """
    if not isinstance(amount_column_texts, list):
        return True
    return all(text is None or (isinstance(text, str) and not text.strip()) for text in amount_column_texts)


def _backfill_empty_line_types(lines: list[dict[str, Any]]) -> None:
    for line in lines:
        if line.get("lineType") != "ITEM":
            continue
        all_empty = _amount_columns_all_empty(line.get("rawAmounts"))
        line.setdefault("lineTypeSignals", {})["amount_columns_all_empty"] = all_empty
        if not all_empty or not str(line.get("rawName") or "").strip():
            continue
        line["lineType"] = "EMPTY"


def _parse_vlm_indent_json(content: str) -> dict[str, Any]:
    """从 VLM 响应文本中解析 indentLevel JSON。

    处理 think 标签、markdown 代码块包裹、裸 JSON 对象等格式。

    Args:
        content: VLM 响应文本。

    Returns:
        dict: {"rowIndex": {"indentLevel": int|null, "inferredParent": int|null}, ...}。
    """
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            return {}
        try:
            result = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    if not isinstance(result, dict):
        return {}
    rows = result.get("rows")
    if isinstance(rows, list):
        normalized: dict[str, Any] = {}
        for item in rows:
            if not isinstance(item, dict):
                continue
            row_index = item.get("rowIndex")
            if row_index is None:
                continue
            normalized[str(row_index)] = {
                "indentLevel": item.get("indentLevel"),
                "inferredParent": item.get("inferredParent"),
            }
        return normalized
    normalized: dict[str, Any] = {}
    for row_index, value in result.items():
        if isinstance(value, dict):
            normalized[str(row_index)] = value
            continue
        if isinstance(value, int) or value is None:
            normalized[str(row_index)] = {
                "indentLevel": value,
                "inferredParent": None,
            }
    return normalized


# ---------------------------------------------------------------------------
# 父级推断
# ---------------------------------------------------------------------------

def _fill_inferred_parents(lines: list[dict[str, Any]]) -> None:
    """为每行填充 inferredParent 字段。

    从当前行向上查找第一个 indentLevel == 当前 indentLevel - 1 的有效行，
    将该父行的 rowIndex 写入 inferredParent。
    跳过 BLANK、HEADER、NOTE 行（不入栈也不设父级）。

    Args:
        lines: 行级字段字典列表，原地修改。
    """
    for current_index, line in enumerate(lines):
        if line["lineType"] in ("BLANK", "HEADER", "NOTE"):
            continue
        level = line["indentLevel"]
        target_parent_level = level - 1
        line["inferredParent"] = None
        for parent in reversed(lines[:current_index]):
            if parent["lineType"] in ("BLANK", "HEADER", "NOTE"):
                continue
            if parent.get("indentLevel") == target_parent_level:
                line["inferredParent"] = parent.get("rowIndex")
                break
