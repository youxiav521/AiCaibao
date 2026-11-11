import re
import json
import time
import asyncio
from typing import Any, Dict, List, Tuple, Optional
from loguru import logger

from src.applications.parsing.utils.table_classification.common import (
    ALLOWED_TABLE_TYPE_LABELS,
    ALLOWED_TABLE_TYPE_LABELS_SET,
    LARGE_TABLE_SET_THRESHOLD,
    LOCALE_EN_US,
    LOCALE_ZH_CN,
    MIN_COL_COUNT_FOR_STATEMENT,
    MIN_ROW_COUNT_FOR_STATEMENT,
    STATEMENT_TABLE_TYPE_LABELS,
    STATEMENT_TABLE_TYPE_LABELS_SET,
    compact_text,
    contains_any,
    count_contains,
    coerce_allowed_label,
    has_statement_content_keywords,
    normalize_locale,
    normalize_profit_statement_label,
)
from src.applications.parsing.utils.table_classification.en_rules import (
    apply_en_scope,
    detect_statement_type_by_content_en,
    has_primary_statement_title_phrase,
    is_derivative_statement_title_en,
    is_extended_non_financial_english_table,
    is_full_en_statement_table_md,
    is_non_financial_title_en,
    is_non_statement_exact_title_en,
    normalize_en_rule_text,
    strip_ifrs_parens,
)
from src.applications.parsing.utils.table_classification.prompts import (
    build_classification_prompt as build_language_aware_classification_prompt,
)
from src.applications.parsing.service.three_statements.statement_labels.en import (
    normalize_pl_oci_combined_label,
)
from src.applications.parsing.service.three_statements.continuation import (
    repair_statement_continuation_labels,
)
from src.applications.parsing.utils.table_classification.zh_rules import (
    detect_statement_type_by_content_zh,
    is_non_statement_exact_title,
    is_derivative_statement_title as zh_is_derivative_statement_title,
    is_full_zh_statement_table_md,
    is_non_financial_title,
)
from src.utils.chinese_converter import traditional_to_simplified
from src.utils.llm_invoke import ainvoke_llm_with_prompt

# ---------------------------------------------------------------------------
# English content-detection keywords (all lowercase for case-insensitive match)
# ---------------------------------------------------------------------------
_EN_BALANCE_TERMS: Tuple[str, ...] = (
    "total assets", "total liabilities", "total equity",
    "current assets", "non-current assets",
    "current liabilities", "non-current liabilities",
    "shareholders' equity", "stockholders' equity",
    "total liabilities and equity", "total equity and liabilities",
    "owners' equity",
)

_EN_PROFIT_TERMS: Tuple[str, ...] = (
    # "revenue" alone excluded: too broad, appears in balance sheets as "Deferred revenue"
    "total revenue", "operating revenue",
    "net income", "net profit", "profit for the year",
    "profit for the period", "profit for period",
    "gross profit", "operating income", "operating profit",
    "total comprehensive income", "earnings per share",
    "profit before tax", "profit before income tax",
    "income tax expense",
    "net sales", "turnover", "sales",
    "net loss", "loss for the year",
    "loss for the period", "profit from operations",
)

_EN_CASH_TERMS: Tuple[str, ...] = (
    "operating activities",
    "investing activities",
    "financing activities",
    "cash flows from operating",
    "cash flows from investing",
    "cash flows from financing",
    "cash flow from operating",
    "cash flow from investing",
    "cash flow from financing",
    "net cash from operating",
    "net cash used in",
    "cash and cash equivalents",
    "cash at end of",
    "cash at beginning of",
    "net increase in cash",
    "net change in cash",
)

_IFRS_PARENS_RE = re.compile(r'\s*\((?:loss(?:es)?|gain(?:s)?)\)', re.IGNORECASE)


def _strip_ifrs_parens(text: str) -> str:
    return strip_ifrs_parens(text)


def _normalize_level(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _get_heading_level(entry: Dict[str, Any]) -> Optional[int]:
    lvl = _normalize_level(entry.get("headingLevel"))
    if lvl is not None and lvl != 0:
        return lvl
    return _normalize_level(entry.get("textLevel"))


def _to_int(v: Any) -> Optional[int]:
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    return None


def _get_page_number(entry: Any) -> Optional[int]:
    if not isinstance(entry, dict):
        return None
    return _to_int(entry.get("page")) or _to_int(entry.get("pageAt"))


def _get_end_page(element: Dict[str, Any], start_page: int) -> int:
    """从 rows 中的 source_page_no 取表格实际结束页，取不到则返回 start_page。"""
    rows = element.get("rows")
    if not isinstance(rows, list):
        return start_page
    max_page = start_page
    for row in rows:
        if not isinstance(row, list):
            continue
        for cell in row:
            if isinstance(cell, dict):
                p = cell.get("source_page_no")
                if isinstance(p, int) and p > max_page:
                    max_page = p
    return max_page


def is_derivative_statement_title(text: Any) -> bool:
    return zh_is_derivative_statement_title(text)


def extract_statement_table_type(text: Any) -> Optional[str]:
    if not isinstance(text, str):
        return None
    stripped = traditional_to_simplified(text.strip())
    if not stripped:
        return None

    if is_derivative_statement_title(stripped):
        return None

    base: Optional[str] = None
    if "资产负债表" in stripped:
        base = "资产负债表"
    elif "利润表" in stripped or ("利润" in stripped and "分配表" in stripped):
        base = "利润表"
    elif "现金流量表" in stripped:
        base = "现金流量表"

    if base is None:
        # 尝试英文标题匹配
        lower = stripped.lower()
        if any(t in lower for t in ("balance sheet", "statement of financial position", "statements of financial position")):
            base = "资产负债表"
        elif any(t in lower for t in (
            "income statement", "profit and loss", "profit or loss",
            "statement of profit or loss", "statement of comprehensive income",
            "statement of operations",
        )):
            base = "利润表"
        elif any(t in lower for t in (
            "cash flow statement", "statement of cash flows",
            "cashflow statement",
        )):
            base = "现金流量表"

    if base is None:
        return None

    if "合并" in stripped:
        label = f"合并{base}"
    elif "母公司" in stripped or ("母" in stripped and "公司" in stripped):
        label = f"母公司{base}"
    else:
        # 英文口径判断
        lower = stripped.lower()
        if "consolidated" in lower:
            label = f"合并{base}"
        elif any(t in lower for t in ("standalone", "parent company", "parent only", "unconsolidated")):
            label = f"母公司{base}"
        else:
            label = base

    if label in STATEMENT_TABLE_TYPE_LABELS_SET:
        return normalize_profit_statement_label(label)
    return None


_NON_FINANCIAL_TITLE_TERMS: Tuple[str, ...] = (
    "构成", "介绍", "概述", "概况",
    "客户", "供应商",
    "排名", "名次", "竞争",
    "核心技术", "研发投入",
    "坏账准备", "长期股权投资", "递延所得税", "应付债券",
    "合同资产", "合同负债", "存货", "预付款项", "其他应收款",
    "商誉", "无形资产",
    "应付职工薪酬", "短期薪酬", "离职后福利",
    "联营", "合营", "子公司投资",
    "非经常性损益", "主营业务", "公允价值",
    "独立董事", "董事会", "监事会", "审计委员", "薪酬与提名",
    "担保", "会计师事务所",
    "持股", "限售股份", "总股本",
    "关键审计事项",
    "释义",
    "股东权益变动", "所有者权益变动", "权益变动",
)

# 三大表内容必备关键词（中文 + 英文），用于 md 内容快速排除
_STATEMENT_CONTENT_KEYWORDS_ZH: Tuple[str, ...] = (
    "资产", "负债", "权益", "利润", "净利", "收入",
    "现金流", "经营活动", "投资活动", "筹资活动",
)
_STATEMENT_CONTENT_KEYWORDS_EN: Tuple[str, ...] = (
    "assets", "liabilities", "equity", "profit", "revenue",
    "cash flow", "operating", "investing", "financing",
)

_EN_OTHER_DIRECT_PHRASES: Tuple[str, ...] = (
    "statement of changes in equity",
    "changes in equity",
    "changes in shareholders equity",
    "changes in owners equity",
    "movements during the financial year",
    "risk area and rationale",
    "our response",
    "maturity analysis",
    "ageing analysis",
    "fair value of financial instruments",
    "financial instruments carried at fair value",
    "key management personnel",
    "related party",
    "directors fees and remuneration",
    "share based payment",
    "contingent liabilities",
    "capital commitments",
    "operating segments",
    "business segments",
    "geographical segments",
    "rating agency",
    "rating classification",
    "principal activities",
    "name of company",
)

_EN_EQUITY_CHANGE_OTHER_TERMS: Tuple[str, ...] = (
    "attributable to owners",
    "owners of the company",
    "attributable to equity holders",
    "share capital",
    "retained earnings",
    "non controlling interests",
    "total equity",
    "reserve",
    "reserves",
    "translation reserve",
    "treasury shares",
    "dividends",
    "balance at",
    "comprehensive income",
)

_EN_EQUITY_CHANGE_ANCHORS: Tuple[str, ...] = (
    "attributable to owners",
    "attributable to equity holders",
    "owners of the company",
    "balance at",
    "dividends",
    "translation reserve",
    "treasury shares",
)

_EN_FINANCIAL_NOTE_OTHER_TERMS: Tuple[str, ...] = (
    "financial instruments",
    "risk management",
    "fair value",
    "credit risk",
    "liquidity risk",
    "sensitivity analysis",
    "expected credit losses",
    "stage 1",
    "stage 2",
    "stage 3",
    "12 month ecl",
    "lifetime ecl",
)

_EN_DETAIL_NOTE_OTHER_TERMS: Tuple[str, ...] = (
    "property plant and equipment",
    "right of use assets",
    "investment properties",
    "inventories",
    "trade and other receivables",
    "trade and other payables",
    "borrowings",
    "lease liabilities",
    "deferred tax",
    "income tax",
    "commitments",
    "contingencies",
    "remuneration",
    "allowance",
    "impairment",
    "carrying amount",
    "reconciliation",
    "movement",
)

_EN_DETAIL_NOTE_ANCHORS: Tuple[str, ...] = (
    "carrying amount",
    "reconciliation",
    "movement",
    "allowance",
    "impairment",
    "fair value",
    "as at",
)

_EN_PRIMARY_STATEMENT_TITLE_PHRASES: Tuple[str, ...] = (
    "statement of financial position",
    "balance sheet",
    "income statement",
    "statement of profit or loss",
    # "statement of comprehensive income" 不算主三大表，独立一类
    "statement of operations",
    "statement of cash flows",
    "cash flow statement",
)

# 大数据集规则启用阈值：表格总数 >= 该值时启用扩展标题/内容关键词排除
_LARGE_TABLE_SET_THRESHOLD: int = 40

# 行列数快速排除阈值：低于该值直接判"其他表"
_MIN_ROW_COUNT_FOR_STATEMENT: int = 10
_MIN_COL_COUNT_FOR_STATEMENT: int = 2


def _is_non_financial_title(label: str) -> bool:
    return is_non_financial_title(label)



def _normalize_en_rule_text(text: Any) -> str:
    return normalize_en_rule_text(text)


def _has_primary_statement_title_phrase(text: str) -> bool:
    normalized = _normalize_en_rule_text(text)
    if not normalized:
        return False
    return any(phrase in normalized for phrase in _EN_PRIMARY_STATEMENT_TITLE_PHRASES)


def _is_extended_non_financial_english_table(label: str, table_md: str) -> bool:
    return is_extended_non_financial_english_table(label, table_md)


def _has_statement_content_keywords(table_md: str, locale: Optional[str] = None) -> bool:
    return has_statement_content_keywords(table_md, locale=locale)


def _compact_text(text: str) -> str:
    return compact_text(text)


def _contains_any(text: str, terms: Tuple[str, ...]) -> bool:
    return contains_any(text, terms)


def _count_contains(text: str, terms: Tuple[str, ...]) -> int:
    return count_contains(text, terms)


def _is_full_en_statement_table_md(table_md: str, label: str) -> bool:
    return is_full_en_statement_table_md(table_md, label)


def is_full_statement_table_md(table_md: str, label: str, locale: Optional[str] = None) -> bool:
    if not isinstance(label, str) or not label.strip() or label == "其他表":
        return False
    if not _compact_text(traditional_to_simplified(table_md or "")):
        return False
    if isinstance(locale, str) and locale.strip():
        _locale = normalize_locale(locale)
        if _locale == LOCALE_EN_US:
            return is_full_en_statement_table_md(table_md, label)
        return is_full_zh_statement_table_md(table_md, label)
    # locale 缺省时兼容旧行为：中英文都尝试
    return is_full_zh_statement_table_md(table_md, label) or is_full_en_statement_table_md(table_md, label)


_EN_SCOPE_CONSOLIDATED_KW: Tuple[str, ...] = ("consolidated",)
_EN_SCOPE_STANDALONE_KW: Tuple[str, ...] = (
    "standalone", "parent company", "parent only", "unconsolidated", "parent-only",
)


def _apply_en_scope(base: str, text_lower: str) -> str:
    return apply_en_scope(base, text_lower)


def _detect_statement_type_by_content_en(table_md: str) -> Optional[str]:
    return detect_statement_type_by_content_en(table_md)


def detect_statement_type_by_content(table_md: str, locale: Optional[str] = None) -> Optional[str]:
    compact = _compact_text(traditional_to_simplified(table_md or ""))
    if not compact:
        return None

    explicit_locale = isinstance(locale, str) and bool(locale.strip())
    _locale = normalize_locale(locale) if explicit_locale else None

    # 完整三大表判定（按 locale 分流）
    for label in ("资产负债表", "利润表", "现金流量表"):
        if is_full_statement_table_md(table_md, label, locale=locale):
            text_lower = re.sub(r"\s+", " ", (table_md or "").lower())
            return _apply_en_scope(label, text_lower)

    if explicit_locale and _locale == LOCALE_EN_US:
        return _detect_statement_type_by_content_en(table_md or "")
    if explicit_locale and _locale == LOCALE_ZH_CN:
        return detect_statement_type_by_content_zh(table_md)

    # locale 缺省 → 先中后英（兼容旧行为）
    content_label = detect_statement_type_by_content_zh(table_md)
    if content_label:
        return content_label
    return _detect_statement_type_by_content_en(table_md or "")


def find_table_label(elements: List[Dict[str, Any]], table_index: int, locale: Optional[str] = None) -> str:
    table_element = elements[table_index]
    if isinstance(table_element, dict) and table_element.get("type") == "excel_table":
        sheet_name = (table_element.get("sheetName") or "").strip()
        if sheet_name:
            return sheet_name

    explicit_locale = isinstance(locale, str) and bool(locale.strip())
    _locale = normalize_locale(locale) if explicit_locale else None
    is_en = explicit_locale and _locale == LOCALE_EN_US

    raw_caption_value = None
    if isinstance(table_element, dict):
        raw_caption_value = table_element.get("table_caption") or table_element.get("tableCaption")

    caption_lines: List[str] = []
    if isinstance(raw_caption_value, list):
        caption_lines = [str(item).strip() for item in raw_caption_value if isinstance(item, str) and item.strip()]
    elif isinstance(raw_caption_value, str) and raw_caption_value.strip():
        caption_lines = [raw_caption_value.strip()]

    caption_label = "\n".join(caption_lines).strip() if caption_lines else ""
    if caption_label:
        if is_en:
            return caption_label
        if "表" in caption_label:
            return caption_label

    table_page = _get_page_number(table_element)
    for idx in range(table_index - 1, -1, -1):
        element = elements[idx]
        if element.get("type") in ("table", "excel_table"):
            break
        if element.get("type") != "text":
            continue
        element_page = _get_page_number(element)
        if table_page is not None and element_page is not None and table_page - element_page > 1:
            break
        label = element.get("md") or element.get("value") or ""
        if not isinstance(label, str) or not label.strip():
            continue
        if is_en:
            if has_primary_statement_title_phrase(label):
                return label.strip()
        elif "表" in label:
            return label.strip()

    # 向上无 text 时，从表格 md 前 75 字符提取标题
    md_head = (table_element.get("md") or "")[:75]
    if md_head.strip():
        if is_en:
            if has_primary_statement_title_phrase(md_head):
                return md_head.strip()
        elif "表" in md_head:
            return md_head.strip()

    return ""


def _element_to_text(element: Dict[str, Any]) -> str:
    """从元素中提取文本内容（md 或 value）"""
    if not isinstance(element, dict):
        return ""
    text = element.get("md") or element.get("value") or ""
    if not isinstance(text, str) or not text.strip():
        return ""
    return text.strip()


def _extract_table_context(elements: List[Dict[str, Any]], table_index: int) -> str:
    """提取表格上方4个文本元素作为上下文，跨页超过1页则截断"""
    table_element = elements[table_index]
    table_page = _get_page_number(table_element)

    # caption 是表格元素的属性，不在上方文本元素中，需单独提取
    caption_lines = table_element.get("table_caption") or []
    if isinstance(caption_lines, str):
        caption_lines = [caption_lines]
    caption_text = "\n".join(str(line).strip() for line in caption_lines if str(line).strip())

    above_texts: List[str] = []
    for idx in range(table_index - 1, -1, -1):
        element = elements[idx]
        if element.get("type") in ("table", "excel_table"):
            break
        if element.get("type") != "text":
            continue
        element_page = _get_page_number(element)
        if table_page is not None and element_page is not None and table_page - element_page > 1:
            break
        text = _element_to_text(element)
        if text:
            above_texts.append(text)
        if len(above_texts) >= 4:
            break
    above_texts.reverse()

    if caption_text:
        above_joined = "\n".join(above_texts)
        return f"{caption_text}\n{above_joined}".strip()
    return "\n".join(above_texts).strip()


def build_classification_prompt(table_md: str, table_label: str, table_context: str = "", locale: Optional[str] = None) -> List[Dict[str, str]]:
    return build_language_aware_classification_prompt(table_md, table_label, table_context, locale=locale)


def _coerce_allowed_label(label: Any) -> str:
    return coerce_allowed_label(label)


def parse_model_output_to_label(text: Any) -> str:
    if not isinstance(text, str):
        return "其他表"
    # 移除 LLM 思考模式输出的 <think>...</think> 标签
    candidate = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    try:
        m = re.search(r"\{[\s\S]*?\}", candidate)
        if m:
            obj = json.loads(m.group(0))
            value = obj.get("table_type") or obj.get("label") or obj.get("type")
            if isinstance(value, str) and value.strip():
                return _coerce_allowed_label(value)
    except Exception:
        pass
    if not candidate:
        return "其他表"
    lines = candidate.splitlines()
    first_line = lines[0].strip() if lines else ""
    first_line = first_line.strip('\"\'\u3000 ').strip()
    return _coerce_allowed_label(first_line or "其他表")


async def classify_table_type_async(
    table_md: str,
    table_label: str,
    table_context: str,
    semaphore: asyncio.Semaphore,
    model_id: str = "model_for_table_classify",
    locale: Optional[str] = None,
) -> str:
    try:
        prompt_messages = build_classification_prompt(
            table_md=table_md, table_label=table_label, table_context=table_context, locale=locale,
        )
        prompt_text = "\n\n".join(str(msg.get("content") or "") for msg in (prompt_messages or [])).strip()
        t0 = time.monotonic()
        async with semaphore:
            response_text = await ainvoke_llm_with_prompt(prompt_text, model_id=model_id, temperature=0.01)
        elapsed = time.monotonic() - t0
        result = parse_model_output_to_label(str(response_text or ""))
        return result
    except Exception as e:
        logger.warning("表格分类(LLM)失败: label={!r}, error={}", table_label, e)
        return "其他表"


def _extract_scope_prefix(text: str) -> Optional[str]:
    """Extract scope prefix ('合并' or '母公司') from a label/heading string."""
    if not isinstance(text, str):
        return None
    if "合并" in text:
        return "合并"
    if "母公司" in text or ("母" in text and "公司" in text):
        return "母公司"
    return None


def _table_caption_blob(element: Dict[str, Any]) -> str:
    """聚合表格的 caption / sheetName 文本，形式与 three_statements 组装阶段一致。"""
    cap = element.get("table_caption") or element.get("tableCaption")
    lines: List[str] = []
    sheet_name = element.get("sheetName") or element.get("sheet_name")
    if isinstance(sheet_name, str) and sheet_name.strip():
        lines.append(sheet_name.strip())
    if isinstance(cap, list):
        for x in cap:
            if isinstance(x, str) and x.strip():
                lines.append(x.strip())
            elif isinstance(x, dict):
                text = str(x.get("content") or x.get("text") or "").strip()
                if text:
                    lines.append(text)
    elif isinstance(cap, str) and cap.strip():
        lines.append(cap.strip())
    return "\n".join(lines)


def _apply_oci_caption_rescue(items: List[Dict[str, Any]]) -> None:
    """LLM 分类后、聚簇前的 P&L+OCI 合并表救回。

    某些英文年报里"Statement of profit or loss and other comprehensive income"
    会被 LLM 判为"其他表"或"利润表"，导致下游聚簇看不到它，或丢失"利润表+OCI"
    独立类别。此处用 caption 明文规则补一刀，把这种合并表统一改写成
    ``利润表及其他综合收益`` 家族标签。
    """
    rescued = 0
    for element in items:
        if not isinstance(element, dict):
            continue
        if element.get("type") not in {"table", "excel_table"}:
            continue
        caption_blob = _table_caption_blob(element)
        md_head = str(element.get("md") or "")[:4000]
        blob = "\n".join(x for x in (caption_blob, md_head) if x)
        if not blob:
            continue
        oci_label = normalize_pl_oci_combined_label(blob)
        if not oci_label:
            continue
        current = element.get("table_type") or ""
        if current == oci_label:
            continue
        element["table_type"] = oci_label
        element["table_type_is_full_statement"] = True
        element.pop("table_type_evidence", None)
        rescued += 1
    if rescued:
        logger.info("OCI caption 救回: 共 {} 张表改写为 P&L+OCI 合并表", rescued)


def _reclassify_statement_outliers_by_page_cluster(
    items: List[Dict[str, Any]],
    *,
    coverage_threshold: float = 0.5,
    page_span_ratio: float = 0.2,
) -> int:
    # 改为基于 table_type 是否落入三大表家族集合判定（已包含中文家族标签；
    # 英文文档经 en_rules.apply_en_scope 也会被映射成"合并资产负债表"等中文标签，
    # 故同时覆盖中英文文档）。同时尊重 table_type_is_full_statement 标志，
    # 仅对"看起来是完整三大表"的候选参与页码主簇分析。
    statement_candidates: List[Tuple[int, int, int, Dict[str, Any]]] = []
    all_pages: List[int] = []
    for idx, element in enumerate(items):
        if not isinstance(element, dict):
            continue
        page = _get_page_number(element)
        if page is not None:
            all_pages.append(page)
        if element.get("type") not in {"table", "excel_table"}:
            continue
        raw_label = element.get("table_type") or ""
        is_statement_label = raw_label in STATEMENT_TABLE_TYPE_LABELS_SET
        is_full_statement = bool(element.get("table_type_is_full_statement"))
        if not (is_statement_label or is_full_statement):
            continue
        if page is None:
            continue
        end_page = _get_end_page(element, page)
        statement_candidates.append((page, end_page, idx, element))

    total = len(statement_candidates)
    if total == 0:
        return 0

    total_pages = (max(all_pages) - min(all_pages) + 1) if all_pages else 1

    statement_candidates.sort(key=lambda item: (item[0], item[2]))
    clusters: List[List[Tuple[int, int, int, Dict[str, Any]]]] = []
    current: List[Tuple[int, int, int, Dict[str, Any]]] = []
    prev_end: Optional[int] = None
    for candidate in statement_candidates:
        start, end = candidate[0], candidate[1]
        if current and prev_end is not None and start - prev_end > 3:
            clusters.append(current)
            current = []
        current.append(candidate)
        prev_end = max(prev_end or end, end)
    if current:
        clusters.append(current)

    # 多簇时取最靠前的（页码最小，即财务报表正文位置），而非最大的，
    # 避免主表+附注几乎对半分时被附注簇压过；同时附注通常出现在文档后段，
    # 取最靠前簇能更稳定地锚定真正的三大表区间。
    main_cluster = clusters[0]
    if len(main_cluster) / total < coverage_threshold:
        return 0

    cluster_start = main_cluster[0][0]
    cluster_end = max(c[1] for c in main_cluster)
    cluster_span = cluster_end - cluster_start + 1
    if cluster_span / total_pages > page_span_ratio:
        return 0

    changed = 0
    for start, end, _, element in statement_candidates:
        if cluster_start <= start <= cluster_end:
            continue
        element["table_type"] = "其他表"
        element["table_type_is_full_statement"] = False
        element.pop("table_type_evidence", None)
        changed += 1

    if changed:
        logger.info(
            "三大表页码主区间后处理: 候选数={}, 主区间={}~{}(跨度{}/总{}页={:.0%}), 覆盖率={:.0%}, 区间外改为其他表={}",
            total, cluster_start, cluster_end, cluster_span, total_pages,
            cluster_span / total_pages, len(main_cluster) / total, changed,
        )
    return changed


async def classify_tables_async(
    items: List[Dict[str, Any]],
    force_reclassify: bool = False,
    max_concurrency: int = 10,
    locale: Optional[str] = None,
    skip_page_cluster: bool = False,
) -> Tuple[int, int]:
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks: List[Any] = []
    tables_to_process: List[Tuple[Dict[str, Any], str]] = []
    total_tables = 0
    classified_count = 0

    table_indices: List[int] = [
        idx for idx, el in enumerate(items)
        if isinstance(el, dict) and el.get("type") in {"table", "excel_table"}
    ]
    total_table_count = len(table_indices)
    use_extended_rules = total_table_count >= _LARGE_TABLE_SET_THRESHOLD

    # 规则命中计数（用于汇总日志）
    hit_size = 0
    hit_title_exclude = 0
    hit_no_content_kw = 0
    hit_exact_title = 0

    # 预计算 locale 归一化值，避免循环内重复计算
    explicit_locale = isinstance(locale, str) and bool(locale.strip())
    _loc = normalize_locale(locale) if explicit_locale else None
    is_en = explicit_locale and _loc == LOCALE_EN_US


    for element_index, element in enumerate(items):
        if element.get("type") not in {"table", "excel_table"}:
            continue

        total_tables += 1
        if "table_type" in element and not force_reclassify:
            classified_count += 1
            continue

        table_md = element.get("md") or ""
        label_text = find_table_label(items, element_index, locale=locale)

        # 规则1：行列数阈值排除（始终启用）
        row_count = count_rows_from_table_element(element)
        col_count = count_cols_from_table_element(element)
        if row_count < _MIN_ROW_COUNT_FOR_STATEMENT or col_count < _MIN_COL_COUNT_FOR_STATEMENT:
            element["table_type"] = "其他表"
            element["table_type_is_full_statement"] = False
            element.pop("table_type_evidence", None)
            classified_count += 1
            hit_size += 1
            continue

        # 规则2：标题精确匹配三大表（中文标签） → 直接分类
        # 中文报表或未指定 locale 时启用；en_us 跳过（标签集合本身是中文，但用户上传的英文报表标题不会命中，跑也无副作用）
        label_normalized = traditional_to_simplified(str(label_text).strip()) if label_text else ""
        if not is_en and label_normalized in STATEMENT_TABLE_TYPE_LABELS_SET:
            element["table_type"] = normalize_profit_statement_label(label_normalized)
            element["table_type_is_full_statement"] = bool(
                is_full_statement_table_md(table_md, label_normalized, locale=locale)
            )
            element.pop("table_type_evidence", None)
            classified_count += 1
            hit_exact_title += 1
            continue

        # 规则3：非三大表精确标题排除（按 locale 分流）
        if label_text:
            if is_en:
                if (
                    is_non_statement_exact_title_en(label_text)
                    or is_derivative_statement_title_en(label_text)
                ):
                    element["table_type"] = "其他表"
                    element["table_type_is_full_statement"] = False
                    element.pop("table_type_evidence", None)
                    classified_count += 1
                    hit_title_exclude += 1
                    continue
            else:
                if is_non_statement_exact_title(label_text):
                    element["table_type"] = "其他表"
                    element["table_type_is_full_statement"] = False
                    element.pop("table_type_evidence", None)
                    classified_count += 1
                    hit_title_exclude += 1
                    continue

        # 规则4：标题排除 + 内容关键词排除（大数据集场景启用，按 locale 分流）
        if use_extended_rules:
            title_excluded = (
                is_non_financial_title_en(label_text)
                if is_en
                else _is_non_financial_title(label_text)
            )
            extended_excluded = (
                _is_extended_non_financial_english_table(label_text, table_md) if is_en else False
            )
            if title_excluded or extended_excluded:
                element["table_type"] = "其他表"
                element["table_type_is_full_statement"] = False
                element.pop("table_type_evidence", None)
                classified_count += 1
                hit_title_exclude += 1
                continue
            if not _has_statement_content_keywords(table_md, locale=locale):
                element["table_type"] = "其他表"
                element["table_type_is_full_statement"] = False
                element.pop("table_type_evidence", None)
                classified_count += 1
                hit_no_content_kw += 1
                continue


        table_context = _extract_table_context(items, element_index)
        task = classify_table_type_async(
            table_md, label_text, table_context, semaphore,
            model_id="model_for_table_classify", locale=locale,
        )
        tasks.append(task)
        tables_to_process.append((element, table_md))

    if not tasks:
        _apply_oci_caption_rescue(items)
        if use_extended_rules and not skip_page_cluster:
            _reclassify_statement_outliers_by_page_cluster(items)
        continuation_adjusted = repair_statement_continuation_labels(
            items,
            normalize_statement_label=extract_statement_table_type,
        )
        logger.info(
            "表格分类汇总: locale={}, 共{}张表, 规则命中={}, LLM调用=0 | 精确标题={}, 标题排除={}, 行列阈值={}, 内容关键词={}, 续表纠正={}",
            _loc or "auto", total_tables, classified_count, hit_exact_title, hit_title_exclude, hit_size, hit_no_content_kw,
            continuation_adjusted,
        )
        return total_tables, classified_count

    t_llm_start = time.monotonic()
    results = await asyncio.gather(*tasks)
    t_llm_elapsed = time.monotonic() - t_llm_start
    for (element, table_md), result in zip(tables_to_process, results):
        table_type = parse_model_output_to_label(result)
        is_full = is_full_statement_table_md(table_md, table_type, locale=locale)
        element["table_type"] = table_type
        element["table_type_is_full_statement"] = bool(is_full)
        element.pop("table_type_evidence", None)
        classified_count += 1

    _apply_oci_caption_rescue(items)
    if use_extended_rules and not skip_page_cluster:
        _reclassify_statement_outliers_by_page_cluster(items)
    continuation_adjusted = repair_statement_continuation_labels(
        items,
        normalize_statement_label=extract_statement_table_type,
    )
    logger.info(
        "表格分类汇总: locale={}, 共{}张表, 规则命中={}, LLM调用={}, LLM总耗时={:.2f}s | 精确标题={}, 标题排除={}, 行列阈值={}, 内容关键词={}, 续表纠正={}",
        _loc or "auto", total_tables, classified_count - len(tasks), len(tasks), t_llm_elapsed,
        hit_exact_title, hit_title_exclude, hit_size, hit_no_content_kw,
        continuation_adjusted,
    )
    return total_tables, classified_count



SIX_STATEMENT_LABELS: Tuple[str, ...] = (
    "合并资产负债表",
    "母公司资产负债表",
    "合并利润表",
    "母公司利润表",
    "合并现金流量表",
    "母公司现金流量表",
)


def normalize_table_type(label: Optional[str]) -> Optional[str]:
    """
    将模型/规则判定的表格标签标准化到六种三大表之一；否则返回 None。
    - 兼容“金流量表”误写为“现金流量表”。
    - 根据是否含“合并/母公司”与“资产负债表/利润表/现金流量表”进行归一。
    """
    if not isinstance(label, str):
        return None
    text = label.strip()
    if not text:
        return None
    text = text.replace("金流量表", "现金流量表")

    is_merge = "合并" in text
    is_parent = ("母公司" in text) or ("母" in text and "公司" in text)

    if "资产负债表" in text:
        if is_parent:
            return "母公司资产负债表"
        if is_merge:
            return "合并资产负债表"
    if "利润表" in text or ("利润" in text and "分配表" in text):
        if is_parent:
            return "母公司利润表"
        if is_merge:
            return "合并利润表"
    if "现金流量表" in text:
        if is_parent:
            return "母公司现金流量表"
        if is_merge:
            return "合并现金流量表"

    if text in SIX_STATEMENT_LABELS:
        return text
    return None


def count_rows_from_table_element(table_el: Dict[str, Any]) -> int:
    """
    估算表格的行数：优先用 rows 字段，其次用 md 行数(减去表头)。
    用于在同一类别内挑选“更完整/行数更多”的那一张。
    """
    rows = table_el.get("rows")
    if isinstance(rows, list):
        try:
            return len(rows)
        except Exception:
            return 0
    md = table_el.get("md")
    if isinstance(md, str) and md.strip():
        lines = [ln for ln in md.splitlines() if ln.strip()]
        return max(0, len(lines) - 1)
    return 0


def count_cols_from_table_element(table_el: Dict[str, Any]) -> int:
    """估算表格的列数：优先用 rows 的多行最大有效列数，其次用 md 的第一行管道数。"""
    rows = table_el.get("rows")
    if isinstance(rows, list) and rows:
        max_cols = 0
        for row in rows[:10]:
            if not isinstance(row, list) or not row:
                continue
            row_cols = 0
            for cell in row:
                if not isinstance(cell, dict):
                    row_cols += 1
                    continue
                colspan = cell.get("colspan", 1)
                colstart = cell.get("colstart")
                try:
                    span = max(1, int(colspan))
                except (TypeError, ValueError):
                    span = 1
                if isinstance(colstart, int):
                    row_cols = max(row_cols, colstart + span)
                else:
                    row_cols += span
            max_cols = max(max_cols, row_cols)
        if max_cols > 0:
            return max_cols
    md = table_el.get("md")
    if isinstance(md, str) and md.strip():
        for line in md.splitlines():
            if "|" in line:
                return max(0, line.count("|") - 1)
    return 0


def select_best_statement_tables(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    在六种三大表中，每类最多保留一张：按行数从大到小挑选最优。
    返回: {标准化标签: 表格元素}
    """
    groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for el in items:
        if not isinstance(el, dict):
            continue
        if el.get("type") not in {"table", "excel_table"}:
            continue
        label = normalize_table_type(el.get("table_type"))
        if not label:
            continue
        if label not in SIX_STATEMENT_LABELS:
            continue
        rc = count_rows_from_table_element(el)
        groups.setdefault(label, []).append((rc, el))

    best: Dict[str, Dict[str, Any]] = {}
    for label, lst in groups.items():
        if not lst:
            continue
        lst_sorted = sorted(lst, key=lambda t: t[0], reverse=True)
        best[label] = lst_sorted[0][1]
    return best


def mark_best_statement_tables_inplace(items: List[Dict[str, Any]]) -> None:
    """
    在原始 items 上打标：为被选中的六种三大表置 is_best_statement=True，其余为 False/不设置。
    不改变原始列表结构，便于后续按需过滤或导出。
    """
    best = select_best_statement_tables(items)
    best_set = {id(v) for v in best.values()}
    for el in items:
        if not isinstance(el, dict) or el.get("type") not in {"table", "excel_table"}:
            continue
        el["is_best_statement"] = (id(el) in best_set)
