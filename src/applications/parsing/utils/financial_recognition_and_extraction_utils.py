"""财报识别与财报事实抽取的共享工具。

`detect_gaap_type`：先显式关键词（IFRS/US GAAP/SEC/ESEF/CAS 监管用语），
无命中时再语义推断（中文列报→CAS，国际英文列报→IFRS，含 MFRS 关键词归 IFRS）。
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

StatementTypeCode = Literal["BS", "IS", "CF"]
GaapTypeCode = Literal["IFRS", "US_GAAP", "ESEF", "SEC", "CAS"]
CaliberLabel = Literal["合并", "单体", "混合", "未知"]

_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_YEAR_ZH_RE = re.compile(r"(19\d{2}|20\d{2})\s*年")

_GAAP_KEYWORDS: list[tuple[GaapTypeCode, tuple[str, ...]]] = [
    ("ESEF", ("esef", "european single electronic format", "eu taxonomy regulation")),
    ("US_GAAP", ("u.s. gaap", "us gaap", "fasb asc", "fasb", "accounting standards codification")),
    (
        "IFRS",
        (
            "ifrs",
            "international financial reporting",
            "ias 1",
            "ias1",
            "ias ",
            "ipsas",
            "mfrs",
            "malaysian financial reporting standards",
        ),
    ),
    ("SEC", ("form 10-k", "form 10-q", "sec filing", "regulation s-k", "item 8", "edgar")),
]

_CAS_KEYWORDS: tuple[str, ...] = (
    "企业会计准则",
    "中国会计准则",
    "财政部",
    "证监会",
    "公开发行证券的公司信息披露",
)

# 无语境「准则」声明时：从表体/科目语言与常见披露用语推断中国内地财报 → CAS
_CAS_SEMANTIC_TERMS: tuple[str, ...] = (
    "合并资产负债表",
    "合并利润表",
    "合并现金流量表",
    "资产负债表",
    "利润表",
    "现金流量表",
    "所有者权益",
    "股东权益合计",
    "股东权益",
    "流动资产",
    "非流动资产",
    "流动负债",
    "非流动负债",
    "货币资金",
    "应收账款",
    "存货",
    "营业收入",
    "营业成本",
    "净利润",
    "综合收益总额",
    "每股收益",
    "期末余额",
    "期初余额",
    "本期金额",
    "上期金额",
    "附注",
    "人民币",
    "千元",
    "万元",
    "亿元",
)

# 国际通用英文披露结构（含 MFRS 等实质与 IFRS 对齐的列报）→ 推断 IFRS
_INTL_IFRS_SEMANTIC_TERMS: tuple[str, ...] = (
    "equity and liabilities",
    "assets and liabilities",
    "statement of financial position",
    "statement of profit or loss",
    "statement of cash flows",
    "statement of comprehensive income",
    "share capital",
    "retained earnings",
    "non-controlling interests",
    "non-controlling interest",
    "trade payables",
    "trade receivables",
    "lease liabilities",
    "contract liabilities",
    "deferred tax liabilities",
    "deferred tax assets",
    "short-term borrowings",
    "long-term borrowings",
)


def _keyword_gaap_scores(combined: str, lower: str) -> dict[str, int]:
    scores: dict[str, int] = {}
    for code, keys in _GAAP_KEYWORDS:
        s = sum(1 for k in keys if k in lower)
        if s:
            scores[code] = scores.get(code, 0) + s
    for ck in _CAS_KEYWORDS:
        if ck in combined:
            scores["CAS"] = scores.get("CAS", 0) + 2
    return scores


def _heuristic_infer_gaap(combined: str, lower: str) -> Optional[GaapTypeCode]:
    """显式关键词未命中时的语义推断：中文列报 → CAS；国际英文列报 → IFRS。"""
    if len(combined.strip()) < 12:
        return None

    cas_hits = sum(1 for t in _CAS_SEMANTIC_TERMS if t in combined)
    intl_hits = sum(1 for t in _INTL_IFRS_SEMANTIC_TERMS if t in lower)

    cjk = len(re.findall(r"[\u4e00-\u9fff]", combined))
    letters = len(re.findall(r"[a-zA-Z]", combined))

    # 中文密度：典型 A 股/港股中文列报
    if cjk >= 12:
        cas_hits += 2
    elif cjk >= 6:
        cas_hits += 1

    # 英文主导且汉字很少：典型国际英文年报/马来等英文列报
    if letters >= 40 and cjk <= max(8, letters // 25):
        intl_hits += 2
    elif letters >= 80 and cjk < 15:
        intl_hits += 1

    if cas_hits == 0 and intl_hits == 0:
        return None

    if cas_hits > intl_hits:
        return "CAS"
    if intl_hits > cas_hits:
        return "IFRS"

    # 平局：按字符类型倾向
    if cas_hits > 0 and cjk >= letters:
        return "CAS"
    if intl_hits > 0 and letters > cjk:
        return "IFRS"
    return None


def detect_gaap_type(*text_parts: str) -> Optional[GaapTypeCode]:
    """识别准则类型：优先显式关键词；无命中时再用语义推断（中文→CAS，国际英文→IFRS）。"""
    combined = "\n".join(p for p in text_parts if p).strip()
    if not combined:
        return None
    lower = combined.lower()
    scores = _keyword_gaap_scores(combined, lower)
    if scores:
        best = max(scores.items(), key=lambda x: x[1])
        return best[0] if best[1] > 0 else None  # type: ignore[return-value]
    return _heuristic_infer_gaap(combined, lower)


def extract_years_from_text(text: str) -> list[int]:
    if not text:
        return []
    found: set[int] = set()
    for m in _YEAR_RE.finditer(text):
        y = int(m.group(1))
        if 1990 <= y <= 2050:
            found.add(y)
    for m in _YEAR_ZH_RE.finditer(text):
        y = int(m.group(1))
        if 1990 <= y <= 2050:
            found.add(y)
    return sorted(found, reverse=True)


def infer_accounting_period(
    *text_parts: str,
) -> dict[str, Any]:
    """从表格 md、标题、上下文中抽取年份列表与可读标签。"""
    blob = "\n".join(p for p in text_parts if p).strip()
    years = extract_years_from_text(blob)
    label: Optional[str] = None
    if years:
        if len(years) == 1:
            label = str(years[0])
        else:
            label = f"{years[-1]}-{years[0]}"
    return {"years": years, "period_label": label}


def _fix_blob_for_statement_map(blob: str) -> str:
    return blob.replace("金流量表", "现金流量表")


def map_to_statement_type(
    raw_table_type: str,
    normalized_zh_label: Optional[str],
) -> Optional[StatementTypeCode]:
    """将财报分类映射为 BS / IS / CF。"""
    blob = _fix_blob_for_statement_map(f"{raw_table_type or ''}\n{normalized_zh_label or ''}")
    if "资产负债表" in blob:
        return "BS"
    if "利润表" in blob or ("利润" in blob and "分配表" in blob):
        return "IS"
    if "现金流量表" in blob:
        return "CF"
    return None


def collect_neighbor_text_context(
    items: Sequence[Any],
    table_index: int,
    *,
    max_chars: int = 4000,
) -> tuple[str, str]:
    """取当前表在 items 序列中的上文、下文纯文本（用于准则/期间推断）。"""
    before_chunks: list[str] = []
    after_chunks: list[str] = []
    for i in range(table_index - 1, -1, -1):
        el = items[i]
        if not isinstance(el, dict):
            continue
        if el.get("type") in {"table", "excel_table"}:
            break
        t = _element_plain_text(el)
        if t:
            before_chunks.append(t)
        if sum(len(x) for x in before_chunks) >= max_chars:
            break
    before = "\n".join(reversed(before_chunks))[-max_chars:]

    for i in range(table_index + 1, len(items)):
        el = items[i]
        if not isinstance(el, dict):
            continue
        if el.get("type") in {"table", "excel_table"}:
            break
        t = _element_plain_text(el)
        if t:
            after_chunks.append(t)
        if sum(len(x) for x in after_chunks) >= max_chars:
            break
    after = "\n".join(after_chunks)[:max_chars]
    return before, after


def _element_plain_text(el: Any) -> str:
    if not isinstance(el, dict):
        return ""
    md = el.get("md")
    if isinstance(md, str) and md.strip():
        return md.strip()
    val = el.get("value")
    if isinstance(val, str) and val.strip():
        return val.strip()
    return ""


def table_element_markdown(table_el: Dict[str, Any]) -> str:
    md = table_el.get("md")
    if isinstance(md, str) and md.strip():
        return md.strip()
    return ""


def is_financial_statement_category(raw_table_type: str, statement_type: Optional[str]) -> bool:
    if not raw_table_type:
        return statement_type in {"BS", "IS", "CF"}
    if raw_table_type.strip() == "其他表":
        return False
    return statement_type in {"BS", "IS", "CF"}


def detect_caliber_label(*text_parts: str) -> CaliberLabel:
    blob = "\n".join(p for p in text_parts if p)
    has_merge = "合并" in blob or "consolidated" in blob.lower()
    has_parent = (
        "母公司" in blob
        or "单体" in blob
        or "parent only" in blob.lower()
        or "standalone" in blob.lower()
    )
    if has_merge and has_parent:
        return "混合"
    if has_merge:
        return "合并"
    if has_parent:
        return "单体"
    return "未知"


_VALUE_TEXT_RE = re.compile(r"^[\d,.\-—\s%()（）±+]+$")


def _cell_text(cell: Any) -> str:
    if cell is None:
        return ""
    if hasattr(cell, "text"):
        v = getattr(cell, "text", None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    if isinstance(cell, dict):
        for k in ("text", "value", "md"):
            v = cell.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    return str(cell).strip()


def _cell_bbox_page(cell: Any) -> tuple[dict[str, Any], Optional[int]]:
    bbox: Any = None
    page: Any = None
    if isinstance(cell, dict):
        bbox = cell.get("bBox") or cell.get("bbox")
        page = cell.get("pageAt") or cell.get("page") or cell.get("source_page_no")
    elif cell is not None:
        bbox = getattr(cell, "bBox", None)
        page = getattr(cell, "pageAt", None)
    if bbox is not None and not isinstance(bbox, dict):
        if hasattr(bbox, "model_dump"):
            bbox = bbox.model_dump(by_alias=True)
        elif hasattr(bbox, "__dict__"):
            bbox = {
                "l": getattr(bbox, "l", 0),
                "t": getattr(bbox, "t", 0),
                "r": getattr(bbox, "r", 0),
                "b": getattr(bbox, "b", 0),
                "coordOrigin": getattr(bbox, "coordOrigin", "TOPLEFT"),
            }
    if isinstance(bbox, dict):
        bb = {
            "l": float(bbox.get("l", 0) or 0),
            "t": float(bbox.get("t", 0) or 0),
            "r": float(bbox.get("r", 0) or 0),
            "b": float(bbox.get("b", 0) or 0),
            "coordOrigin": str(bbox.get("coordOrigin") or bbox.get("coord_origin") or "TOPLEFT"),
        }
    else:
        bb = {}
    try:
        p = int(page) if page is not None else None
    except (TypeError, ValueError):
        p = None
    return bb, p


def parse_header_year_columns(header_cells: Sequence[Any]) -> list[Optional[int]]:
    years: list[Optional[int]] = []
    for c in header_cells:
        txt = _cell_text(c)
        m = re.search(r"(19\d{2}|20\d{2})", txt)
        years.append(int(m.group(1)) if m else None)
    return years


_EN_MONTH_NAME = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?"
    r"|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
_EN_MONTH_TO_NUM = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def parse_statement_column_date_from_text(text: str) -> Optional[date]:
    """从表头单元格文本解析「可比报告期」日期；无法解析时返回 None。

    支持：
    - 完整中文日：``2025年12月31日``
    - ISO 风格：``2025-12-31``
    - 英文长日期：``31 March 2025``（英式/IFRS）、``March 31, 2025``（美式）
    - 利润表/现金流量表常见 **年度** 列：``2025年度``、``2025 年度`` — 按该会计年度 **年末 12 月 31 日**
      作为可比时点（与多列「YYYY 年度」横向对比、相邻年间隔约 1 年的拆分逻辑一致）。
    """
    if not text or not str(text).strip():
        return None
    raw = str(text).strip()
    compact = re.sub(r"\s+", "", raw)

    # 区间型表头优先取末尾日期，例如：
    # ``2023年5月22日起至2023年12月31日止期间发生额``
    m_range_cn = re.search(
        r"(?:起|自|从)?(?:.*?)(?:至|到)(19\d{2}|20\d{2})年(1[0-2]|0?[1-9])月([0-3]?\d)日?",
        compact,
    )
    if m_range_cn:
        y, mo, d = int(m_range_cn.group(1)), int(m_range_cn.group(2)), int(m_range_cn.group(3))
        try:
            return date(y, mo, d)
        except ValueError:
            return None
    m_range_iso = re.search(
        r"(?:起|自|从)?(?:.*?)(?:至|到)(19\d{2}|20\d{2})[-/.](1[0-2]|0?[1-9])[-/.]([0-3]?\d)",
        compact,
    )
    if m_range_iso:
        y, mo, d = int(m_range_iso.group(1)), int(m_range_iso.group(2)), int(m_range_iso.group(3))
        try:
            return date(y, mo, d)
        except ValueError:
            return None

    m = re.search(
        r"(19\d{2}|20\d{2})年(1[0-2]|0?[1-9])月([0-3]?\d)日?",
        compact,
    )
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(y, mo, d)
        except ValueError:
            return None
    m2 = re.search(
        r"\b(19\d{2}|20\d{2})[-/.](1[0-2]|0?[1-9])[-/.]([0-3]?\d)\b",
        raw,
    )
    if m2:
        y, mo, d = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
        try:
            return date(y, mo, d)
        except ValueError:
            return None
    # 「2025 年度」「2025年度」等（企查查等披露常见）
    m_fy = re.search(r"(19\d{2}|20\d{2})\s*年度", raw)
    if m_fy:
        y = int(m_fy.group(1))
        if 1990 <= y <= 2050:
            return date(y, 12, 31)
    # 英文长日期：``31 March 2025``（英式/IFRS 常见，含无空格粘连 ``31 March2025``）
    m_en_dmy = re.search(
        r"\b([0-3]?\d)\s*(" + _EN_MONTH_NAME + r")\.?\s*,?\s*(19\d{2}|20\d{2})\b",
        raw,
        re.IGNORECASE,
    )
    if m_en_dmy:
        y, mo, d = int(m_en_dmy.group(3)), _EN_MONTH_TO_NUM[m_en_dmy.group(2).lower()], int(m_en_dmy.group(1))
        try:
            return date(y, mo, d)
        except ValueError:
            return None
    # 美式英文日期：``March 31, 2025``
    m_en_mdy = re.search(
        r"\b(" + _EN_MONTH_NAME + r")\.?\s+([0-3]?\d)\s*,?\s*(19\d{2}|20\d{2})\b",
        raw,
        re.IGNORECASE,
    )
    if m_en_mdy:
        y, mo, d = int(m_en_mdy.group(3)), _EN_MONTH_TO_NUM[m_en_mdy.group(1).lower()], int(m_en_mdy.group(2))
        try:
            return date(y, mo, d)
        except ValueError:
            return None
    return None


def _is_statement_metadata_year_row(row: Sequence[Any]) -> bool:
    """识别“编制单位/公司名 + YYYY年 + 单位”这类表级元信息行。"""
    texts = [_cell_text(cell).strip() for cell in row]
    non_empty = [t for t in texts if t]
    if len(non_empty) < 2:
        return False
    joined = "".join(non_empty)
    year_matches = re.findall(r"(19\d{2}|20\d{2})\s*年", joined)
    if len(year_matches) != 1:
        return False
    compact = re.sub(r"\s+", "", joined)
    has_unit = "单位" in compact or any(
        re.fullmatch(r"(?:人民币)?(?:元|万元|千元|百万元)", re.sub(r"\s+", "", t))
        for t in non_empty
    )
    has_preparer = any(k in compact for k in ("编制单位", "编报单位", "制表单位"))
    has_company_like = any(k in compact for k in ("有限公司", "股份", "集团", "公司"))
    strong_column_markers = (
        "本期金额",
        "上期金额",
        "期末余额",
        "期初余额",
        "期末数",
        "期初数",
        "本年累计",
        "上年同期",
        "本期数",
        "上期数",
    )
    has_strong_column_marker = any(k in compact for k in strong_column_markers)
    if has_strong_column_marker:
        return False
    return has_unit or has_preparer or has_company_like


def infer_column_index_to_year_for_table(
    rows: Sequence[Sequence[Any]],
    *,
    context_above: str = "",
    context_below: str = "",
    report_name: str = "",
) -> dict[int, Optional[int]]:
    """数据列索引（1..n-1）-> 表头解析到的年份；首列 0 不在结果中。"""
    raw_rows = [list(r) for r in rows]
    if not raw_rows:
        return {}
    context_near = (context_above or "")[-100:]
    global_years = infer_accounting_period(context_near, context_below, report_name)["years"]
    header_years_map: dict[int, Optional[int]] = {}
    for hi in (0, 1):
        if hi >= len(raw_rows):
            break
        hdr = raw_rows[hi]
        if len(hdr) < 2:
            continue
        if _is_statement_metadata_year_row(hdr):
            continue
        candidate: dict[int, Optional[int]] = {}
        cursor = 0
        for cell in hdr:
            cstart = _cell_col_start(cell)
            if cstart is not None:
                cursor = cstart
            span = _cell_col_span(cell)
            txt = _cell_text(cell)
            m = re.search(r"(19\d{2}|20\d{2})", txt)
            year = int(m.group(1)) if m else None
            for cidx in range(cursor, cursor + span):
                if cidx > 0:
                    candidate[cidx] = year
            cursor += span
        if sum(1 for y in candidate.values() if y is not None) >= 1:
            header_years_map = candidate
            break
    out: dict[int, Optional[int]] = {}
    if header_years_map:
        out.update(header_years_map)
        return out
    if global_years:
        ncol = 0
        for row in raw_rows:
            cursor = 0
            for cell in row:
                cstart = _cell_col_start(cell)
                if cstart is not None:
                    cursor = cstart
                span = _cell_col_span(cell)
                ncol = max(ncol, cursor + span)
                cursor += span
        for col_idx in range(1, ncol):
            out[col_idx] = global_years[0]
        return out
    return {}


def augment_column_years_for_comparative_period_labels(
    rows: Sequence[Sequence[Any]],
    col_year: dict[int, Optional[int]],
    *,
    doc_year: Optional[int],
    doc_period: str,
) -> dict[int, Optional[int]]:
    """在「期末/期初」「本期/上期」等表头无四位年份时，用文档报告年区分两列比较年。

    典型年报：期末（或本期）列 = ``doc_year``，期初 / 上期 / 上年同期列 = ``doc_year - 1``。
    仅在**恰好两列金额**且能识别出一列「期末类」、一列「上期类」语义，且两列原年份
    均为空或**相同**时写入，避免覆盖已正确解析的 2024/2023 等多列年。
    """
    out = dict(col_year)
    if doc_year is None or not (1990 <= int(doc_year) <= 2050):
        return out
    if doc_period not in _FILING_PERIOD_ALLOWED:
        return out

    header_by_col = _infer_column_header_texts(rows, header_rows=4)
    data_cols = sorted(c for c in header_by_col if c > 0)
    if len(data_cols) != 2:
        return out

    def _norm(txt: str) -> str:
        return re.sub(r"\s+", "", txt or "")

    def _is_prior_year_opening_header(txt: str) -> bool:
        t = _norm(txt)
        if not t:
            return False
        if "年初" in t and "期初" not in t:
            return False
        return any(
            k in t
            for k in (
                "期初余额",
                "期初数",
                "期初",
                "上期金额",
                "上期数",
                "上年同期",
                "上年金额",
            )
        )

    def _is_current_period_closing_header(txt: str) -> bool:
        t = _norm(txt)
        if not t:
            return False
        if _is_prior_year_opening_header(txt) and "期末" not in t and "本期" not in t:
            return False
        return any(
            k in t
            for k in (
                "期末余额",
                "期末数",
                "期末",
                "本期金额",
                "本期数",
                "本期",
                "本年累计",
                "本年数",
            )
        )

    c0, c1 = data_cols[0], data_cols[1]
    t0, t1 = header_by_col.get(c0, ""), header_by_col.get(c1, "")

    closing_idx: Optional[int] = None
    opening_idx: Optional[int] = None

    if _is_current_period_closing_header(t0) and _is_prior_year_opening_header(t1) and not _is_prior_year_opening_header(
        t0
    ):
        closing_idx, opening_idx = c0, c1
    elif _is_current_period_closing_header(t1) and _is_prior_year_opening_header(t0) and not _is_prior_year_opening_header(
        t1
    ):
        closing_idx, opening_idx = c1, c0
    elif "期末" in _norm(t0) and "期初" in _norm(t1) and "期初" not in _norm(t0):
        closing_idx, opening_idx = c0, c1
    elif "期末" in _norm(t1) and "期初" in _norm(t0) and "期初" not in _norm(t1):
        closing_idx, opening_idx = c1, c0

    if closing_idx is None or opening_idx is None or closing_idx == opening_idx:
        return out

    yc, yo = out.get(closing_idx), out.get(opening_idx)
    py = int(doc_year) - 1
    if py < 1990:
        return out

    if yc is not None and yo is not None:
        if yc == yo:
            out[closing_idx] = int(doc_year)
            out[opening_idx] = py
        return out

    if yc is not None and yo is None:
        if int(yc) == int(doc_year):
            out[opening_idx] = py
        return out

    if yc is None and yo is not None:
        if int(yo) == py:
            out[closing_idx] = int(doc_year)
        return out

    out[closing_idx] = int(doc_year)
    out[opening_idx] = py
    return out


_NEAR_DATE_ZH_RE = re.compile(r"(19\d{2}|20\d{2})\s*年\s*(1[0-2]|0?[1-9])\s*月\s*([0-3]?\d)\s*日")
_NEAR_DATE_ISO_RE = re.compile(r"\b(19\d{2}|20\d{2})[-/.](1[0-2]|0?[1-9])[-/.]([0-3]?\d)\b")


def _extract_nearest_date_from_context(text: str) -> Optional[tuple[int, int, int]]:
    """从近邻上文中提取最后出现的完整日期，返回 (year, month, day) 或 None。"""
    if not text:
        return None
    last: Optional[tuple[int, int, int, int]] = None
    for m in _NEAR_DATE_ZH_RE.finditer(text):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1990 <= y <= 2050 and 1 <= d <= 31:
            last = (y, mo, d, m.start())
    for m in _NEAR_DATE_ISO_RE.finditer(text):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1990 <= y <= 2050 and 1 <= d <= 31:
            if last is None or m.start() > last[3]:
                last = (y, mo, d, m.start())
    if last is None:
        return None
    return (last[0], last[1], last[2])


def extract_explicit_filing_year_from_blob(blob: str) -> Optional[int]:
    """从披露用语/截至日等文本中提取单一报告年（与 ``_pick_single_reporting_year`` 显式段一致）。"""
    s = (blob or "").strip()
    if not s:
        return None
    for pat in (
        r"(19\d{2}|20\d{2})\s*年\s*第\s*一\s*季度\s*报告",
        r"(19\d{2}|20\d{2})\s*年\s*第\s*三\s*季度\s*报告",
        r"(19\d{2}|20\d{2})\s*半年度报告",
        r"(19\d{2}|20\d{2})\s*年度报告",
        r"(19\d{2}|20\d{2})\s*年报",
    ):
        m = re.search(pat, s)
        if m:
            y = int(m.group(1))
            if 1990 <= y <= 2050:
                return y
    matches_year_du = re.findall(r"(19\d{2}|20\d{2})\s*年度\b", s)
    if matches_year_du:
        candidates = [int(x) for x in matches_year_du if 1990 <= int(x) <= 2050]
        if candidates:
            return max(candidates)
    m = re.search(r"截至\s*(19\d{2}|20\d{2})年", s)
    if m:
        y = int(m.group(1))
        if 1990 <= y <= 2050:
            return y
    m = re.search(r"报告期[：:\s（(]*\s*(19\d{2}|20\d{2})", s)
    if m:
        y = int(m.group(1))
        if 1990 <= y <= 2050:
            return y
    m = re.search(r"(?:year|period)\s+ended\s+.*?\b(19\d{2}|20\d{2})\b", s, re.I)
    if m:
        y = int(m.group(1))
        if 1990 <= y <= 2050:
            return y
    return None


def infer_single_table_reporting_year(
    rows: Sequence[Sequence[Any]],
    *,
    column_year_by_index: Optional[dict[int, Optional[int]]] = None,
    context_above: str = "",
    context_below: str = "",
    report_name: str = "",
    md: str = "",
    caption: str = "",
    fallback_doc_year: Optional[int] = None,
) -> Optional[int]:
    """单张三大表上的「报告所属年」：以本表 md/表头/列年为准，避免整份 PDF 文档锚点年误套。

    审计报告常按年度装订多套报表（如 2022、2023、2024 连续附表）；接口顶层 ``years`` 应对齐
    **当前这张表** 的列报年度（一般为表内可比列中较新的会计年，或表头 ``YYYY年``）。
    """
    md_head = (md or "")[:4200]
    cap_s = (caption or "").strip()
    rn = (report_name or "").strip()

    cy = column_year_by_index
    if cy is None:
        ctx_combined = "\n".join(x for x in (context_above, md, caption, report_name) if x)
        cy = infer_column_index_to_year_for_table(
            rows,
            context_above=ctx_combined[:12000],
            context_below=context_below or "",
            report_name=report_name,
        )
    ctx_near = (context_above or "")[-100:]
    title_blob = "\n".join(x for x in ((context_above or "")[-4500:], cap_s, md_head, rn) if x).strip()
    title_y = _fiscal_year_from_statement_title_area(title_blob)
    if title_y is not None:
        return title_y

    head_blob = "\n".join(x for x in (md_head, cap_s, rn, ctx_near) if x).strip()
    y = extract_explicit_filing_year_from_blob(head_blob)
    if y is not None:
        return y

    vals = [int(v) for v in cy.values() if isinstance(v, int) and 1990 <= v <= 2050]
    if vals:
        return max(vals)

    near_date = _extract_nearest_date_from_context(ctx_near)
    if near_date is not None:
        return near_date[0]

    if rows:
        first_row = list(rows[0]) if rows[0] else []
        hdr0_map = _infer_column_header_texts([first_row], header_rows=1)
        hy_hdr = extract_years_from_text("\n".join(hdr0_map.values())[:4000])
        if hy_hdr:
            return hy_hdr[0]

    hy = extract_years_from_text(md_head)
    if hy:
        return hy[0]
    return fallback_doc_year


def _element_table_caption_plain(el: Dict[str, Any]) -> str:
    cap = el.get("table_caption") or el.get("tableCaption")
    if isinstance(cap, list):
        return "\n".join(str(x) for x in cap if isinstance(x, str))
    return str(cap or "")


def _fiscal_year_from_statement_title_area(blob: str) -> Optional[int]:
    """从表题/上文前几行中解析独立「YYYY年」（排除 ``YYYY年MM月…`` 等完整日期）。

    审计 PDF 中现金流量表常仅有「期末余额 / 期初余额」列无四位年，而「制表单位…YYYY年」
    落在表块上方的独立文本节点；若不并入严格年，会被 peer 前缀填补误判为相邻利润表年。
    """
    if not blob or not str(blob).strip():
        return None
    lines = str(blob).strip().splitlines()
    parts: list[str] = []
    for ln in lines[:24]:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("|") and s.count("|") >= 4:
            break
        parts.append(s)
        if sum(len(x) for x in parts) > 3600:
            break
    head = "\n".join(parts).strip()
    if not head:
        return None
    bare_years: list[int] = []
    for m in re.finditer(r"(19\d{2}|20\d{2})年", head):
        tail = head[m.end() : m.end() + 14]
        if re.match(r"^\s*\d{1,2}\s*月", tail):
            continue
        y = int(m.group(1))
        if 1990 <= y <= 2050:
            bare_years.append(y)
    if not bare_years:
        return None
    if len(bare_years) == 1:
        return bare_years[0]
    return bare_years[-1]


def _strict_reporting_year_for_peer_gap_table(
    el: Dict[str, Any],
    *,
    items: Optional[Sequence[Any]] = None,
    item_idx: Optional[int] = None,
) -> Optional[int]:
    """与 ``interpolate_*`` 一致的「无文档兜底」单表严格报告年。"""
    rows = el.get("rows")
    if not isinstance(rows, list) or not rows:
        return None
    tt = str(el.get("table_type") or "")
    md = table_element_markdown(el)
    cap_s = _element_table_caption_plain(el)
    neighbor_above = ""
    if items is not None and item_idx is not None:
        neighbor_above, _ = collect_neighbor_text_context(items, item_idx)
    ctx_top = "\n".join(x for x in (neighbor_above[-4500:], cap_s, md[:8000]) if x).strip()
    cy = infer_column_index_to_year_for_table(
        rows,
        context_above=ctx_top,
        context_below="",
        report_name=tt,
    )
    y = infer_single_table_reporting_year(
        rows,
        column_year_by_index=dict(cy),
        context_above=ctx_top[:12000],
        context_below="",
        report_name=tt,
        md=md,
        caption=cap_s,
        fallback_doc_year=None,
    )
    if y is not None:
        return y
    title_blob = "\n".join(x for x in (neighbor_above[-6000:], cap_s, md[:10000]) if x)
    return _fiscal_year_from_statement_title_area(title_blob)


def _max_strict_reporting_year_before_item_excluding_cap(
    items: Sequence[Any],
    before_idx: int,
    cap_year: int,
) -> Optional[int]:
    """在 ``items[0:before_idx]`` 中找 BS/IS/CF 已解析严格年，且严格小于 *cap_year* 的最大值。"""
    found: list[int] = []
    for i in range(max(0, before_idx)):
        el = items[i]
        if not isinstance(el, dict) or el.get("type") not in {"table", "excel_table"}:
            continue
        tt = str(el.get("table_type") or "")
        if map_to_statement_type(tt, tt) not in {"BS", "IS", "CF"}:
            continue
        sy = _strict_reporting_year_for_peer_gap_table(el, items=items, item_idx=i)
        if sy is not None and int(sy) < int(cap_year):
            found.append(int(sy))
    return max(found) if found else None


def _peer_year_gap_fill_for_statement_chain(
    blocks: list[tuple[int, int, Any, Optional[int]]],
    years_list: list[Optional[int]],
    items: Sequence[Any],
) -> Dict[Any, int]:
    """对单一报表类型（已按页、item 序排好）做中间段 + 前缀段年份填补。"""
    filled = list(years_list)
    n = len(filled)
    i = 0
    while i < n:
        if filled[i] is not None:
            i += 1
            continue
        j = i - 1
        while j >= 0 and filled[j] is None:
            j -= 1
        k = i
        while k < n and filled[k] is None:
            k += 1
        if k >= n:
            break
        if j < 0 or filled[j] is None or filled[k] is None:
            i = k + 1
            continue
        yj, yk = int(filled[j]), int(filled[k])
        span = k - j
        diff = yk - yj
        if span <= 0 or diff <= 0:
            i = k + 1
            continue
        interior_strict_none = all(years_list[t] is None for t in range(j + 1, k))
        if diff == span:
            for t in range(j + 1, k):
                filled[t] = yj + (t - j)
        elif yk == yj + 2 and interior_strict_none:
            for t in range(j + 1, k):
                filled[t] = yj + 1
        else:
            i = k + 1
            continue
        i = k + 1

    k0 = next((i for i, v in enumerate(filled) if v is not None), None)
    if k0 is not None and k0 > 0 and all(years_list[i] is None for i in range(k0)):
        yk = int(filled[k0])
        first_item_idx = blocks[0][1]
        y_prev = _max_strict_reporting_year_before_item_excluding_cap(items, first_item_idx, yk)
        if y_prev is not None:
            if yk == y_prev + 1:
                for t in range(k0):
                    filled[t] = y_prev
            elif yk == y_prev + 2:
                for t in range(k0):
                    filled[t] = y_prev + 1

    out: Dict[Any, int] = {}
    for i, b in enumerate(blocks):
        if years_list[i] is None and filled[i] is not None:
            eid = b[2]
            if eid is not None:
                out[eid] = int(filled[i])
    return out


def _norm_cf_amount_token(s: str) -> str:
    return re.sub(r"[,\s，]", "", (s or "").strip())


def _norm_cf_line_item_label(s: str) -> str:
    return re.sub(r"[\s：:]", "", (s or "").strip())


def _logical_col_text_map(row: Sequence[Any], *, max_logical: int = 8) -> dict[int, str]:
    """将一行单元格展开为「逻辑列索引 -> 文本」，兼容 MinerU 仅对部分格带 colStart 的版式。"""
    out: dict[int, str] = {}
    col_cursor = 0
    for cell in row:
        if not isinstance(cell, dict):
            continue
        cstart = _cell_col_start(cell)
        span = _cell_col_span(cell)
        start = int(cstart) if cstart is not None else col_cursor
        txt = _cell_text(cell).strip()
        for j in range(start, min(max_logical + 1, start + span)):
            if txt:
                out[j] = txt
        col_cursor = start + span
    return out


def _find_cf_close_open_data_columns(rows: Sequence[Sequence[Any]]) -> Optional[tuple[int, int]]:
    """返回 (期末类列索引, 期初类列索引)；仅适用于「项目 + 两列金额」的常见现金流量表。"""
    ht = _infer_column_header_texts(rows, header_rows=4)
    close_i: Optional[int] = None
    open_i: Optional[int] = None
    for c in sorted(ht.keys()):
        if c <= 0:
            continue
        t = re.sub(r"\s+", "", ht.get(c) or "")
        if not t:
            continue
        has_open = any(
            k in t
            for k in ("期初余额", "期初数", "期初", "上期金额", "上期数", "上年同期", "上期")
        )
        has_close = any(
            k in t
            for k in ("期末余额", "期末数", "期末", "本期金额", "本期数", "本期", "本年")
        )
        year_only = bool(re.search(r"(19\d{2}|20\d{2})年", t)) and "月" not in t and "单位" not in t
        if has_open and not has_close:
            if open_i is None:
                open_i = c
        elif has_close or (year_only and not has_open):
            if close_i is None:
                close_i = c
    if close_i is None or open_i is None or close_i == open_i:
        return None
    return close_i, open_i


def _cf_consecutive_year_bridge_by_amounts(
    prev_rows: Sequence[Sequence[Any]],
    next_rows: Sequence[Sequence[Any]],
) -> bool:
    """若「次表期初列金额」与「前表期末列金额」在多行上一致，则认为两表为连续会计年度（次 = 前 + 1）。"""
    po = _find_cf_close_open_data_columns(prev_rows)
    pn = _find_cf_close_open_data_columns(next_rows)
    if po is None or pn is None:
        return False
    prev_close_i, _prev_open_i = po
    _next_close_i, next_open_i = pn
    hits = 0
    for pr in prev_rows[1:55]:
        pm = _logical_col_text_map(pr)
        if 0 not in pm:
            continue
        item_p = _norm_cf_line_item_label(pm.get(0, ""))
        amt_pc = _norm_cf_amount_token(pm.get(prev_close_i, ""))
        if len(amt_pc) < 4 or not re.fullmatch(r"-?[\d.]+", amt_pc):
            continue
        for nr in next_rows[1:55]:
            nm = _logical_col_text_map(nr)
            if 0 not in nm:
                continue
            item_n = _norm_cf_line_item_label(nm.get(0, ""))
            if item_p != item_n or len(item_n) < 4:
                continue
            amt_no = _norm_cf_amount_token(nm.get(next_open_i, ""))
            if amt_pc == amt_no and amt_no:
                hits += 1
                if hits >= 2:
                    return True
                if "销售商品" in item_p and "销售商品" in item_n:
                    return True
    return False


def _cf_md_banner_anchor_year(md: str) -> Optional[int]:
    """现金流量表 md 前几行中的表头报告年（如 ``| 项目 | 2024年 | 单位：元 |``）。"""
    s = (md or "").strip()
    if not s:
        return None
    head = "\n".join(s.splitlines()[:8])
    ys: list[int] = []
    for m in re.finditer(r"(19\d{2}|20\d{2})年", head):
        tail = head[m.end() : m.end() + 14]
        if re.match(r"^\s*\d{1,2}\s*月", tail):
            continue
        y = int(m.group(1))
        if 1990 <= y <= 2050:
            ys.append(y)
    if not ys:
        return None
    if len(ys) == 1:
        return ys[0]
    return ys[-1]


def _amount_chain_reporting_years_for_stmt(
    items: Sequence[Any], stmt: StatementTypeCode
) -> Dict[Any, int]:
    """用「次表期初 = 前表期末」金额链矫正单一报表类型（BS/IS/CF）报告年，覆盖 peer 前缀误填。

    典型审计 PDF 连续附「2022 / 2023 / 2024」三套报表：仅末套 md 带 ``2024年``，前两套仅有
    期末/期初列无四位年；仅凭 peer 前缀会把 2022 套误标为 2023。无法识别「期末/期初」两列的
    碎片表（如利润表续表的其他综合收益尾部）会被跳过，避免夹在中间打断相邻金额链。
    """
    blocks: list[dict[str, Any]] = []
    for idx, el in enumerate(items):
        if not isinstance(el, dict) or el.get("type") not in {"table", "excel_table"}:
            continue
        tt = str(el.get("table_type") or "")
        if map_to_statement_type(tt, tt) != stmt:
            continue
        rows = el.get("rows")
        if not isinstance(rows, list) or len(rows) < 2:
            continue
        # 跳过无法识别「期末/期初」两列的碎片表（如利润表续表的其他综合收益/每股收益尾部），
        # 否则它夹在两套主表之间会打断相邻金额链。
        if _find_cf_close_open_data_columns(rows) is None:
            continue
        page_raw = el.get("page")
        try:
            page = int(page_raw) if page_raw is not None else 0
        except (TypeError, ValueError):
            page = 0
        blocks.append(
            {
                "page": page,
                "idx": idx,
                "eid": el.get("element_id"),
                "rows": rows,
                "md": table_element_markdown(el),
            }
        )
    if len(blocks) < 2:
        return {}
    blocks.sort(key=lambda x: (x["page"], x["idx"]))
    n = len(blocks)
    link_ok = [False] * (n - 1)
    for i in range(n - 1):
        link_ok[i] = _cf_consecutive_year_bridge_by_amounts(blocks[i]["rows"], blocks[i + 1]["rows"])
    anchor: list[Optional[int]] = []
    for b in blocks:
        yb = _cf_md_banner_anchor_year(str(b.get("md") or ""))
        anchor.append(yb if yb is not None and 1990 <= int(yb) <= 2050 else None)
    if not any(y is not None for y in anchor):
        return {}
    years: list[Optional[int]] = [None] * n
    for i, y in enumerate(anchor):
        if y is not None:
            years[i] = int(y)
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if years[i] is None:
                continue
            j = i - 1
            while j >= 0 and link_ok[j]:
                expect = int(years[j + 1]) - 1
                if years[j] is None:
                    years[j] = expect
                    changed = True
                elif years[j] != expect:
                    break
                j -= 1
            j = i + 1
            while j < n and link_ok[j - 1]:
                expect = int(years[j - 1]) + 1
                if years[j] is None:
                    years[j] = expect
                    changed = True
                elif years[j] != expect:
                    break
                j += 1
    out: Dict[Any, int] = {}
    for i, b in enumerate(blocks):
        y = years[i]
        if y is None:
            continue
        eid = b.get("eid")
        if eid is not None:
            out[eid] = int(y)
    return out


def interpolate_balance_sheet_reporting_years_by_peer_pages(
    items: Sequence[Any],
) -> Dict[Any, int]:
    """当某张三大表 md/表头无四位年、单列推断亦无时，用文档序前后「同类型表」及前序其他三大表已解析年做填补。

    - **资产负债表 / 利润表 / 现金流量表** 各自按 item 顺序成链，中间 ``2022,None,2024`` 等仍用等差 /
      ``Y`` 与 ``Y+2`` 夹心规则。
    - **前缀全空**（如现金流量表 p7、p9 无 md 年，首条有年为 p11 的 2024）：取前序任意 BS/IS/CF
      中已解析且 **小于** 该锚点年的最大严格年（如利润表 p3 的 2023），若 ``Yk==Yprev+1`` 或 ``+2``，
      则将前缀表标为 ``Yprev`` 或 ``Yprev+1``，以覆盖「表头 2023 年未进 MinerU md」的现金流量表。
    - **严格年**计算会并入表前邻接文本，并对表题区独立「YYYY年」做兜底，避免将「制表单位…2022年」
      等历史表误套为后文锚点的前缀填补年。
    - **现金流量表金额链**：若 md 中至少有一套表带表头年（如 ``2024年``），且相邻两套表满足
      「次表期初列金额 = 前表期末列金额」行匹配，则按链回推/顺推各套报告年，**覆盖** peer 前缀误填。
    """
    merged: Dict[Any, int] = {}
    for stmt in ("BS", "IS", "CF"):
        blocks: list[tuple[int, int, Any, Optional[int]]] = []
        for idx, el in enumerate(items):
            if not isinstance(el, dict) or el.get("type") not in {"table", "excel_table"}:
                continue
            tt = str(el.get("table_type") or "")
            if map_to_statement_type(tt, tt) != stmt:
                continue
            rows = el.get("rows")
            if not isinstance(rows, list) or not rows:
                continue
            page_raw = el.get("page")
            try:
                page = int(page_raw) if page_raw is not None else 0
            except (TypeError, ValueError):
                page = 0
            strict_y = _strict_reporting_year_for_peer_gap_table(el, items=items, item_idx=idx)
            eid = el.get("element_id")
            blocks.append((page, idx, eid, strict_y))
        if not blocks:
            continue
        blocks.sort(key=lambda x: (x[0], x[1]))
        years_list: list[Optional[int]] = [b[3] for b in blocks]
        merged.update(_peer_year_gap_fill_for_statement_chain(blocks, years_list, items))
        merged.update(_amount_chain_reporting_years_for_stmt(items, stmt))
    return merged


def _cell_col_start(cell: Any) -> Optional[int]:
    if isinstance(cell, dict):
        v = cell.get("colStart") or cell.get("colstart") or cell.get("col_start")
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None
    return None


def _cell_col_span(cell: Any) -> int:
    if isinstance(cell, dict):
        v = cell.get("colSpan") or cell.get("colspan") or cell.get("col_span") or 1
        try:
            return max(1, int(v))
        except (TypeError, ValueError):
            return 1
    return 1


def _is_banner_row(row: Sequence[Any]) -> bool:
    """检测「章节横幅行」——只有一个跨多列的非空文本。

    这种行视觉上是一个跨列章节标题（如 ``STATEMENT OF COMPREHENSIVE INCOME
    CONSOLIDATED AND SEPARATE``），分两种形态：

    1. 单 cell + ``colSpan >= 2``（PDF 解析正确识别合并）；
    2. 多 cell 且每个 cell 文本完全相同（PDF 解析未识别合并，把同一文本复制到
       每个 cell）。

    若被当作表头并入列文本，会把章节里的「consolidated / standalone / group /
    company」等口径词污染到不相关的金额列，导致按列推断口径失败（出现整列
    被误判成「混合」从而无法 dual-caliber 拆分）。
    """
    non_empty = [(cell, _cell_text(cell)) for cell in row if _cell_text(cell)]
    if not non_empty:
        return False

    if len(non_empty) == 1:
        cell, _ = non_empty[0]
        return _cell_col_span(cell) >= 2

    first_text = non_empty[0][1]
    if all(t == first_text for _, t in non_empty[1:]):
        total_span = sum(_cell_col_span(cell) for cell, _ in non_empty)
        return total_span >= 2
    return False


def _caliber_header_focus_text(htxt: str) -> str:
    """从多行列表头中取出用于「合并/单体」打分的片段。

    MinerU 等解析结果常把 ``STATEMENT OF … CONSOLIDATED AND SEPARATE`` 类章节横幅
    重复拼进**每一列**的表头单元格。若整段参与打分，会同时命中 ``consolidated`` 与
    ``standalone``，列被判为「混合」，进而无法触发 ``_grp``/``_std`` 双列拆分
    （如 Omani 报表中「综合收益表」与「利润表」并列 Cons/Stand 四金额列）。

    规则：优先使用**首行**若其已含列级口径词；否则在后续行中找首条含口径词的行；
    否则退回首行。
    """
    text = (htxt or "").strip()
    if not text:
        return ""
    lines = [x.strip() for x in text.split("\n") if x.strip()]
    if not lines:
        return ""
    markers_lower = (
        "consolidated",
        "standalone",
        "group",
        "company",
        "bank",
        "合并",
        "母公司",
        "单体",
    )
    first = lines[0]
    if any(m in first.lower() for m in markers_lower):
        return first
    for ln in lines[1:]:
        if any(m in ln.lower() for m in markers_lower):
            return ln
    return first


def _infer_column_header_texts(rows: Sequence[Sequence[Any]], header_rows: int = 3) -> dict[int, str]:
    """提取每列表头文本，支持多层表头与 colSpan 展开。

    会自动跳过 banner 行（见 ``_is_banner_row``），避免章节横幅文本污染列口径。
    """
    raw_rows = [list(r) for r in rows]
    out: dict[int, list[str]] = {}
    for row in raw_rows[:header_rows]:
        if not row:
            continue
        if _is_banner_row(row):
            continue
        cursor = 0
        for cell in row:
            txt = _cell_text(cell)
            cstart = _cell_col_start(cell)
            if cstart is not None:
                cursor = cstart
            span = _cell_col_span(cell)
            for cidx in range(cursor, cursor + span):
                if txt:
                    out.setdefault(cidx, []).append(txt)
            cursor += span
    return {k: "\n".join(v).strip() for k, v in out.items()}


def _extract_column_header_texts_from_header_cells(
    header_cells: Sequence[Sequence[Any]],
) -> dict[int, str]:
    """从多行 ``headerCells`` 提取每列表头文本。

    与 :func:`_infer_column_header_texts` 行为对齐（按 colStart/colSpan 展开、跳过 banner 行、
    多行用 ``\n`` 拼接），但输入是干净的多行表头，避免依赖 ``rows[0]`` 拍平后的语义损失。
    """
    out: dict[int, list[str]] = {}
    for row in header_cells:
        if not row:
            continue
        if _is_banner_row(row):
            continue
        cursor = 0
        for cell in row:
            txt = _cell_text(cell)
            cstart = _cell_col_start(cell)
            if cstart is not None:
                cursor = cstart
            span = _cell_col_span(cell)
            for cidx in range(cursor, cursor + span):
                if txt:
                    out.setdefault(cidx, []).append(txt)
            cursor += span
    return {k: "\n".join(v).strip() for k, v in out.items()}


def infer_column_index_to_date_for_table(
    rows: Sequence[Sequence[Any]],
    *,
    context_above: str = "",
    context_below: str = "",
    report_name: str = "",
) -> dict[int, Optional[date]]:
    """逻辑列索引（与 ``colStart`` / 表头推断一致）-> 表头解析到的报告期日期；列 0 不在结果中。

    当表头为完整年月日、ISO 日期或「YYYY 年度」等形式时写入对应日期（年度列按该年 12 月 31 日）；
    不做「全局年份摊到各列」式回填，以免误伤非多期表。
    """
    _ = (context_above, context_below, report_name)
    raw_rows = [list(r) for r in rows]
    if not raw_rows:
        return {}
    header_texts = _infer_column_header_texts(raw_rows, header_rows=4)
    return _resolve_column_dates_from_header_texts(header_texts)


def _resolve_column_dates_from_header_texts(
    header_texts: dict[int, str],
) -> dict[int, Optional[date]]:
    out: dict[int, Optional[date]] = {}
    for cidx, htxt in header_texts.items():
        if cidx <= 0:
            continue
        dt = parse_statement_column_date_from_text(htxt)
        if dt is not None:
            out[cidx] = dt
    return out


def infer_column_index_to_date_from_header_cells(
    header_cells: Sequence[Sequence[Any]],
) -> dict[int, Optional[date]]:
    """基于 ``headerCells`` 多行表头按列推断报告期日期；列 0 不在结果中。"""
    if not header_cells:
        return {}
    header_texts = _extract_column_header_texts_from_header_cells(header_cells)
    return _resolve_column_dates_from_header_texts(header_texts)


def _refine_caliber_for_group_company_dual_columns(
    header_texts: dict[int, str],
    col_out: dict[int, CaliberLabel],
) -> dict[int, CaliberLabel]:
    """修正并列 Group / Company（及 Group / Bank）列段。

    IFRS / 马来上市常见：先 Group 再 Company；子列头只有年份时原先会整表回退 table_level，
    导致 Company（或 Bank）金额列被标成合并，拆分单体表时列错位或无法触发双列拆分。
    """
    idxs = sorted(set(col_out.keys()) | set(header_texts.keys()))
    if not idxs:
        return col_out

    group_anchor: Optional[int] = None
    # Company（单体）或 Bank（单体银行报表）作为第二段锚点
    standalone_anchor: Optional[int] = None
    for c in idxs:
        t = header_texts.get(c, "")
        low = t.lower()
        if group_anchor is None and re.search(r"\bgroup\b", low) and not re.search(
            r"\bcompany\b", low
        ):
            group_anchor = c
        if standalone_anchor is None:
            if re.search(r"\bcompany\b", low):
                standalone_anchor = c
            elif re.search(r"\bbank\b", low):
                standalone_anchor = c

    if (
        group_anchor is None
        or standalone_anchor is None
        or group_anchor >= standalone_anchor
    ):
        return col_out

    for c in idxs:
        if c <= 0:
            continue
        if col_out.get(c) == "混合":
            continue
        ht = header_texts.get(c, "")
        low = ht.lower().strip()
        # 已有明确 Group/Company/Bank 或中英口径词：保持已由打分得到的结果
        if re.search(r"\bgroup\b", low) or re.search(r"\bcompany\b", low) or re.search(
            r"\bbank\b", low
        ):
            continue
        if any(k in ht for k in ("合并", "单体", "母公司", "consolidated", "standalone")):
            continue
        # 其余（纯年份、RM、空、科目复述等）：按列位置归入 Group 段或单体段
        if group_anchor <= c < standalone_anchor:
            col_out[c] = "合并"
        elif c >= standalone_anchor:
            col_out[c] = "单体"
    return col_out


def infer_column_index_to_caliber_for_table(
    rows: Sequence[Sequence[Any]],
    *,
    table_caliber: Optional[CaliberLabel] = None,
    report_name: str = "",
) -> dict[int, CaliberLabel]:
    """按列推断口径（合并/单体/混合/未知），并处理 Company 歧义。"""
    header_texts = _infer_column_header_texts(rows)
    return _score_caliber_from_header_texts(
        header_texts, table_caliber=table_caliber, report_name=report_name
    )


def infer_column_index_to_caliber_from_header_cells(
    header_cells: Sequence[Sequence[Any]],
    *,
    table_caliber: Optional[CaliberLabel] = None,
    report_name: str = "",
) -> dict[int, CaliberLabel]:
    """基于 ``headerCells`` 多行表头按列推断口径（合并/单体/混合/未知）。"""
    if not header_cells:
        return {}
    header_texts = _extract_column_header_texts_from_header_cells(header_cells)
    return _score_caliber_from_header_texts(
        header_texts, table_caliber=table_caliber, report_name=report_name
    )


def _score_caliber_from_header_texts(
    header_texts: dict[int, str],
    *,
    table_caliber: Optional[CaliberLabel] = None,
    report_name: str = "",
) -> dict[int, CaliberLabel]:
    table_level = table_caliber or detect_caliber_label(report_name)
    group_tokens = ("group", "consolidated", "合并")
    parent_strong_tokens = ("standalone", "parent only", "parent company", "母公司", "单体")
    parent_weak_tokens = ("company", "parent")

    interim: dict[int, tuple[int, int, bool]] = {}
    # 值: (group_score, parent_score, has_company_weak)
    for cidx, htxt in header_texts.items():
        focus = _caliber_header_focus_text(htxt)
        low = focus.lower()
        group_score = sum(1 for t in group_tokens if t in low)
        parent_score = 2 * sum(1 for t in parent_strong_tokens if t in low)
        has_company_weak = any(t in low for t in parent_weak_tokens)
        if re.search(r"\bbank\b", low):
            has_company_weak = True
        if has_company_weak:
            parent_score += 1
        # 列级语义仅看该列头，避免被 report_name 的“合并”噪声污染。
        zh_signal = detect_caliber_label(focus)
        if zh_signal == "合并":
            group_score += 2
        elif zh_signal == "单体":
            parent_score += 2
        elif zh_signal == "混合":
            group_score += 1
            parent_score += 1
        interim[cidx] = (group_score, parent_score, has_company_weak)

    has_explicit_group_col = any(g > 0 for g, _, _ in interim.values())
    out: dict[int, CaliberLabel] = {}
    for cidx, (g_score, p_score, has_company_weak) in interim.items():
        if g_score > 0 and p_score > 0:
            out[cidx] = "混合"
            continue
        if g_score > p_score:
            out[cidx] = "合并"
            continue
        if p_score > g_score:
            # 仅 company 弱信号时，若全表没有 group 证据，优先回退表级，减少误判。
            if has_company_weak and p_score <= 1 and not has_explicit_group_col:
                out[cidx] = table_level if table_level != "未知" else "未知"
            else:
                out[cidx] = "单体"
            continue
        out[cidx] = table_level
    return _refine_caliber_for_group_company_dual_columns(header_texts, out)


def infer_column_index_to_statement_type_for_table(
    rows: Sequence[Sequence[Any]],
    *,
    report_name: str = "",
    fallback_statement_type: Optional[StatementTypeCode] = None,
) -> dict[int, Optional[StatementTypeCode]]:
    """按列表头文本推断 statementType；缺失时回退到表级类型。"""
    header_texts = _infer_column_header_texts(rows)
    return _score_statement_type_from_header_texts(
        header_texts, report_name=report_name, fallback_statement_type=fallback_statement_type
    )


def infer_column_index_to_statement_type_from_header_cells(
    header_cells: Sequence[Sequence[Any]],
    *,
    report_name: str = "",
    fallback_statement_type: Optional[StatementTypeCode] = None,
) -> dict[int, Optional[StatementTypeCode]]:
    """基于 ``headerCells`` 多行表头按列推断 statementType；缺失时回退到表级类型。"""
    if not header_cells:
        return {}
    header_texts = _extract_column_header_texts_from_header_cells(header_cells)
    return _score_statement_type_from_header_texts(
        header_texts, report_name=report_name, fallback_statement_type=fallback_statement_type
    )


def _score_statement_type_from_header_texts(
    header_texts: dict[int, str],
    *,
    report_name: str = "",
    fallback_statement_type: Optional[StatementTypeCode] = None,
) -> dict[int, Optional[StatementTypeCode]]:
    out: dict[int, Optional[StatementTypeCode]] = {}
    for cidx, htxt in header_texts.items():
        out[cidx] = map_to_statement_type(htxt, report_name) or fallback_statement_type
    return out


def infer_column_index_to_gaap_for_table(
    rows: Sequence[Sequence[Any]],
    *,
    report_name: str = "",
    context_blob: str = "",
    fallback_gaap: Optional[GaapTypeCode] = None,
) -> dict[int, Optional[GaapTypeCode]]:
    """按列表头文本推断 gaapType；缺失时回退到表级类型。"""
    header_texts = _infer_column_header_texts(rows)
    return _score_gaap_from_header_texts(
        header_texts, report_name=report_name, context_blob=context_blob, fallback_gaap=fallback_gaap
    )


def infer_column_index_to_gaap_from_header_cells(
    header_cells: Sequence[Sequence[Any]],
    *,
    report_name: str = "",
    context_blob: str = "",
    fallback_gaap: Optional[GaapTypeCode] = None,
) -> dict[int, Optional[GaapTypeCode]]:
    """基于 ``headerCells`` 多行表头按列推断 gaapType；缺失时回退到表级类型。"""
    if not header_cells:
        return {}
    header_texts = _extract_column_header_texts_from_header_cells(header_cells)
    return _score_gaap_from_header_texts(
        header_texts, report_name=report_name, context_blob=context_blob, fallback_gaap=fallback_gaap
    )


def _score_gaap_from_header_texts(
    header_texts: dict[int, str],
    *,
    report_name: str = "",
    context_blob: str = "",
    fallback_gaap: Optional[GaapTypeCode] = None,
) -> dict[int, Optional[GaapTypeCode]]:
    out: dict[int, Optional[GaapTypeCode]] = {}
    for cidx, htxt in header_texts.items():
        out[cidx] = detect_gaap_type(htxt, report_name, context_blob) or fallback_gaap
    return out


def match_caliber_label(report_caliber: Optional[int], report_name: str) -> CaliberLabel:
    """结合 reportCaliber 与报表名称得到 合并/单体/混合/未知。"""
    blob = report_name or ""
    if report_caliber == 2:
        blob = f"合并\n{blob}"
    elif report_caliber == 1:
        blob = f"母公司\n{blob}"
    return detect_caliber_label(blob)


def extract_financial_facts_from_grid(
    rows: Sequence[Sequence[Any]],
    *,
    context_above: str,
    context_below: str,
    report_name: str = "",
    fallback_page_at: Optional[int] = None,
) -> list[dict[str, Any]]:
    """从二维单元格网格抽取事实行：科目名、年份、口径、值、bbox。

    假设：首列为科目，右侧列为金额；表头 1～2 行内可解析年份。
    """
    if not rows:
        return []
    raw_rows = [list(r) for r in rows]
    caliber = detect_caliber_label(context_above, context_below, report_name)

    global_years = infer_accounting_period(context_above, context_below, report_name)["years"]

    header_years: list[Optional[int]] = []
    data_start = 0
    for hi in (0, 1):
        if hi >= len(raw_rows):
            break
        hdr = raw_rows[hi]
        if len(hdr) < 2:
            continue
        candidate = parse_header_year_columns(hdr[1:])
        if sum(1 for y in candidate if y is not None) >= 1:
            header_years = candidate
            data_start = hi + 1
            break

    if not header_years and global_years:
        first = raw_rows[0] if raw_rows else []
        if len(first) > 1:
            header_years = [global_years[0]] * (len(first) - 1)
            data_start = 1

    facts: list[dict[str, Any]] = []
    for ri in range(data_start, len(raw_rows)):
        row = raw_rows[ri]
        if not row:
            continue
        subject = _cell_text(row[0])
        if not subject or len(subject) < 2:
            continue
        if subject in {"项目", "科目", "资产", "负债", "附注", "行次", "序号"}:
            continue
        if _VALUE_TEXT_RE.match(subject):
            continue

        value_cells = row[1:]
        if header_years:
            for j, cell in enumerate(value_cells):
                if j >= len(header_years):
                    break
                yr = header_years[j]
                if yr is None:
                    continue
                val_txt = _cell_text(cell)
                if not val_txt or not _VALUE_TEXT_RE.match(val_txt):
                    continue
                bb, pg = _cell_bbox_page(cell)
                facts.append(
                    {
                        "subject_name": subject,
                        "accounting_year": yr,
                        "caliber": caliber,
                        "value_text": val_txt,
                        "bBox": bb,
                        "pageAt": pg if pg is not None else fallback_page_at,
                    }
                )
        else:
            for j, cell in enumerate(value_cells):
                val_txt = _cell_text(cell)
                if not val_txt or not _VALUE_TEXT_RE.match(val_txt):
                    continue
                bb, pg = _cell_bbox_page(cell)
                yr_guess = global_years[0] if global_years else None
                facts.append(
                    {
                        "subject_name": subject,
                        "accounting_year": yr_guess,
                        "caliber": caliber,
                        "value_text": val_txt,
                        "bBox": bb,
                        "pageAt": pg if pg is not None else fallback_page_at,
                    }
                )
                break

    return facts


# ---------------------------------------------------------------------------
# 文档级报告年度 years（单一整数）与财报周期 period（与 doc_extract FIN_EXTRACT_RULES 对齐）
# ---------------------------------------------------------------------------

_FILING_PERIOD_ALLOWED: tuple[str, ...] = ("年报", "半年报", "一季度报", "三季度报", "月报")

_REPORTING_DATE_POSITIVE_RE = re.compile(
    r"报告期|截至|资产负债表日|末日|期末|as\s+at|balance\s+sheet\s+date|"
    r"year\s+ended|statement\s+of\s+financial\s+position|"
    r"for\s+the\s+(?:year|period)\s+ended",
    re.I,
)
_REPORTING_DATE_NEGATIVE_RE = re.compile(
    r"签署|签章|签字|批准日期|核准之日|披露日期|发布日期|之日期|"
    r"date\s+of\s+(?:approval|authorisation|authorization)|signed\s+on|"
    r"board\s+of\s+directors.*date",
    re.I,
)

_DOC_DATE_ISO_RE = re.compile(r"\b(19\d{2}|20\d{2})[-/.](1[0-2]|0?[1-9])[-/.]([0-3]?\d)\b")
_DOC_DATE_ZH_RE = re.compile(r"(19\d{2}|20\d{2})\s*年\s*(1[0-2]|0?[1-9])\s*月\s*([0-3]?\d)\s*日")


def collect_document_head_text(items: Sequence[Any], *, max_elements: int = 120, max_chars: int = 14000) -> str:
    """封面、目录等篇首文本（跳过 table/excel_table，避免表内日期噪声）。"""
    chunks: list[str] = []
    n = 0
    for el in items:
        if n >= max_elements:
            break
        n += 1
        if not isinstance(el, dict):
            continue
        if el.get("type") in {"table", "excel_table"}:
            continue
        t = _element_plain_text(el)
        if t:
            chunks.append(t)
        if sum(len(x) for x in chunks) >= max_chars:
            break
    return "\n".join(chunks)[:max_chars]


def collect_financial_table_text_snippets(
    items: Sequence[Any], *, max_tables: int = 8, max_chars_per: int = 2800
) -> str:
    """三大表 md / 标题，用于报告期截止日与「年度报表」语义。"""
    parts: list[str] = []
    n = 0
    for el in items:
        if not isinstance(el, dict):
            continue
        if el.get("type") not in {"table", "excel_table"}:
            continue
        tt = str(el.get("table_type") or "")
        stmt = map_to_statement_type(tt, tt)
        if stmt not in {"BS", "IS", "CF"}:
            continue
        n += 1
        if n > max_tables:
            break
        cap = el.get("table_caption") or el.get("tableCaption")
        cap_s = "\n".join(str(x) for x in cap if isinstance(x, str)) if isinstance(cap, list) else str(cap or "")
        md = table_element_markdown(el)
        blob = "\n".join(x for x in (cap_s, md) if x).strip()
        if blob:
            parts.append(blob[:max_chars_per])
    return "\n".join(parts)


def collect_first_balance_sheet_column_years(
    items: Sequence[Any],
    *,
    max_tables: int = 40,
    min_rows: int = 8,
) -> list[int]:
    """文档序下首张「足够大」且表头列能解析出年份的资产负债表之年份（降序）。

    同一 PDF 可能含多套报表（如不同审计年度或附录重述）；合并全部 BS 的列年份后取 max
    易被后面的表覆盖。以**首张**满足行数阈值的 BS 为准，与常见「先出主表、后出附表」版式一致。
    """
    n = 0
    for el in items:
        if not isinstance(el, dict) or el.get("type") not in {"table", "excel_table"}:
            continue
        tt = str(el.get("table_type") or "")
        if map_to_statement_type(tt, tt) != "BS":
            continue
        n += 1
        if n > max_tables:
            break
        rows = el.get("rows")
        if not isinstance(rows, list) or len(rows) < min_rows:
            continue
        col_y = infer_column_index_to_year_for_table(
            rows,
            context_above="",
            context_below="",
            report_name=tt,
        )
        ys = sorted(
            {int(y) for y in col_y.values() if isinstance(y, int) and 1990 <= y <= 2050},
            reverse=True,
        )
        if ys:
            return ys
    return []


def collect_column_years_from_statement_tables(items: Sequence[Any], *, max_tables: int = 14) -> list[int]:
    """从三大表 **表头列** 解析年份（不含 caption/md 全文正则，避免脚注/审计用语污染）。

    审计报告常在篇首含「于 2024 年 3 月出具」等日期；若与表体「2022 年 12 月 31 日」混在同一
    套年份候选里取最大值，会误把报告年度判成签署年。列级表头解析与财报片段优先可规避该问题。
    """
    found: set[int] = set()
    n = 0
    for el in items:
        if not isinstance(el, dict) or el.get("type") not in {"table", "excel_table"}:
            continue
        tt = str(el.get("table_type") or "")
        if map_to_statement_type(tt, tt) not in {"BS", "IS", "CF"}:
            continue
        n += 1
        if n > max_tables:
            break
        rows = el.get("rows")
        if not isinstance(rows, list) or not rows:
            continue
        col_y = infer_column_index_to_year_for_table(
            rows,
            context_above="",
            context_below="",
            report_name=tt,
        )
        for y in col_y.values():
            if isinstance(y, int) and 1990 <= y <= 2050:
                found.add(y)
    return sorted(found, reverse=True)


def _pick_single_reporting_year(
    head: str,
    column_statement_years: Sequence[int],
    fs_snippet: str = "",
    *,
    balance_sheet_column_years: Sequence[int] = (),
) -> Optional[int]:
    """单一报告年度，禁止输出跨年区间字符串。

    *fs_snippet* 为三大表 md/caption 拼接；*column_statement_years* 为表头列解析到的年份（降序）。

    若将篇首文本与表内 md 合并后只做「取最大四位年份」，审计意见/签章日中的年份（如 2024）
    会覆盖资产负债表表头的 **2022 年** 等真实报表年度；故先信显式披露用语，再信表头列，
    再信 *仅财报片段* 中的年份，最后才用篇首+片段全文兜底。

    *balance_sheet_column_years*：部分审计 PDF 中利润表/现金流量表页会带误解析的「2024年」装饰行，
    而资产负债表列报年度正确；若该序列非空，则优先以其最大值作为整份文档的报告年度锚点。
    实际传入时建议使用 ``collect_first_balance_sheet_column_years``：取文档序首张足够大的、
    能解析出列年份的 BS，避免多套 BS 合并后 max 被附录表覆盖。
    """
    head_s = head.strip()
    fs_s = fs_snippet.strip()
    blob = "\n".join(x for x in (head_s, fs_s) if x).strip()
    y = extract_explicit_filing_year_from_blob(blob)
    if y is not None:
        return y

    if balance_sheet_column_years:
        y0 = int(balance_sheet_column_years[0])
        if 1990 <= y0 <= 2050:
            return y0

    if column_statement_years:
        y0 = int(column_statement_years[0])
        if 1990 <= y0 <= 2050:
            return y0

    hy_fs = extract_years_from_text(fs_s)
    if hy_fs:
        return hy_fs[0]

    hy_all = extract_years_from_text(blob)
    if hy_all:
        return hy_all[0]
    return None


def _period_from_month_day(month: int, day: int) -> str:
    if month == 3 and day == 31:
        return "一季度报"
    if month == 6 and day == 30:
        return "半年报"
    if month == 9 and day == 30:
        return "三季度报"
    if month == 12 and day == 31:
        return "年报"
    return "月报"


def _score_date_context(snip: str) -> int:
    s = 0
    if _REPORTING_DATE_POSITIVE_RE.search(snip):
        s += 12
    if _REPORTING_DATE_NEGATIVE_RE.search(snip):
        s -= 14
    if re.search(r"年度报告|(?<!半)年报(?!摘要)|annual\s+report", snip, re.I):
        s += 4
    if re.search(r"半年度|中期报告|interim", snip, re.I):
        s += 3
    if re.search(r"第\s*一\s*季度|一季度|Q1\b", snip, re.I):
        s += 3
    if re.search(r"第\s*三\s*季度|三季度|Q3\b", snip, re.I):
        s += 3
    if re.search(r"月度|月报|monthly", snip, re.I):
        s += 2
    return s


def _extract_triple_dates(blob: str) -> list[tuple[int, int, int, int]]:
    """(year, month, day, score)"""
    out: list[tuple[int, int, int, int]] = []
    for m in _DOC_DATE_ISO_RE.finditer(blob):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (1990 <= y <= 2050 and 1 <= mo <= 12 and 1 <= d <= 31):
            continue
        snip = blob[max(0, m.start() - 80) : m.end() + 80]
        out.append((y, mo, d, _score_date_context(snip)))
    for m in _DOC_DATE_ZH_RE.finditer(blob):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (1990 <= y <= 2050 and 1 <= mo <= 12 and 1 <= d <= 31):
            continue
        snip = blob[max(0, m.start() - 80) : m.end() + 80]
        out.append((y, mo, d, _score_date_context(snip)))
    return out


def _period_from_month_range_text(blob: str) -> Optional[str]:
    for m in re.finditer(
        r"(?:\b(?:19\d{2}|20\d{2})\s*(?:年|[./-])\s*)?"
        r"1\s*月\s*(?:-|－|—|–|~|至|到)\s*(3|6|9|12)\s*月?",
        blob,
        re.I,
    ):
        end_month = int(m.group(1))
        if end_month == 3:
            return "一季度报"
        if end_month == 6:
            return "半年报"
        if end_month == 9:
            return "三季度报"
        if end_month == 12:
            return "年报"
    for m in re.finditer(
        r"(?:\b(?:19\d{2}|20\d{2})\s*(?:年|[./-])\s*)?"
        r"1\s*(?:-|－|—|–|~|至|到)\s*(3|6|9|12)\s*月",
        blob,
        re.I,
    ):
        end_month = int(m.group(1))
        if end_month == 3:
            return "一季度报"
        if end_month == 6:
            return "半年报"
        if end_month == 9:
            return "三季度报"
        if end_month == 12:
            return "年报"
    return None



def _infer_filing_period_from_statement_table_rows(
    rows: Sequence[Sequence[Any]],
    *,
    report_name: str = "",
) -> Optional[str]:
    stmt = map_to_statement_type(report_name, report_name)
    if stmt not in {"BS", "IS", "CF"}:
        return None

    col_dates = infer_column_index_to_date_for_table(rows, report_name=report_name)
    dates = sorted(
        {d for idx, d in col_dates.items() if idx > 0 and d is not None},
        reverse=True,
    )
    if not dates:
        return None

    newest = dates[0]
    if newest.month == 12 and newest.day == 31:
        return "年报"
    if newest.month == 6 and newest.day == 30:
        return "半年报"
    if newest.month == 3 and newest.day == 31:
        return "一季度报"
    if newest.month == 9 and newest.day == 30:
        return "三季度报"
    return "月报"


def _infer_filing_period_from_strong_title(blob: str) -> Optional[str]:
    blob = (blob or "").strip()
    if not blob:
        return None
    low = blob.lower()
    first_lines = "\n".join(blob.splitlines()[:30])

    if re.search(r"(?:19\d{2}|20\d{2})\s*年\s*年度报告|(?:19\d{2}|20\d{2})\s*年度报告|年度报告|(?<!半)年报(?!摘要)|annual\s+report|form\s+10-k\b", first_lines, re.I):
        return "年报"
    if re.search(r"(?:19\d{2}|20\d{2})\s*年度的.{0,20}(?:资产负债表|利润表|现金流量表|股东权益变动表|所有者权益变动表)", blob):
        return "年报"
    if re.search(r"year\s+ended\s+.*31\s*december", low):
        return "年报"
    if "12月31日止年度" in blob or "截至12月31日" in blob:
        return "年报"

    if re.search(r"(?:19\d{2}|20\d{2})\s*年\s*半年度报告|(?:19\d{2}|20\d{2})\s*半年度报告|中期报告|interim\s+report|half[\s-]year\s+report|semi[\s-]annual\s+report", first_lines, re.I):
        return "半年报"
    if re.search(r"(?:19\d{2}|20\d{2})\s*年\s*(?:第一季度报告|一季度报告)|\bQ1\b|first\s+quarter\s+report", first_lines, re.I):
        return "一季度报"
    if re.search(r"(?:19\d{2}|20\d{2})\s*年\s*(?:第三季度报告|三季度报告)|\bQ3\b|third\s+quarter\s+report", first_lines, re.I):
        return "三季度报"
    return None


def _infer_filing_period_from_blob(blob: str) -> Optional[str]:
    blob = (blob or "").strip()
    if not blob:
        return None
    low = blob.lower()

    if re.search(r"\bQ1\b|第一季度|一季度报|一季报|first\s+quarter", blob, re.I):
        return "一季度报"
    if re.search(r"\bQ3\b|第三季度|三季度报|三季报|third\s+quarter", blob, re.I):
        return "三季度报"
    if re.search(r"半年度报告|半年报|中期报告|半年度\b|half[\s-]year|semi[\s-]annual|\bH1\b", blob, re.I):
        return "半年报"
    if "月报" in blob or re.search(r"\bmonthly\b", low):
        if re.search(r"年度资产负债表|年度利润表|年度现金流量表|年度报告|年报", blob):
            pass
        else:
            return "月报"

    range_period = _period_from_month_range_text(blob)
    if range_period:
        return range_period

    strong_period = _infer_filing_period_from_strong_title(blob)
    if strong_period:
        return strong_period

    if "年度" in blob and re.search(r"报告|报表", blob):
        triples = _extract_triple_dates(blob)
        scored = [t for t in triples if t[3] >= 4]
        pool = scored if scored else triples
        if pool:
            pool.sort(key=lambda t: (t[3], t[0], t[1], t[2]))
            _y, mo, d, _ = pool[-1]
            return _period_from_month_day(mo, d)

    if re.search(r"年度资产负债表|年度利润表|年度现金流量表", blob):
        return "年报"

    triples = _extract_triple_dates(blob)
    if triples:
        triples.sort(key=lambda t: (t[3], t[0], t[1], t[2]))
        _y, mo, d, _ = triples[-1]
        return _period_from_month_day(mo, d)

    return None


def infer_single_table_filing_period(
    *,
    rows: Sequence[Sequence[Any]] = (),
    context_above: str = "",
    context_below: str = "",
    report_name: str = "",
    md: str = "",
    caption: str = "",
    fallback_doc_period: Optional[str] = None,
) -> str:
    """当前表优先的财报周期判断；文档级周期只作为兜底。"""
    table_period = _infer_filing_period_from_statement_table_rows(rows, report_name=report_name)
    if table_period:
        return table_period

    ctx_near = (context_above or "")[-100:]
    near_date = _extract_nearest_date_from_context(ctx_near)
    if near_date is not None:
        _y, mo, d = near_date
        p = _period_from_month_day(mo, d)
        if p:
            return p

    primary_blob = "\n".join(x for x in (caption, report_name, md) if x).strip()
    period = _infer_filing_period_from_strong_title(primary_blob)
    if period:
        return period

    context_near = "\n".join(x for x in ((context_above or "")[-100:], context_below) if x).strip()
    context_blob = "\n".join(x for x in (context_above, context_below) if x).strip()
    period = _infer_filing_period_from_strong_title(context_near)
    if period:
        return period
    period = _infer_filing_period_from_strong_title(context_blob)
    if period:
        return period

    period = _infer_filing_period_from_blob(primary_blob)
    if period:
        return period

    period = _infer_filing_period_from_blob(context_near)
    if period:
        return period
    period = _infer_filing_period_from_blob(context_blob)
    if period:
        return period

    if fallback_doc_period in _FILING_PERIOD_ALLOWED:
        return fallback_doc_period
    return fallback_doc_period if fallback_doc_period else "年报"


def infer_document_filing_period_cn(
    head: str,
    fs_snippet: str,
    *,
    table_period: Optional[str] = None,
) -> str:
    """与 ``fin_extract_router.FIN_EXTRACT_RULES`` 中财报周期规则一致，仅五种取值。"""
    if table_period in _FILING_PERIOD_ALLOWED:
        return table_period
    blob = "\n".join(x for x in (head, fs_snippet) if x)
    period = _infer_filing_period_from_strong_title(blob)
    if period:
        return period
    period = _infer_filing_period_from_blob(blob)
    return period or "年报"


def infer_document_years_and_period(items: Sequence[Any]) -> tuple[Optional[int], str]:
    """整份 converted items 统一的报告年度与财报周期（每张表重复输出相同值）。"""
    head = collect_document_head_text(items)
    fs_snip = collect_financial_table_text_snippets(items)
    col_years = collect_column_years_from_statement_tables(items)
    bs_col_years = collect_first_balance_sheet_column_years(items)
    year = _pick_single_reporting_year(
        head, col_years, fs_snip, balance_sheet_column_years=bs_col_years
    )

    table_period: Optional[str] = None
    for el in items:
        if not isinstance(el, dict) or el.get("type") not in {"table", "excel_table"}:
            continue
        tt = str(el.get("table_type") or "")
        rows = el.get("rows") or []
        if not isinstance(rows, list) or not rows:
            continue
        tp = _infer_filing_period_from_statement_table_rows(rows, report_name=tt)
        if tp in _FILING_PERIOD_ALLOWED:
            table_period = tp
            break

    period = infer_document_filing_period_cn(head, fs_snip, table_period=table_period)
    if period not in _FILING_PERIOD_ALLOWED:
        period = "年报"
    return year, period
