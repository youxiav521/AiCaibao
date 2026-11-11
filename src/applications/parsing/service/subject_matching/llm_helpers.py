from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from ...utils.semantic_matcher import normalize_accounting_terms, repair_truncation
from .en.prompts import (
    build_normalize_batch_prompt as build_en_normalize_batch_prompt,
    build_review_matches_prompt as build_en_review_matches_prompt,
)
from .strategy import FinancialStatementLanguage
from .text_utils import normalize_subject_text


def _clean_json_content(content: str) -> str:
    content = content.strip()
    if "```json" in content:
        content = content[content.find("```json") + 7:]
        content = content[:content.find("```")].strip()
    elif "```" in content:
        content = content[content.find("```") + 3:]
        content = content[:content.find("```")].strip()
    content = re.sub(r",\s*([}\]])", r"\1", content)
    content = re.sub(r"//[^\n]*", "", content)
    return re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)


async def llm_review_matches(
    text_to_first_cells: dict[str, list],
    subject_id_to_name: dict[str, str],
    report_name: str,
    stmt_type: Optional[str],
    matcher: Any,
    language: FinancialStatementLanguage = FinancialStatementLanguage.ZH,
    trusted_rule_matches: dict[str, str] | None = None,
) -> None:
    matched_pairs: list[tuple[str, str, str]] = []
    for txt, cells in text_to_first_cells.items():
        sid = cells[0].subjectId if cells else None
        if sid:
            if (trusted_rule_matches or {}).get(txt) == sid:
                continue
            tname = subject_id_to_name.get(sid, sid)
            matched_pairs.append((txt, sid, tname))

    if not matched_pairs:
        return

    pair_lines = []
    for i, (src, _sid, tname) in enumerate(matched_pairs, 1):
        if src == tname:
            pair_lines.append(f"{i}. {src!r} → {tname!r} (完全一致)")
        else:
            pair_lines.append(f"{i}. {src!r} → {tname!r}")
    pairs_str = "\n".join(pair_lines)

    stmt_label = {
        "balance": "资产负债表",
        "income": "利润表",
        "cashflow": "现金流量表",
    }.get(stmt_type or "", report_name)

    if language == FinancialStatementLanguage.EN:
        prompt = build_en_review_matches_prompt(
            report_name=report_name,
            statement_label=stmt_label,
            pairs_text=pairs_str,
        )
    else:
        prompt = f"""你是一位精通中国会计准则（CAS）的财务审计专家。

报表类型：{stmt_label}

以下是系统自动匹配的结果（左侧为表格中识别的科目文本，右侧为匹配到的标准模板科目）：
{pairs_str}

请逐条审核上述匹配，找出**明显错误**的匹配。判断标准：
1. 会计含义完全不同（如"特准储备基金"≠"公益金"，"补贴收入"≠"营业外收入"）
2. 方向/期限/分类互斥（应收↔应付，短期↔长期，流动↔非流动，经营↔投资↔筹资活动）
3. 层级关系混淆（父科目匹配到子科目，或反之）
4. 跨报表科目误匹配

注意：
- **仅否决明显错误，不要否决合理匹配**
- 以下情况是合理的，不应否决：标记为"(完全一致)"的、新旧准则同一概念（如"债券投资"≈"债权投资"）、公认同义词/简称
- 如果所有匹配都合理，返回空列表

请以JSON格式返回：{{"reject": [编号列表], "reasons": {{"编号": "原因"}}}}
只返回JSON，不要返回其他内容。"""

    try:
        if not matcher or not matcher.model or not hasattr(matcher.model, "ainvoke"):
            return

        response = await matcher.model.ainvoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        result = json.loads(_clean_json_content(content))
        reject_indices = result.get("reject", [])
        reasons = result.get("reasons", {})

        if not reject_indices:
            logger.info("[LLM review] All {} matches approved for {}", len(matched_pairs), stmt_label)
            return

        rejected_count = 0
        for idx in reject_indices:
            idx_int = int(idx)
            if idx_int < 1 or idx_int > len(matched_pairs):
                continue
            src_text, _sid, tname = matched_pairs[idx_int - 1]
            reason = reasons.get(str(idx), reasons.get(idx, ""))
            cells = text_to_first_cells.get(src_text, [])
            for cell in cells:
                cell.subjectId = None
            rejected_count += 1
            logger.warning(
                "[LLM review] Rejected: {!r} -> {!r}, reason: {}",
                src_text, tname, reason,
            )

        logger.info(
            "[LLM review] {}/{} matches rejected for {}",
            rejected_count, len(matched_pairs), stmt_label,
        )

    except Exception as e:
        logger.warning("[LLM review] Failed (non-fatal), skipping: {}", str(e)[:200])


async def llm_normalize_batch(
    texts: list[str],
    report_name: str,
    matcher: Any,
    language: FinancialStatementLanguage = FinancialStatementLanguage.ZH,
) -> dict[str, str]:
    if not texts:
        return {}

    if not matcher or not matcher.model or not hasattr(matcher.model, "ainvoke"):
        return {}

    text_list = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))

    if language == FinancialStatementLanguage.EN:
        prompt = build_en_normalize_batch_prompt(report_name=report_name, text_list=text_list)
    else:
        prompt = f"""你是中国会计准则（CAS）专家。以下科目名称来自「{report_name}」，可能使用了旧准则术语、行业简称或非标准表述。

科目列表：
{text_list}

请逐条判断：
- 如果该科目使用了旧准则术语或非标准表述，且存在对应的现代标准名称，返回标准名称
- 如果已经是标准/通用名称，或者无法确定，请省略该条

常见映射举例（不限于此）：
- 应收股息 ↔ 应收股利
- 未付利润 → 应付股利
- 应交税金 / 未交税金 → 应交税费
- 递延税款借项 → 递延所得税资产
- 递延税款贷项 → 递延所得税负债
- 应付工资 → 应付职工薪酬
- 债券投资 ↔ 债权投资

注意：
- 只做术语标准化，不要改变科目的会计含义
- 仅返回需要映射的条目，无需映射的请省略

请以JSON格式返回：{{"mappings": [{{"idx": 编号, "standard": "标准名称"}}]}}
只返回JSON，不要返回其他内容。"""

    try:
        response = await matcher.model.ainvoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        result = json.loads(_clean_json_content(content))
        mappings = result.get("mappings", [])

        norm_map: dict[str, str] = {}
        for item in mappings:
            idx = int(item.get("idx", 0))
            standard = item.get("standard")
            if 1 <= idx <= len(texts) and standard and isinstance(standard, str):
                original = texts[idx - 1]
                std = standard.strip()
                if std != original:
                    norm_map[original] = std

        if norm_map:
            logger.info(
                "[LLM normalize] {}/{} terms standardized for {}",
                len(norm_map), len(texts), report_name,
            )
            for k, v in norm_map.items():
                logger.debug("[LLM normalize]   {!r} -> {!r}", k, v)

        return norm_map

    except Exception as e:
        logger.warning("[LLM normalize] Failed (non-fatal): {}", str(e)[:200])
        return {}


def post_llm_similarity(source_text: str, matched_name: str) -> float:
    def _clean(t: str) -> str:
        t = re.sub(r"[\s:：，。、；！？（）【】\(\)\-—]", "", t)
        t = normalize_subject_text(t)
        t = repair_truncation(t)
        t = normalize_accounting_terms(t)
        return t

    a, b = _clean(source_text), _clean(matched_name)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def filter_candidates_by_similarity(
    text: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 50,
) -> Tuple[List[Dict[str, Any]], float]:
    text_clean = re.sub(r"[0-9\s\(\)（）：:、，。\-—]", "", text)
    if not text_clean:
        return candidates[:top_k], 0.0

    chars_text = set(text_clean)
    scored: list[tuple[float, dict]] = []

    for candidate in candidates:
        name = candidate.get("name", "")
        name_clean = re.sub(r"[0-9\s\(\)（）：:、，。\-—]", "", name)
        if not name_clean:
            scored.append((0.0, candidate))
            continue

        seq_score = SequenceMatcher(None, text_clean, name_clean).ratio()
        chars_name = set(name_clean)
        union_len = len(chars_text | chars_name)
        jaccard = len(chars_text & chars_name) / union_len if union_len > 0 else 0.0
        contain_bonus = 0.3 if (text_clean in name_clean or name_clean in text_clean) else 0.0
        combined = 0.5 * seq_score + 0.3 * jaccard + 0.2 * contain_bonus
        scored.append((combined, candidate))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_score = scored[0][0] if scored else 0.0
    return [candidate for _, candidate in scored[:top_k]], top_score
