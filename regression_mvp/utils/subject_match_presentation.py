from __future__ import annotations

import asyncio
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field, model_validator


_REPORT_TYPES = ("BS", "IS", "CF")
_REPORT_TYPE_MARKERS = {
    "BS": (
        "资产负债",
        "financial position",
        "balance sheet",
    ),
    "IS": (
        "利润表",
        "综合收益",
        "income statement",
        "comprehensive income",
        "profit or loss",
        "statement of operations",
    ),
    "CF": (
        "现金流量",
        "cash flow",
        "cashflow",
    ),
}


class PresentationDecision(BaseModel):
    report_type: Literal["BS", "IS", "CF"]
    presentation: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class PresentationDecisionResponse(BaseModel):
    items: list[PresentationDecision] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def accept_common_shapes(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        return {
            "items": value.get("items")
            or value.get("decisions")
            or value.get("results")
            or []
        }


async def ainvoke_llm_with_structured_output(*args, **kwargs):
    from src.utils.llm_invoke import (
        ainvoke_llm_with_structured_output as invoke_structured_output,
    )

    return await invoke_structured_output(*args, **kwargs)


def infer_report_presentations(
    report_list: list[Any],
    statement_tables: list[Any],
    *,
    model_id: str,
    timeout_s: float | None,
) -> dict[str, str]:
    report_types = infer_statement_report_types(statement_tables)
    if not report_types:
        raise ValueError("无法从英文财报表格识别 BS、IS 或 CF 报表类型")

    options = build_presentation_options(report_list, report_types)
    missing_options = sorted(report_types - set(options))
    if missing_options:
        raise ValueError(
            "英文科目模板缺少列报方式候选: {}".format(",".join(missing_options))
        )

    resolved: dict[str, PresentationDecision] = {}
    ambiguous: dict[str, list[tuple[str, str]]] = {}
    for report_type, report_options in options.items():
        if len(report_options) == 1:
            presentation, root_name = report_options[0]
            resolved[report_type] = PresentationDecision(
                report_type=report_type,
                presentation=presentation,
                confidence=1.0,
                reason="模板仅有一个可选树: {}".format(root_name),
            )
        elif report_options:
            ambiguous[report_type] = report_options

    if ambiguous:
        try:
            response = asyncio.run(
                ainvoke_llm_with_structured_output(
                    prompt=build_presentation_decision_prompt(
                        ambiguous,
                        statement_tables,
                    ),
                    model_id=model_id,
                    response_type=PresentationDecisionResponse,
                    timeout_s=timeout_s,
                    temperature=0,
                )
            )
        except Exception as exc:
            logger.exception(
                "英文财报列报方式判断失败 | report_types={}",
                sorted(ambiguous),
            )
            raise RuntimeError(
                "英文财报列报方式判断失败，已停止科目匹配，未使用全量模板"
            ) from exc

        for decision in response.items:
            allowed = {value for value, _ in ambiguous.get(decision.report_type, [])}
            if decision.presentation not in allowed:
                raise ValueError(
                    "列报方式判断返回模板外值 | report_type={}, presentation={}, allowed={}".format(
                        decision.report_type,
                        decision.presentation,
                        sorted(allowed),
                    )
                )
            existing = resolved.get(decision.report_type)
            if existing and existing.presentation != decision.presentation:
                raise ValueError(
                    "列报方式判断冲突 | report_type={}, presentations={},{}".format(
                        decision.report_type,
                        existing.presentation,
                        decision.presentation,
                    )
                )
            if existing:
                logger.warning(
                    "列报方式判断返回重复同值，已合并 | report_type={}, presentation={}",
                    decision.report_type,
                    decision.presentation,
                )
                if decision.confidence > existing.confidence:
                    resolved[decision.report_type] = decision
                continue
            resolved[decision.report_type] = decision

    missing_decisions = sorted(report_types - set(resolved))
    if missing_decisions:
        raise ValueError(
            "大模型未返回报表列报方式判断: {}".format(",".join(missing_decisions))
        )

    logger.info(
        "英文财报列报方式判断完成 | decisions={}",
        {
            report_type: {
                "presentation": decision.presentation,
                "confidence": decision.confidence,
                "reason": decision.reason,
            }
            for report_type, decision in resolved.items()
        },
    )
    return {
        report_type: decision.presentation
        for report_type, decision in resolved.items()
    }


def infer_statement_report_types(statement_tables: list[Any]) -> set[str]:
    report_types: set[str] = set()
    for table in statement_tables:
        report_type = _table_report_type(table)
        if report_type:
            report_types.add(report_type)
    return report_types


def build_presentation_options(
    report_list: list[Any],
    report_types: set[str],
) -> dict[str, list[tuple[str, str]]]:
    options: dict[str, list[tuple[str, str]]] = {}
    for report_type in sorted(report_types):
        report = next(
            (
                item
                for item in report_list
                if isinstance(item, dict)
                and str(item.get("reportType") or "").strip().upper() == report_type
            ),
            None,
        )
        if report is None or not isinstance(report.get("subjectList"), list):
            continue

        roots_by_presentation: dict[str, tuple[int, str]] = {}
        for subject in report["subjectList"]:
            if not isinstance(subject, dict):
                continue
            presentation = str(subject.get("presentation") or "").strip()
            if not presentation:
                raise ValueError(
                    "英文科目模板缺少 presentation | report_type={}, subject_id={}".format(
                        report_type,
                        subject.get("subjectId"),
                    )
                )
            name = str(subject.get("subjectName") or presentation).strip()
            level = _safe_int(subject.get("subjectLevel"), default=999)
            current = roots_by_presentation.get(presentation)
            if current is None or level < current[0]:
                roots_by_presentation[presentation] = (level, name)

        options[report_type] = [
            (presentation, roots_by_presentation[presentation][1])
            for presentation in sorted(roots_by_presentation)
        ]
    return options


def build_presentation_decision_prompt(
    options: dict[str, list[tuple[str, str]]],
    statement_tables: list[Any],
) -> str:
    option_text = "\n".join(
        "{}: {}".format(
            report_type,
            "；".join(
                "{} = {}".format(value, root_name)
                for value, root_name in report_options
            ),
        )
        for report_type, report_options in options.items()
    )
    table_preview = _statement_table_preview(statement_tables, set(options))
    return """你是资深 IFRS 财务报表专家。请判断当前财报中每一种报表采用哪一棵标准列报树。
只能从对应报表类型给出的 presentation 值中选择，不允许自造值。

判断重点：
- BS：区分 current/non-current 与 liquidity order。
- IS：区分 by function of expense 与 by nature of expense。
- CF：区分 direct、indirect、indirect extended financial institution。
- 根据整张报表的标题、分组结构和主要科目综合判断，不要只看单行关键词。
- 即使同一报表被拆成多个表格块，每种报表类型也只能返回一个决定；没有出现在候选列表中的报表类型不要返回。

可选树：
{option_text}

原始表格预览：
{table_preview}

严格输出 JSON：
{{
  "items": [
    {{
      "report_type": "BS | IS | CF",
      "presentation": "必须是上面给出的值",
      "confidence": 0.0,
      "reason": "简短说明报表结构证据"
    }}
  ]
}}
""".format(option_text=option_text, table_preview=table_preview)


def _statement_table_preview(
    statement_tables: list[Any],
    report_types: set[str],
    *,
    max_chars: int = 12000,
) -> str:
    parts: list[str] = []
    for table in statement_tables:
        if _table_report_type(table) not in report_types or not isinstance(table, dict):
            continue
        title = _first_text(
            table.get("reportName"),
            table.get("text"),
            table.get("category"),
        )
        rows = table.get("rows")
        lines = ["报表标题: {}".format(title)]
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, list):
                    continue
                texts = [
                    _cell_text(cell)
                    for cell in row
                    if _cell_text(cell)
                ]
                if texts:
                    lines.append("\t".join(texts))
        parts.append("\n".join(lines))
    return "\n\n".join(parts)[:max_chars]


def _table_report_type(table: Any) -> str | None:
    if not isinstance(table, dict):
        return None
    explicit = str(table.get("reportType") or "").strip().upper()
    if explicit in _REPORT_TYPES:
        return explicit

    identifying_text = " ".join(
        str(value or "").strip().lower()
        for value in (
            table.get("reportName"),
            table.get("text"),
            table.get("category"),
            table.get("reportId"),
        )
    )
    for report_type in _REPORT_TYPES:
        if any(marker in identifying_text for marker in _REPORT_TYPE_MARKERS[report_type]):
            return report_type
    return None


def _cell_text(cell: Any) -> str:
    if isinstance(cell, dict):
        return str(cell.get("text") or "").strip()
    return str(cell or "").strip()


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
