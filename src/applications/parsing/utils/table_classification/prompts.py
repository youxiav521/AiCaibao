from __future__ import annotations

from typing import Dict, List, Optional

from src.applications.parsing.utils.table_classification.common import (
    ALLOWED_TABLE_TYPE_LABELS,
    LOCALE_EN_US,
    LOCALE_ZH_CN,
    normalize_locale,
)


def build_classification_prompt(
    table_md: str,
    table_label: str,
    table_context: str = "",
    locale: Optional[str] = None,
) -> List[Dict[str, str]]:
    table_md_trimmed = (table_md or "").strip()
    if len(table_md_trimmed) > 6000:
        table_md_trimmed = table_md_trimmed[:6000] + "\n..."

    _locale = normalize_locale(locale)

    if _locale == LOCALE_EN_US:
        # en_us 始终用英文 Prompt
        content = _build_english_prompt(table_md_trimmed, table_label, table_context)
        system_content = "You are a professional financial statement table classifier."
    else:
        # zh_cn 或未提供 locale -> 始终用中文 Prompt
        content = _build_chinese_prompt(table_md_trimmed, table_label, table_context)
        system_content = "你是专业的财务报表表格分类助手。"

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": content},
    ]


def _build_chinese_prompt(table_md_trimmed: str, table_label: str, table_context: str = "") -> str:
    return (
        "你是一个财务表格类型分类助手。现在给你一个表格的 Markdown 内容，以及该表格上方的内容（如有）。\n"
        "请判断该表格属于哪一类，仅从以下标签中选择一个并严格只输出该标签本身：\n"
        f"{', '.join(ALLOWED_TABLE_TYPE_LABELS)}\n\n"
        "判定原则：\n"
        "**以上方内容（标题/caption）为主判断表格类型，只有当上方内容缺失或无法判断时，才用表格内容来辅助分析。**\n\n"
        "严格判定规则：\n"
        "1) 首先根据上方内容中的标题判断表格类型：\n"
        "   1a) 上方内容明确包含'合并资产负债表''母公司资产负债表'等完整三大表标题时，直接按标题分类；\n"
        "   1b) 上方内容仅为编号+短标题（如'5. 现金流''3. 费用'）时，不是三大表标题，应判为'其他表'；\n"
        "   1c) 上方内容为附注编号（如'59. 未分配利润''（1）现金流量表补充资料'）时，是附注明细表，应判为'其他表'；\n"
        "   1d) 上方内容为'续表'时，无法判断类型，需结合表格内容分析；\n"
        "   1e) 上方内容含'同比增减''比重增减''重大变动说明'等列名时，是概要表，应判为'其他表'；\n"
        "2) 仅当上方内容缺失或无法判断时，才用表格内容辅助分析：\n"
        "   2a) 表格内容具有对应主表的完整结构特征（如资产负债表需包含资产/负债/权益多类科目、利润表需包含收入/成本/费用等多类项目、现金流量表需包含经营/投资/筹资活动等）；\n"
        "   2b) 若表格只包含某一类科目的明细拆分（如仅货币资金各币种明细、仅利息收支明细），判为'其他表'；\n"
        "3) 合并/母公司前缀的判断也优先依据上方内容中的标题；\n"
        "4) 如存在任何不确定，一律输出'其他表'。\n\n"
        f"表格上方内容:\n{table_context or '无'}\n"
        "表格Markdown如下：\n"
        f"{table_md_trimmed}\n\n"
        "请只输出一个标签，不要输出任何解释、前后缀或标点。"
    )


def _build_english_prompt(table_md_trimmed: str, table_label: str, table_context: str = "") -> str:
    return (
        "You are given a table in Markdown and the content above it, if any.\n"
        "Classify the table into exactly one of the following internal labels and output only that label:\n"
        f"{', '.join(ALLOWED_TABLE_TYPE_LABELS)}\n\n"
        "Decision principle:\n"
        "**Prioritize the content above (title/caption) to determine the table type. Only use table content to assist analysis when the content above is missing or inconclusive.**\n\n"
        "Decision rules:\n"
        "1) First, judge the table type based on the content above:\n"
        "   1a) If the content above clearly contains a full statement title (e.g. 'Consolidated Balance Sheet', 'Parent Company Income Statement'), classify accordingly;\n"
        "   1b) If the content above is a numbered short heading (e.g. '5. Cash Flow', '3. Expenses'), it is NOT a statement title, output '其他表';\n"
        "   1c) If the content above is a note number (e.g. '59. Retained Earnings', '(1) Supplementary Cash Flow Information'), it is a note detail table, output '其他表';\n"
        "   1d) If the content above is 'Continued' or '续表', the type cannot be determined from the title, analyze using table content;\n"
        "   1e) If the content above contains column names like 'YoY Change', 'Percentage Change', it is a summary table, output '其他表';\n"
        "2) Only when the content above is missing or inconclusive, use table content to assist:\n"
        "   2a) The table must have complete structural characteristics of that main statement (balance sheet must contain multiple categories like assets/liabilities/equity; income statement must contain revenue/cost/expense items; cash flow must contain operating/investing/financing sections);\n"
        "   2b) If the table only contains breakdown of a single category (e.g. currency breakdown, interest rate details), output '其他表';\n"
        "3) For English annual reports, map statement of financial position/balance sheet to 资产负债表, statement of profit or loss/income statement to 利润表, and statement of cash flows to 现金流量表.\n"
        "   3a) 'Statement of comprehensive income' is treated as part of the profit-or-loss family and classified as 利润表 when the content shows a P&L-style sequence.\n"
        "4) Use 合并 when the content above/table says consolidated; use 母公司 only for standalone/parent-company statements. Prefix determination also prioritizes the content above.\n"
        "5) If uncertain or ambiguous, output '其他表'. Do not guess.\n\n"
        f"Content above table:\n{table_context or 'N/A'}\n"
        "Table Markdown:\n"
        f"{table_md_trimmed}\n\n"
        "Output exactly one label only. Do not include explanation, JSON, prefix, suffix, or punctuation."
    )
