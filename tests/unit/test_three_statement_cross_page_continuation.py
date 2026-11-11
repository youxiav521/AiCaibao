"""跨页财报续表应合并为同一个三大表输出。"""

from __future__ import annotations

import pytest

from src.applications.parsing.service.three_statements_frontend_service import (
    _merge_forward_statement_continuations,
    extract_three_statements_frontend,
)
from src.applications.parsing.service.three_statements.balance_sheet_merge import (
    align_continuation_rows_to_primary_layout,
)
from src.applications.parsing.service.three_statements.notes_context import (
    truncate_cashflow_supplementary,
)
from src.applications.parsing.utils.table_classification_utils import (
    classify_tables_async,
)


def _row(*texts: str) -> list[dict[str, object]]:
    return [{"text": text, "rowSpan": 1, "colSpan": 1} for text in texts]


def _table(element_id: str, page: int, table_type: str, *rows: list[dict[str, object]]) -> dict:
    return {
        "type": "table",
        "element_id": element_id,
        "page": page,
        "page_idx": page,
        "table_type": table_type,
        "md": "\n".join(" | ".join(str(cell["text"]) for cell in row) for row in rows),
        "rows": list(rows),
        "bBox": {"l": 0, "t": 0, "r": 100, "b": 100},
    }


def test_extract_merges_cross_page_statement_continuations() -> None:
    items = [
        _table(
            "asset-1",
            1,
            "合并资产负债表",
            _row("资产", "2024年", "2023年"),
            _row("货币资金", "100", "90"),
        ),
        {"type": "text", "page_idx": 1, "text": "后附财务报表附注为本财务报表的组成部分"},
        _table(
            "asset-2",
            2,
            "资产负债表",
            _row("负债", "2024年", "2023年"),
            _row("短期借款", "20", "10"),
        ),
        _table(
            "asset-3",
            3,
            "合并资产负债表",
            _row("股东权益", "2024年", "2023年"),
            _row("股本", "80", "80"),
        ),
        _table(
            "profit-1",
            4,
            "合并利润表",
            _row("营业收入", "200", "180"),
        ),
    ]

    result = extract_three_statements_frontend(items, locale=None)
    balance_sheets = [
        table
        for table in result
        if table.get("category") == "资产负债表" and table.get("caliberLabel") == "合并"
    ]

    assert len(balance_sheets) == 1
    balance_sheet = balance_sheets[0]
    assert balance_sheet["mergeIds"] == ["asset-1", "asset-2", "asset-3"]
    row_texts = [
        str(cell.get("text") or "")
        for row in balance_sheet["rows"]
        for cell in row
        if isinstance(cell, dict)
    ]
    assert {"资产", "负债", "股东权益", "货币资金", "短期借款", "股本"}.issubset(row_texts)


def test_extract_uses_balance_sheet_equation_to_override_wrong_continuation_scope() -> None:
    assets = _table(
        "parent-assets",
        10,
        "母公司资产负债表",
        _row("资产", "2025年12月31日", "2024年12月31日"),
        _row("货币资金", "16,513,550", "30,794,989"),
        _row("资产总计", "164,831,252", "150,143,573"),
    )
    liabilities = _table(
        "misclassified-liabilities",
        11,
        "合并资产负债表",
        _row("负债和股东权益", "2025年12月31日", "2024年12月31日"),
        _row("应付账款", "147,961", "2,024,742"),
        _row("负债和股东权益总计", "164,831,252", "150,143,573"),
    )
    items = [
        {"type": "text", "page_idx": 10, "text": "5、公司资产负债表"},
        assets,
        {"type": "text", "page_idx": 10, "text": "后附财务报表附注为本财务报表的组成部分"},
        liabilities,
    ]

    result = extract_three_statements_frontend(items, locale=None)
    balance_sheets = [
        table for table in result if table.get("category") == "资产负债表"
    ]

    assert len(balance_sheets) == 1
    assert balance_sheets[0]["caliberLabel"] == "单体"
    assert balance_sheets[0]["mergeIds"] == [
        "parent-assets",
        "misclassified-liabilities",
    ]
    assert [row[0]["text"] for row in balance_sheets[0]["rows"]] == [
        "资产",
        "货币资金",
        "资产总计",
        "负债和股东权益",
        "应付账款",
        "负债和股东权益总计",
    ]


def test_extract_does_not_override_scope_when_balance_sheet_totals_differ() -> None:
    assets = _table(
        "parent-assets",
        10,
        "母公司资产负债表",
        _row("资产", "2025年12月31日", "2024年12月31日"),
        _row("资产总计", "164,831,252", "150,143,573"),
    )
    liabilities = _table(
        "consolidated-liabilities",
        11,
        "合并资产负债表",
        _row("负债和股东权益", "2025年12月31日", "2024年12月31日"),
        _row("负债和股东权益总计", "883,729,883", "783,355,855"),
    )

    result = extract_three_statements_frontend([assets, liabilities], locale=None)
    balance_sheets = [
        table for table in result if table.get("category") == "资产负债表"
    ]

    assert len(balance_sheets) == 2
    assert {tuple(table["mergeIds"]) for table in balance_sheets} == {
        ("parent-assets",),
        ("consolidated-liabilities",),
    }


def test_cashflow_year_end_balance_is_not_treated_as_supplementary_data() -> None:
    rows = [
        _row("五、现金及现金等价物净减少额", "(10)", "(8)"),
        _row("加：年初现金及现金等价物余额", "100", "90"),
        _row("六、年末现金及现金等价物余额", "90", "82"),
    ]

    assert truncate_cashflow_supplementary(rows) == rows


def test_cashflow_real_supplementary_section_is_still_truncated() -> None:
    rows = [
        _row("六、年末现金及现金等价物余额", "90", "82"),
        _row("补充资料", "", ""),
        _row("将净利润调节为经营活动现金流量", "20", "18"),
    ]

    assert truncate_cashflow_supplementary(rows) == rows[:1]


def test_extract_keeps_same_page_statement_tables_separate() -> None:
    first = _table(
        "asset-left",
        1,
        "合并资产负债表",
        _row("资产", "2024年"),
        _row("货币资金", "100"),
    )
    second = _table(
        "asset-right",
        1,
        "合并资产负债表",
        _row("负债", "2024年"),
        _row("短期借款", "20"),
    )
    second["bBox"] = {"l": 120, "t": 0, "r": 220, "b": 100}

    candidate = {
        "label": "合并资产负债表",
        "table": first,
        "rows_count": len(first["rows"]),
        "is_full": False,
        "expanded_idx": 0,
        "page": 1,
    }
    merged, continuation_indices = _merge_forward_statement_continuations(
        candidate,
        [first, second],
        locale=None,
        excluded_indices=set(),
    )

    assert continuation_indices == []
    assert merged["table"]["rows"] == first["rows"]


def test_extract_merges_same_page_vertical_cashflow_fragments_with_wrong_scope() -> None:
    first = _table(
        "cash-parent-1",
        155,
        "母公司现金流量表",
        _row("销售商品、提供劳务收到的现金", "100", "90"),
        _row("经营活动产生的现金流量净额", "20", "10"),
    )
    first["bBox"] = {"l": 99, "t": 144, "r": 528, "b": 311}
    continuation = _table(
        "cash-parent-2",
        155,
        "合并现金流量表",
        _row("取得投资收益收到的现金", "8", "7"),
        _row("期末现金及现金等价物余额", "28", "17"),
    )
    continuation["bBox"] = {"l": 99, "t": 345, "r": 529, "b": 536}
    items = [
        {"type": "text", "page_idx": 155, "text": "8、公司现金流量表"},
        first,
        {"type": "text", "page_idx": 155, "text": "投资活动产生的现金流量："},
        continuation,
    ]

    result = extract_three_statements_frontend(items, locale=None)
    cashflows = [table for table in result if table.get("category") == "现金流量表"]

    assert len(cashflows) == 1
    assert cashflows[0]["caliberLabel"] == "单体"
    assert cashflows[0]["mergeIds"] == ["cash-parent-1", "cash-parent-2"]
    assert [row[0]["text"] for row in cashflows[0]["rows"]] == [
        "销售商品、提供劳务收到的现金",
        "经营活动产生的现金流量净额",
        "取得投资收益收到的现金",
        "期末现金及现金等价物余额",
    ]


def test_extract_keeps_repeated_full_statement_tables_separate() -> None:
    first = _table(
        "cash-2022",
        7,
        "现金流量表",
        _row("项目", "期末余额", "期初余额"),
        _row("一、经营活动产生的现金流量", "", ""),
        _row("六、期末现金及现金等价物余额", "100", "90"),
    )
    next_year = _table(
        "cash-2023",
        9,
        "现金流量表",
        _row("项目", "期末余额", "期初余额"),
        _row("一、经营活动产生的现金流量", "", ""),
        _row("六、期末现金及现金等价物余额", "120", "100"),
    )
    candidate = {
        "label": "现金流量表",
        "table": first,
        "rows_count": len(first["rows"]),
        "is_full": True,
        "expanded_idx": 0,
        "page": 7,
    }

    merged, continuation_indices = _merge_forward_statement_continuations(
        candidate,
        [first, next_year],
        locale=None,
        excluded_indices=set(),
    )

    assert continuation_indices == []
    assert merged["table"]["rows"] == first["rows"]


def test_extract_merges_numbered_unclassified_profit_continuation() -> None:
    first = _table(
        "profit-1",
        1,
        "利润表",
        _row("项目", "期末余额", "期初余额"),
        _row("三、其他综合收益的税后净额", "10", "9"),
        _row("3.金融资产重分类计入其他综合收益的金额", "", ""),
    )
    continuation = _table(
        "profit-2",
        2,
        "其他表",
        _row("4.其他债权投资信用减值准备", "", ""),
        _row("七、综合收益总额", "100", "90"),
        _row("（二）稀释每股收益", "0.45", "0.44"),
    )

    result = extract_three_statements_frontend([first, continuation], locale=None)

    profit = next(table for table in result if table.get("category") == "利润表")
    assert profit["mergeIds"] == ["profit-1", "profit-2"]
    assert [row[0]["text"] for row in profit["rows"][-3:]] == [
        "4.其他债权投资信用减值准备",
        "七、综合收益总额",
        "（二）稀释每股收益",
    ]


def test_extract_does_not_merge_unrelated_table_with_sequential_row_number() -> None:
    profit = _table(
        "profit",
        1,
        "利润表",
        _row("三、其他综合收益的税后净额", "10", "9"),
        _row("3.金融资产重分类计入其他综合收益的金额", "", ""),
    )
    customer_schedule = _table(
        "customer-schedule",
        2,
        "其他表",
        _row("4.客户名称", "销售额", "占比"),
        _row("甲公司", "100", "20%"),
    )

    result = extract_three_statements_frontend(
        [profit, customer_schedule],
        locale=None,
    )

    extracted_profit = next(
        table for table in result if table.get("category") == "利润表"
    )
    assert extracted_profit["mergeIds"] == ["profit"]


@pytest.mark.asyncio
async def test_classifier_repairs_short_numbered_statement_continuation() -> None:
    first = _table(
        "profit-1",
        1,
        "利润表",
        _row("三、其他综合收益的税后净额", "10", "9"),
        _row("3.金融资产重分类计入其他综合收益的金额", "", ""),
    )
    continuation = _table(
        "profit-2",
        2,
        "其他表",
        _row("4.其他债权投资信用减值准备", "", ""),
        _row("七、综合收益总额", "100", "90"),
        _row("（二）稀释每股收益", "0.45", "0.44"),
    )

    await classify_tables_async([first, continuation], locale=None)

    assert continuation["table_type"] == "利润表"
    assert continuation["table_type_evidence"] == (
        "continuation:numbered_statement_content"
    )


@pytest.mark.asyncio
async def test_classifier_repairs_wrong_type_for_numbered_statement_continuation() -> None:
    first = _table(
        "profit-1",
        1,
        "利润表",
        _row("三、其他综合收益的税后净额", "10", "9"),
        _row("3.金融资产重分类计入其他综合收益的金额", "", ""),
    )
    continuation = _table(
        "profit-2",
        2,
        "现金流量表",
        _row("4.其他债权投资信用减值准备", "", ""),
        _row("七、综合收益总额", "100", "90"),
        _row("（二）稀释每股收益", "0.45", "0.44"),
    )

    await classify_tables_async([first, continuation], locale=None)

    assert continuation["table_type"] == "利润表"
    assert continuation["table_type_evidence"] == (
        "continuation:numbered_statement_content"
    )

    result = extract_three_statements_frontend([first, continuation], locale=None)
    profit = next(table for table in result if table.get("category") == "利润表")
    assert profit["mergeIds"] == ["profit-1", "profit-2"]


@pytest.mark.asyncio
async def test_classifier_does_not_repair_unrelated_numbered_table() -> None:
    profit = _table(
        "profit",
        1,
        "利润表",
        _row("三、其他综合收益的税后净额", "10", "9"),
        _row("3.金融资产重分类计入其他综合收益的金额", "", ""),
    )
    customer_schedule = _table(
        "customer-schedule",
        2,
        "其他表",
        _row("4.客户名称", "销售额", "占比"),
        _row("甲公司", "100", "20%"),
    )

    await classify_tables_async([profit, customer_schedule], locale=None)

    assert customer_schedule["table_type"] == "其他表"
    assert "table_type_evidence" not in customer_schedule


def test_extract_aligns_continuation_without_note_column() -> None:
    primary = _table(
        "asset-1",
        1,
        "母公司资产负债表",
        _row("资产", "附注十九", "2024年12月31日", "2023年12月31日"),
        _row("货币资金", "", "100", "90"),
    )
    continuation = _table(
        "asset-2",
        2,
        "母公司资产负债表",
        _row("负债和股东权益", "2024年12月31日", "2023年12月31日"),
        _row("短期借款", "20", "10"),
    )
    for row in continuation["rows"]:
        for column, cell in enumerate(row):
            cell["colstart"] = column

    result = extract_three_statements_frontend([primary, continuation], locale=None)
    balance_sheet = next(table for table in result if table.get("mergeIds") == ["asset-1", "asset-2"])

    assert len(balance_sheet["rows"]) == 4
    assert [row[0]["text"] for row in balance_sheet["rows"]] == [
        "资产",
        "货币资金",
        "负债和股东权益",
        "短期借款",
    ]
    assert balance_sheet["rows"][-1][1]["text"] == "20"
    assert balance_sheet["rows"][-1][2]["text"] == "10"


def test_extract_repairs_overlapping_note_and_amount_columns_before_merging() -> None:
    primary = _table(
        "asset-1",
        1,
        "母公司资产负债表",
        _row("资产", "附注十九", "2024年12月31日", "2023年12月31日"),
        _row("货币资金", "", "100", "90"),
    )
    for row in primary["rows"]:
        for column, cell in enumerate(row):
            cell["colstart"] = (0, 1, 1, 2)[column]
    continuation = _table(
        "asset-2",
        2,
        "母公司资产负债表",
        _row("负债和股东权益", "2024年12月31日", "2023年12月31日"),
        _row("短期借款", "20", "10"),
    )

    result = extract_three_statements_frontend([primary, continuation], locale=None)
    balance_sheet = next(table for table in result if table.get("mergeIds") == ["asset-1", "asset-2"])

    assert balance_sheet["rows"][-1][0]["text"] == "短期借款"
    assert balance_sheet["rows"][-1][1]["text"] == "20"
    assert balance_sheet["rows"][-1][2]["text"] == "10"


def test_aligns_continuation_columns_by_date_not_position() -> None:
    primary_rows = [
        _row("Assets", "Note", "2024-12-31", "2023-12-31"),
        _row("Cash", "", "100", "90"),
    ]
    continuation_rows = [
        _row("Liabilities", "2023-12-31", "2024-12-31"),
        _row("Loans", "10", "20"),
    ]

    _, aligned_rows = align_continuation_rows_to_primary_layout(
        primary_rows, continuation_rows
    )

    assert [cell["text"] for cell in aligned_rows[1]] == ["Loans", "", "20", "10"]


def test_keeps_continuation_rows_when_date_headers_are_unavailable() -> None:
    primary_rows = [_row("Assets", "Note", "Current", "Prior")]
    continuation_rows = [_row("Liabilities", "Current", "Prior")]

    _, aligned_rows = align_continuation_rows_to_primary_layout(
        primary_rows, continuation_rows
    )

    assert aligned_rows == continuation_rows
