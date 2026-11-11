"""财报识别与财报事实抽取（规则层）单元测试。"""

from __future__ import annotations

import pytest

from src.applications.parsing.utils.financial_recognition_and_extraction_utils import (
    augment_column_years_for_comparative_period_labels,
    detect_gaap_type,
    extract_explicit_filing_year_from_blob,
    extract_financial_facts_from_grid,
    infer_accounting_period,
    infer_document_years_and_period,
    infer_column_index_to_caliber_for_table,
    infer_column_index_to_gaap_for_table,
    infer_column_index_to_statement_type_for_table,
    infer_column_index_to_year_for_table,
    infer_single_table_reporting_year,
    interpolate_balance_sheet_reporting_years_by_peer_pages,
    map_to_statement_type,
    match_caliber_label,
    _strict_reporting_year_for_peer_gap_table,
)


def test_detect_gaap_ifrs_and_cas() -> None:
    assert detect_gaap_type("Prepared in accordance with IFRS") == "IFRS"
    assert detect_gaap_type("本公司执行企业会计准则") == "CAS"
    assert detect_gaap_type("random text only") is None


def test_detect_gaap_heuristic_cas_from_table_body() -> None:
    blob = "合并资产负债表\n货币资金\t应收账款\n所有者权益合计\n人民币万元"
    assert detect_gaap_type(blob) == "CAS"


def test_detect_gaap_heuristic_ifrs_from_table_body() -> None:
    blob = (
        "EQUITY AND LIABILITIES\nShare capital\nRetained earnings\n"
        "Non-controlling interests\nTrade payables\nLease liabilities"
    )
    assert detect_gaap_type(blob) == "IFRS"


def test_detect_gaap_mfrs_explicit() -> None:
    assert detect_gaap_type("The financial statements comply with MFRS") == "IFRS"


def test_map_statement_type() -> None:
    assert map_to_statement_type("合并资产负债表", "合并资产负债表") == "BS"
    assert map_to_statement_type("合并利润表", "合并利润表") == "IS"
    assert map_to_statement_type("合并现金流量表", "合并现金流量表") == "CF"


def test_extract_financial_facts_from_grid_header_year() -> None:
    rows = [
        [
            {"text": "项目"},
            {"text": "2023年12月31日", "pageAt": 2, "bBox": {"l": 0, "t": 0, "r": 1, "b": 1}},
        ],
        [
            {"text": "货币资金", "pageAt": 2, "bBox": {"l": 0, "t": 0, "r": 1, "b": 1}},
            {"text": "1,234.56", "pageAt": 2, "bBox": {"l": 10, "t": 0, "r": 20, "b": 1}},
        ],
    ]
    facts = extract_financial_facts_from_grid(
        rows,
        context_above="合并资产负债表",
        context_below="",
        report_name="合并资产负债表",
        fallback_page_at=2,
    )
    assert len(facts) == 1
    assert facts[0]["subject_name"] == "货币资金"
    assert facts[0]["accounting_year"] == 2023
    assert facts[0]["caliber"] == "合并"
    assert facts[0]["value_text"] == "1,234.56"
    assert facts[0]["bBox"]["l"] == 10.0


def test_cell_level_period_statement_gaap_caliber() -> None:
    cell_text = "EQUITY AND LIABILITIES"
    table_name = "Consolidated Statement of Financial Position"
    period = infer_accounting_period(cell_text, table_name, "2024 2023")
    stmt = map_to_statement_type(cell_text, "合并资产负债表")
    gaap = detect_gaap_type(cell_text, table_name, "Share capital Retained earnings")
    assert period["years"] == [2024, 2023]
    assert stmt == "BS"
    assert gaap == "IFRS"
    assert match_caliber_label(2, "Consolidated") == "合并"


def test_infer_column_year_and_caliber_label() -> None:
    rows = [
        [{"text": "项目"}, {"text": "2022年度"}, {"text": "2023年度"}],
        [{"text": "营收"}, {"text": "1"}, {"text": "2"}],
    ]
    m = infer_column_index_to_year_for_table(rows, report_name="合并利润表")
    assert m.get(1) == 2022
    assert m.get(2) == 2023
    assert match_caliber_label(2, "利润表") == "合并"
    assert match_caliber_label(1, "母公司资产负债表") == "单体"


def test_infer_caliber_omi_style_column_headers_with_duplicate_banner() -> None:
    """横幅 ``CONSOLIDATED AND SEPARATE`` 粘在每列表头时不应把各列打成「混合」。"""
    banner = "STATEMENT OF COMPREHENSIVE INCOME CONSOLIDATED AND SEPARATE"
    rows = [
        [
            {"text": "Statement of comprehensive income - Net of tax"},
            {"text": f"Consolidated01/01/2022-31/12/2022\n{banner}"},
            {"text": f"Standalone01/01/2022-31/12/2022\n{banner}"},
            {"text": f"Consolidated01/01/2021-31/12/2021\n{banner}"},
            {"text": f"Standalone01/01/2021-31/12/2021\n{banner}"},
        ],
        [
            {"text": "Net Profit"},
            {"text": "1"},
            {"text": "2"},
            {"text": "3"},
            {"text": "4"},
        ],
    ]
    m = infer_column_index_to_caliber_for_table(
        rows,
        table_caliber="混合",
        report_name="Income Statement",
    )
    assert m.get(1) == "合并"
    assert m.get(2) == "单体"
    assert m.get(3) == "合并"
    assert m.get(4) == "单体"


def test_infer_column_level_metadata_maps() -> None:
    rows = [
        [{"text": ""}, {"text": "Group\n2024"}, {"text": "Company\n2024"}],
        [{"text": "项目"}, {"text": "1"}, {"text": "2"}],
    ]
    caliber_map = infer_column_index_to_caliber_for_table(
        rows,
        table_caliber="混合",
        report_name="Statement of financial position",
    )
    stmt_map = infer_column_index_to_statement_type_for_table(
        rows,
        report_name="合并资产负债表",
        fallback_statement_type="BS",
    )
    gaap_map = infer_column_index_to_gaap_for_table(
        rows,
        report_name="Statement of financial position",
        context_blob="EQUITY AND LIABILITIES Retained earnings",
        fallback_gaap="IFRS",
    )
    assert caliber_map.get(1) == "合并"
    assert caliber_map.get(2) == "单体"
    assert stmt_map.get(1) == "BS"
    assert stmt_map.get(2) == "BS"
    assert gaap_map.get(1) == "IFRS"
    assert gaap_map.get(2) == "IFRS"


def test_infer_column_caliber_with_colspan_headers() -> None:
    rows = [
        [
            {"text": "", "colSpan": 1},
            {"text": "Group", "colSpan": 2},
            {"text": "Company", "colSpan": 2},
        ],
        [
            {"text": "项目"},
            {"text": "2024"},
            {"text": "2023"},
            {"text": "2024"},
            {"text": "2023"},
        ],
    ]
    caliber_map = infer_column_index_to_caliber_for_table(
        rows,
        table_caliber="混合",
        report_name="Statement of financial position",
    )
    assert caliber_map.get(1) == "合并"
    assert caliber_map.get(2) == "合并"
    assert caliber_map.get(3) == "单体"
    assert caliber_map.get(4) == "单体"


def test_company_only_header_uses_table_fallback() -> None:
    rows = [
        [{"text": ""}, {"text": "Company\n2024"}, {"text": "Company\n2023"}],
        [{"text": "项目"}, {"text": "1"}, {"text": "2"}],
    ]
    caliber_map = infer_column_index_to_caliber_for_table(
        rows,
        table_caliber="合并",
        report_name="Consolidated statement",
    )
    # 仅 company 弱信号，不应强行全判单体，回退表级（合并）
    assert caliber_map.get(1) == "合并"
    assert caliber_map.get(2) == "合并"


def test_extract_org_tables_aligns_with_three_statement_shape() -> None:
    pytest.importorskip("fastapi")
    from src.applications.parsing.service.three_statements_frontend_service import (
        extract_org_tables,
    )

    items = [
        {"type": "text", "md": "Consolidated statement of financial position"},
        {
            "type": "table",
            "element_id": 7,
            "table_type": "合并资产负债表",
            "pageAt": 1,
            "rows": [
                [
                    {"text": ""},
                    {"text": "Group\n2024", "colStart": 2},
                    {"text": "Company\n2024", "colStart": 4},
                ],
                [
                    {"text": "Share capital"},
                    {"text": "1,000", "colStart": 2},
                    {"text": "800", "colStart": 4},
                ],
            ],
        },
    ]
    out = extract_org_tables(items, locale=None)
    assert len(out) == 2
    ids = {x["id"] for x in out}
    assert ids == {"7_grp", "7_std"}
    tb = next(x for x in out if x["id"] == "7_grp")
    assert tb["mergeIds"] == ["7_grp"]
    assert tb["primaryElementId"] == "7"
    assert "category" in tb
    assert "isFinancialStatementTable" in tb
    c = tb["rows"][0][1]
    assert "statementType" in c
    assert "gaapType" in c
    assert "caliberLabel" in c
    assert "accountingYear" in c
    assert c["statementType"] is not None
    assert c["gaapType"] is not None
    assert c["caliberLabel"] in {"合并", "单体", "混合", "未知"}


def test_expand_financial_table_dual_scope_uses_logical_col_keys() -> None:
    """同表 Group+Company 且 colStart 稀疏时，按逻辑列拆成两条二维表，不依赖数组下标。"""
    from src.applications.parsing.service.three_statements_frontend_service import (
        expand_financial_table_if_dual_caliber,
    )
    from src.applications.parsing.service.three_statements.cells import (
        align_table_rows,
    )

    table = {
        "category": "资产负债表",
        "years": 2024,
        "period": "年报",
        "rows": [
            [
                {"text": "Item", "colStart": 0},
                {"text": "G 2024", "colStart": 1, "caliberLabel": "合并"},
                {"text": "C 2024", "colStart": 2, "caliberLabel": "单体"},
                {"text": "G 2023", "colStart": 3, "caliberLabel": "合并"},
                {"text": "C 2023", "colStart": 4, "caliberLabel": "单体"},
            ],
            [
                {"text": "a", "colStart": 0},
                {"text": "1", "colStart": 1, "caliberLabel": "合并"},
                {"text": "2", "colStart": 2, "caliberLabel": "单体"},
                {"text": "3", "colStart": 3, "caliberLabel": "合并"},
                {"text": "4", "colStart": 4, "caliberLabel": "单体"},
            ],
            [
                {"text": "b", "colStart": 0},
                {"text": "30", "colStart": 3, "caliberLabel": "合并"},
                {"text": "40", "colStart": 4, "caliberLabel": "单体"},
            ],
        ],
    }
    parts = expand_financial_table_if_dual_caliber(table, locale=None)
    assert len(parts) == 2
    assert [cell["colStart"] for cell in parts[0]["rows"][1]] == [0, 1, 2]
    assert [cell["colStart"] for cell in parts[1]["rows"][1]] == [0, 1, 2]

    for part in parts:
        align_table_rows(part["rows"])

    assert [[cell["text"] for cell in part["rows"][1]] for part in parts] == [
        ["a", "1", "3"],
        ["a", "2", "4"],
    ]
    assert [[cell["text"] for cell in part["rows"][2]] for part in parts] == [
        ["b", "", "30"],
        ["b", "", "40"],
    ]


def test_expand_financial_table_does_not_restore_removed_empty_column() -> None:
    from src.applications.parsing.service.three_statements_frontend_service import (
        expand_financial_table_if_dual_caliber,
    )
    from src.applications.parsing.service.three_statements.cells import (
        align_table_rows,
    )

    table = {
        "category": "现金流量表",
        "years": 2024,
        "period": "年报",
        "rows": [
            [
                {"text": "Item", "colStart": 0},
                {"text": "", "colStart": 1, "caliberLabel": "合并"},
                {"text": "C 2024", "colStart": 2, "caliberLabel": "单体"},
                {"text": "G 2023", "colStart": 3, "caliberLabel": "合并"},
                {"text": "C 2023", "colStart": 4, "caliberLabel": "单体"},
            ],
            [
                {"text": "a", "colStart": 0},
                {"text": "", "colStart": 1, "caliberLabel": "合并"},
                {"text": "2", "colStart": 2, "caliberLabel": "单体"},
                {"text": "3", "colStart": 3, "caliberLabel": "合并"},
                {"text": "4", "colStart": 4, "caliberLabel": "单体"},
            ],
        ],
    }

    consolidated = expand_financial_table_if_dual_caliber(table)[0]
    align_table_rows(consolidated["rows"])

    assert [cell["text"] for cell in consolidated["rows"][1]] == ["a", "3"]
    assert [cell["colStart"] for cell in consolidated["rows"][1]] == [0, 1]


def test_infer_year_map_with_colstart_keeps_correct_year() -> None:
    rows = [
        [
            {"text": ""},
            {"text": "Group\n2024", "colStart": 2},
            {"text": "Group\n2023", "colStart": 3},
            {"text": "Company\n2024", "colStart": 4},
            {"text": "Company\n2023", "colStart": 5},
        ],
        [{"text": "Share capital"}, {"text": "1"}, {"text": "2"}, {"text": "3"}, {"text": "4"}],
    ]
    m = infer_column_index_to_year_for_table(rows, report_name="资产负债表")
    assert m.get(2) == 2024
    assert m.get(3) == 2023
    assert m.get(4) == 2024
    assert m.get(5) == 2023


def test_infer_year_map_falls_back_to_context_year_when_header_missing() -> None:
    rows = [
        [{"text": "项目"}, {"text": "金额"}],
        [{"text": "经营活动产生的现金流量净额"}, {"text": "25,430,417.98"}],
    ]
    m = infer_column_index_to_year_for_table(
        rows,
        context_above="现金流量表\n2024年度",
        context_below="",
        report_name="现金流量表",
    )
    assert m.get(1) == 2024


def test_extract_org_tables_single_scope_table_gets_single_and_year_from_context() -> None:
    pytest.importorskip("fastapi")
    from src.applications.parsing.service.three_statements_frontend_service import (
        extract_org_tables,
    )

    items = [
        {"type": "text", "md": "现金流量表\n2024年度"},
        {
            "type": "table",
            "element_id": 99,
            "table_type": "现金流量表",
            "pageAt": 1,
            "md": "现金流量表 2024年度",
            "rows": [
                [{"text": "项目"}, {"text": "行次"}, {"text": "金额"}],
                [{"text": "经营活动产生的现金流量净额"}, {"text": "1"}, {"text": "25,430,417.98"}],
            ],
        },
    ]

    out = extract_org_tables(items, locale=None)
    assert len(out) == 1
    tb = out[0]
    assert tb["category"] == "现金流量表"
    amount_cell = tb["rows"][1][2]
    assert amount_cell["caliberLabel"] == "单体"
    assert amount_cell["accountingYear"] == 2024


def test_infer_document_year_prefers_balance_sheet_columns_over_audit_signing_date() -> None:
    """篇首审计签章年（如 2024）不得覆盖三大表表头列上的报表年度（如 2022）。"""
    items = [
        {"type": "text", "md": "本报告于2024年3月15日签署。\n注册会计师2024年3月16日"},
        {
            "type": "table",
            "table_type": "资产负债表",
            "md": "项目\t期末余额\t期初余额",
            "rows": [
                [{"text": "项目"}, {"text": "2022年"}, {"text": "2021年"}],
                [{"text": "货币资金"}, {"text": "100"}, {"text": "90"}],
            ],
        },
    ]
    y, _period = infer_document_years_and_period(items)
    assert y == 2022


def test_infer_document_year_uses_financial_snippet_when_head_only_has_signing_year() -> None:
    """表头未解析出年份时，应优先从三大表 md 片段取年，而非与篇首签署年合并后取 max。"""
    items = [
        {"type": "text", "md": "签署日期：2024年6月1日"},
        {
            "type": "table",
            "table_type": "资产负债表",
            "md": "资产负债表\n2022年12月31日\n单位：元\n货币资金 9121495299.31",
            "rows": [
                [{"text": "项目"}, {"text": "期末余额"}, {"text": "期初余额"}],
                [{"text": "货币资金"}, {"text": "9121495299.31"}, {"text": "0"}],
            ],
        },
    ]
    y, _period = infer_document_years_and_period(items)
    assert y == 2022


def test_infer_document_year_explicit_annual_report_in_head_wins() -> None:
    items = [
        {"type": "text", "md": "山东某某股份有限公司\n2019年度报告"},
        {
            "type": "table",
            "table_type": "资产负债表",
            "md": "…",
            "rows": [
                [{"text": "项目"}, {"text": "2022年"}, {"text": "2021年"}],
                [{"text": "货币资金"}, {"text": "1"}, {"text": "2"}],
            ],
        },
    ]
    y, _period = infer_document_years_and_period(items)
    assert y == 2019


def test_infer_document_year_prefers_balance_sheet_columns_when_is_has_spurious_later_year() -> None:
    """利润表等误带「未来年」装饰行时，以资产负债表列报年度为准。"""
    pad = [[{"text": f"占位{i}"}, {"text": ""}, {"text": ""}] for i in range(6)]
    items = [
        {
            "type": "table",
            "table_type": "利润表",
            "md": "…",
            "rows": [
                [{"text": "编制单位"}, {"text": "2024年"}, {"text": "单位:元"}],
                [{"text": "项目"}, {"text": "期末余额"}, {"text": "期初余额"}],
                [{"text": "营业收入"}, {"text": "1"}, {"text": "2"}],
            ]
            + pad,
        },
        {
            "type": "table",
            "table_type": "资产负债表",
            "md": "…",
            "rows": [
                [{"text": "项目"}, {"text": "2022年"}, {"text": "2021年"}],
                [{"text": "货币资金"}, {"text": "100"}, {"text": "90"}],
            ]
            + pad,
        },
    ]
    y, _period = infer_document_years_and_period(items)
    assert y == 2022


def test_augment_column_years_closing_opening_without_digits_in_header() -> None:
    rows = [
        [{"text": "项目"}, {"text": "期末余额"}, {"text": "期初余额"}],
        [{"text": "货币资金"}, {"text": "100"}, {"text": "90"}],
    ]
    col = infer_column_index_to_year_for_table(rows, report_name="资产负债表")
    out = augment_column_years_for_comparative_period_labels(
        rows, col, doc_year=2022, doc_period="年报"
    )
    assert out.get(1) == 2022
    assert out.get(2) == 2021


def test_augment_column_years_does_not_override_explicit_two_years() -> None:
    rows = [
        [{"text": "项目"}, {"text": "2024年度"}, {"text": "2023年度"}],
        [{"text": "营收"}, {"text": "1"}, {"text": "2"}],
    ]
    col = infer_column_index_to_year_for_table(rows, report_name="利润表")
    out = augment_column_years_for_comparative_period_labels(
        rows, col, doc_year=2022, doc_period="年报"
    )
    assert out.get(1) == 2024
    assert out.get(2) == 2023


def test_extract_explicit_filing_year_from_blob() -> None:
    assert extract_explicit_filing_year_from_blob("本公司2023年度报告") == 2023
    assert extract_explicit_filing_year_from_blob("截至2022年12月31日止") == 2022
    assert extract_explicit_filing_year_from_blob("无年份关键词") is None


def test_infer_single_table_reporting_year_from_md_concat_year() -> None:
    rows = [
        [{"text": "编制单位：山东新巨丰科技包装股份有限公司2024年"}, {"text": "单位：元"}],
        [{"text": "项目"}, {"text": "期末余额"}, {"text": "期初余额"}],
        [{"text": "资产"}, {"text": "1"}, {"text": "2"}],
    ]
    y = infer_single_table_reporting_year(
        rows,
        report_name="资产负债表",
        caption="",
        fallback_doc_year=2022,
    )
    assert y == 2024


def test_infer_single_table_reporting_year_prefers_column_max_over_fallback() -> None:
    rows = [
        [{"text": "项目"}, {"text": "2023年12月31日"}, {"text": "2022年12月31日"}],
        [{"text": "货币资金"}, {"text": "10"}, {"text": "9"}],
    ]
    y = infer_single_table_reporting_year(
        rows,
        report_name="资产负债表",
        caption="",
        fallback_doc_year=2022,
    )
    assert y == 2023


def test_interpolate_balance_sheet_fills_year_between_known_peers() -> None:
    """md 无年的中间页 BS，在前后页已解析 2022 / 2024 时填补为 2023。"""
    rows_mid = [
        [{"text": "项目"}, {"text": "期末余额"}, {"text": "期初余额"}],
        [{"text": "货币资金"}, {"text": "1"}, {"text": "2"}],
    ]
    rows_lo = [
        [{"text": "项"}, {"text": "2022年"}, {"text": "2021年"}],
        [{"text": "货币资金"}, {"text": "10"}, {"text": "9"}],
    ]
    rows_hi = [
        [{"text": "项"}, {"text": "2024年"}, {"text": "2023年"}],
        [{"text": "货币资金"}, {"text": "11"}, {"text": "10"}],
    ]
    items = [
        {
            "type": "table",
            "page": 13,
            "element_id": 13,
            "table_type": "资产负债表",
            "rows": rows_lo,
            "md": "| 山东 | 2022年 | 元 |",
        },
        {
            "type": "table",
            "page": 15,
            "element_id": 15,
            "table_type": "资产负债表",
            "rows": rows_mid,
            "md": "| 项目 | 期末余额 | 期初余额 |",
        },
        {
            "type": "table",
            "page": 17,
            "element_id": 17,
            "table_type": "资产负债表",
            "rows": rows_hi,
            "md": "| 编制 | 2024年 | 元 |",
        },
    ]
    m = interpolate_balance_sheet_reporting_years_by_peer_pages(items)
    assert m.get(15) == 2023


def test_interpolate_balance_sheet_two_gaps_same_intermediate_year() -> None:
    r = [
        [{"text": "项目"}, {"text": "期末余额"}, {"text": "期初余额"}],
        [{"text": "x"}, {"text": "1"}, {"text": "2"}],
    ]
    r2 = [
        [{"text": "项"}, {"text": "2022年"}, {"text": "2021年"}],
        [{"text": "x"}, {"text": "1"}, {"text": "2"}],
    ]
    r4 = [
        [{"text": "项"}, {"text": "2024年"}, {"text": "2023年"}],
        [{"text": "x"}, {"text": "1"}, {"text": "2"}],
    ]
    items = [
        {"type": "table", "page": 1, "element_id": 1, "table_type": "资产负债表", "rows": r2, "md": "2022"},
        {"type": "table", "page": 2, "element_id": 2, "table_type": "资产负债表", "rows": r, "md": "|项目|"},
        {"type": "table", "page": 3, "element_id": 3, "table_type": "资产负债表", "rows": r, "md": "|项目|"},
        {"type": "table", "page": 4, "element_id": 4, "table_type": "资产负债表", "rows": r4, "md": "2024"},
    ]
    m = interpolate_balance_sheet_reporting_years_by_peer_pages(items)
    assert m.get(2) == 2023 and m.get(3) == 2023


def test_interpolate_cf_prefix_uses_preceding_is_strict_year() -> None:
    """现金流量表前几页 md 无年：用前文利润表已解析年 + 后文 CF 锚点年补前缀。"""
    r_none = [
        [{"text": "项目"}, {"text": "期末余额"}, {"text": "期初余额"}],
        [{"text": "x"}, {"text": "1"}, {"text": "2"}],
    ]
    r_is = [
        [{"text": "项目"}, {"text": "2023年12月31日"}, {"text": "2022年12月31日"}],
        [{"text": "x"}, {"text": "1"}, {"text": "2"}],
    ]
    r_cf_last = [
        [{"text": "项目"}, {"text": "2024年"}, {"text": "单位：元"}],
        [{"text": "项目"}, {"text": "期末余额"}, {"text": "期初余额"}],
        [{"text": "x"}, {"text": "1"}, {"text": "2"}],
    ]
    items = [
        {"type": "table", "page": 1, "element_id": 101, "table_type": "利润表", "rows": r_is, "md": "2023"},
        {"type": "table", "page": 2, "element_id": 102, "table_type": "现金流量表", "rows": r_none, "md": "|项目|"},
        {"type": "table", "page": 3, "element_id": 103, "table_type": "现金流量表", "rows": r_cf_last, "md": "2024"},
    ]
    m = interpolate_balance_sheet_reporting_years_by_peer_pages(items)
    assert m.get(102) == 2023


def test_strict_peer_year_uses_neighbor_bare_fiscal_year_before_cf_table() -> None:
    """表头「…公司 2022年」在上一文本块时：严格年应为 2022，peer 前缀不得改成 2023。"""
    r_none = [
        [{"text": "项目"}, {"text": "期末余额"}, {"text": "期初余额"}],
        [{"text": "x"}, {"text": "1"}, {"text": "2"}],
    ]
    items = [
        {"type": "table", "page": 1, "element_id": 1, "table_type": "利润表", "rows": r_none, "md": "利润表"},
        {"type": "text", "md": "制表单位：山东新巨丰科技包装股份有限公司 2022年\n单位：元"},
        {
            "type": "table",
            "page": 2,
            "element_id": 102,
            "table_type": "现金流量表",
            "rows": r_none,
            "md": "|项目|期末余额|期初余额|",
        },
    ]
    assert _strict_reporting_year_for_peer_gap_table(items[2], items=items, item_idx=2) == 2022


def test_interpolate_cf_amount_chain_overrides_peer_prefix_for_prior_year_table() -> None:
    """三套连续现金流量表：仅末套 md 含 2024 年，中间套 peer 误填 2023 时，用金额链回推首套为 2022。"""
    r22 = [
        [{"text": "项目"}, {"text": "期末余额"}, {"text": "期初余额"}],
        [{"text": "一、经营活动产生的现金流量:"}, {"text": ""}, {"text": ""}],
        [{"text": "销售商品、提供劳务收到的现金"}, {"text": "1,475,018,955.32"}, {"text": "1,159,828,180.06"}],
    ]
    r23 = [
        [{"text": "项目"}, {"text": "期末余额"}, {"text": "期初余额"}],
        [{"text": "一、经营活动产生的现金流量:"}, {"text": ""}, {"text": ""}],
        [{"text": "销售商品、提供劳务收到的现金"}, {"text": "1,631,297,479.39"}, {"text": "1,475,018,955.32"}],
    ]
    r24 = [
        [{"text": "项目"}, {"text": "2024年"}, {"text": "单位：元"}],
        [{"text": "项目"}, {"text": "期末余额"}, {"text": "期初余额"}],
        [{"text": "一、经营活动产生的现金流量:"}, {"text": ""}, {"text": ""}],
        [{"text": "销售商品、提供劳务收到的现金"}, {"text": "1,957,162,909.37"}, {"text": "1,631,297,479.39"}],
    ]
    items = [
        {"type": "table", "page": 7, "element_id": 7, "table_type": "现金流量表", "rows": r22, "md": "|项目|期末余额|期初余额|"},
        {"type": "table", "page": 9, "element_id": 9, "table_type": "现金流量表", "rows": r23, "md": "|项目|期末余额|期初余额|"},
        {"type": "table", "page": 11, "element_id": 11, "table_type": "现金流量表", "rows": r24, "md": "|项目|2024年|单位：元|"},
    ]
    m = interpolate_balance_sheet_reporting_years_by_peer_pages(items)
    assert m.get(7) == 2022 and m.get(9) == 2023 and m.get(11) == 2024


def test_interpolate_cf_does_not_peer_override_when_neighbor_states_2022() -> None:
    r_none = [
        [{"text": "项目"}, {"text": "期末余额"}, {"text": "期初余额"}],
        [{"text": "x"}, {"text": "1"}, {"text": "2"}],
    ]
    r_is = [
        [{"text": "项目"}, {"text": "2023年12月31日"}, {"text": "2022年12月31日"}],
        [{"text": "x"}, {"text": "1"}, {"text": "2"}],
    ]
    r_cf_last = [
        [{"text": "项目"}, {"text": "2024年"}, {"text": "单位：元"}],
        [{"text": "项目"}, {"text": "期末余额"}, {"text": "期初余额"}],
        [{"text": "x"}, {"text": "1"}, {"text": "2"}],
    ]
    items = [
        {"type": "table", "page": 1, "element_id": 101, "table_type": "利润表", "rows": r_is, "md": "利润表"},
        {"type": "text", "md": "制表单位：山东新巨丰科技包装股份有限公司 2022年\n单位：元"},
        {"type": "table", "page": 2, "element_id": 102, "table_type": "现金流量表", "rows": r_none, "md": "|项目|期末余额|期初余额|"},
        {"type": "table", "page": 3, "element_id": 103, "table_type": "现金流量表", "rows": r_cf_last, "md": "2024"},
    ]
    m = interpolate_balance_sheet_reporting_years_by_peer_pages(items)
    assert 102 not in m
    assert _strict_reporting_year_for_peer_gap_table(items[2], items=items, item_idx=2) == 2022
