from regression_mvp.utils.subject_match_artifacts import (
    _financial_table_views,
    _render_html,
    write_subject_match_table_artifact,
)


def _cell(
    cell_id: str,
    text: str,
    *,
    page: int,
    report_name: str,
    subject_id: str | None = None,
    year: int | None = None,
    caliber_label: str = "合并",
) -> dict:
    return {
        "cellId": cell_id,
        "text": text,
        "page": page,
        "tablePageAt": page,
        "reportName": report_name,
        "caliberLabel": caliber_label,
        "subjectId": subject_id,
        "accountingYear": year,
    }


def _request_payload() -> dict:
    return {
        "reportList": [
            {
                "reportName": "资产负债表",
                "subjectList": [{"subjectId": "cash", "subjectName": "货币资金"}],
            },
            {
                "reportName": "利润表",
                "subjectList": [{"subjectId": "revenue", "subjectName": "营业收入"}],
            },
        ],
        "tables": [
            {
                "reportName": "其他表",
                "pageAt": 1,
                "rows": [[_cell("note_r0_c0", "附注内容", page=1, report_name="其他表")]],
            },
            {
                "reportName": "合并资产负债表",
                "pageAt": 8,
                "rows": [[_cell("bs_r0_c0", "项目", page=8, report_name="合并资产负债表")]],
            },
            {
                "reportName": "合并利润表",
                "pageAt": 5,
                "rows": [[_cell("is_r0_c0", "项目", page=5, report_name="合并利润表")]],
            },
            {
                "reportName": "合并所有者权益变动表",
                "pageAt": 9,
                "rows": [[_cell("eq_r0_c0", "股东权益", page=9, report_name="合并所有者权益变动表")]],
            },
        ],
    }


def test_subject_match_preview_only_renders_three_statement_tables_in_page_order() -> None:
    actual = [
        [_cell("note_r0_c0", "无关附注", page=1, report_name="其他表")],
        [
            _cell("bs_r0_c0", "资产", page=8, report_name="合并资产负债表"),
            _cell("bs_r0_c1", "2024年12月31日", page=8, report_name="合并资产负债表", year=2024),
            _cell("bs_r0_c2", "2023年12月31日", page=8, report_name="合并资产负债表", year=2023),
        ],
        [
            _cell("bs_r1_c0", "货币资金", page=8, report_name="合并资产负债表", subject_id="cash"),
            _cell("bs_r1_c1", "100", page=8, report_name="合并资产负债表", year=2024),
            _cell("bs_r1_c2", "90", page=8, report_name="合并资产负债表", year=2023),
        ],
        [
            _cell("bs_r2_c0", "负债", page=9, report_name="合并资产负债表"),
            _cell("bs_r2_c1", "2024年12月31日", page=9, report_name="合并资产负债表", year=2024),
            _cell("bs_r2_c2", "2023年12月31日", page=9, report_name="合并资产负债表", year=2023),
        ],
        [
            _cell("is_r0_c0", "项目", page=5, report_name="合并利润表"),
            _cell("is_r0_c1", "2024年", page=5, report_name="合并利润表", year=2024),
            _cell("is_r0_c2", "2023年", page=5, report_name="合并利润表", year=2023),
        ],
        [
            _cell("is_r1_c0", "营业收入", page=5, report_name="合并利润表", subject_id="revenue"),
            _cell("is_r1_c1", "50", page=5, report_name="合并利润表", year=2024),
            _cell("is_r1_c2", "40", page=5, report_name="合并利润表", year=2023),
        ],
        [_cell("eq_r0_c0", "权益变动", page=9, report_name="合并所有者权益变动表")],
    ]

    rendered = _render_html(actual, actual, _request_payload())

    assert "模板科目" in rendered
    assert "源文件科目" in rendered
    assert "货币资金" in rendered
    assert "营业收入" in rendered
    assert "无关附注" not in rendered
    assert "权益变动" not in rendered
    assert '<td class="source">负债</td>' not in rendered
    assert rendered.index("合并利润表") < rendered.index("合并资产负债表")


def test_subject_match_preview_marks_unmatched_source_subjects() -> None:
    payload = _request_payload()
    payload["tables"] = [payload["tables"][1]]
    actual = [
        [
            _cell("bs_r1_c0", "其他流动资产", page=8, report_name="合并资产负债表"),
            _cell("bs_r1_c1", "12", page=8, report_name="合并资产负债表", year=2024),
            _cell("bs_r1_c2", "10", page=8, report_name="合并资产负债表", year=2023),
        ]
    ]

    rendered = _render_html(actual, actual, payload)

    assert "货币资金" in rendered
    assert "其他流动资产" in rendered
    assert "模板有，当前未匹配源文件科目" not in rendered
    assert '<td class="source"></td>' in rendered
    assert "文件科目未匹配模板" in rendered


def test_subject_match_source_preview_only_renders_source_rows(tmp_path) -> None:
    payload = _request_payload()
    payload["tables"] = [payload["tables"][1]]
    actual = [
        [
            _cell("bs_r1_c0", "其他流动资产", page=8, report_name="合并资产负债表"),
            _cell("bs_r1_c1", "12", page=8, report_name="合并资产负债表", year=2024),
        ]
    ]

    artifacts = write_subject_match_table_artifact(
        actual,
        actual,
        tmp_path,
        request_payload=payload,
    )
    source_rendered = (tmp_path / "subject_match_source_table.html").read_text(
        encoding="utf-8"
    )

    assert set(artifacts) == {"subject_match_table", "subject_match_source_table"}
    assert "科目匹配源文件预览" in source_rendered
    assert "其他流动资产" in source_rendered
    assert "货币资金" not in source_rendered


def test_subject_match_preview_selects_template_by_statement_caliber() -> None:
    payload = {
        "reportList": [
            {
                "reportId": "balance-parent",
                "reportName": "资产负债表",
                "reportCaliber": 1,
                "subjectList": [
                    {"subjectId": "parent-only", "subjectName": "单体专用科目"}
                ],
            },
            {
                "reportId": "balance-consolidated",
                "reportName": "资产负债表",
                "reportCaliber": 2,
                "subjectList": [
                    {"subjectId": "group-only", "subjectName": "合并专用科目"},
                    {"subjectId": "group-missing", "subjectName": "合并模板未命中科目"},
                ],
            },
        ],
        "tables": [
            {
                "reportName": "合并资产负债表",
                "pageAt": 8,
                "rows": [[
                    _cell(
                        "bs_r0_c0",
                        "项目",
                        page=8,
                        report_name="合并资产负债表",
                    )
                ]],
            }
        ],
    }
    actual = [
        [
            _cell(
                "bs_r1_c0",
                "合并来源科目",
                page=8,
                report_name="合并资产负债表",
                subject_id="group-only",
            ),
            _cell(
                "bs_r1_c1",
                "100",
                page=8,
                report_name="合并资产负债表",
                year=2024,
            ),
        ]
    ]

    rendered = _render_html(actual, actual, payload)

    assert "合并专用科目" in rendered
    assert "合并模板未命中科目" in rendered
    assert "单体专用科目" not in rendered
    assert "模板命中 1 / 2 项" in rendered


def test_subject_match_preview_separates_tables_with_same_cell_id_prefix() -> None:
    payload = {
        "reportList": [],
        "tables": [
            {
                "reportName": "合并现金流量表",
                "pageAt": 1,
                "rows": [[
                    _cell(
                        "1_r1_c0",
                        "Profit for the period before taxation",
                        page=1,
                        report_name="合并现金流量表",
                    )
                ]],
            },
            {
                "reportName": "单体现金流量表",
                "pageAt": 1,
                "rows": [[
                    _cell(
                        "1_r1_c0",
                        "Profit for the period before taxation",
                        page=1,
                        report_name="单体现金流量表",
                        caliber_label="单体",
                    )
                ]],
            },
        ],
    }
    consolidated_row = [
        _cell(
            "1_r1_c0",
            "Profit for the period before taxation",
            page=1,
            report_name="合并现金流量表",
        ),
        _cell(
            "1_r1_c1",
            "318,672",
            page=1,
            report_name="合并现金流量表",
        ),
    ]
    standalone_row = [
        _cell(
            "1_r1_c0",
            "Profit for the period before taxation",
            page=1,
            report_name="单体现金流量表",
            caliber_label="单体",
        ),
        _cell(
            "1_r1_c1",
            "170,975",
            page=1,
            report_name="单体现金流量表",
            caliber_label="单体",
        ),
    ]

    views, source_table_count = _financial_table_views(
        [consolidated_row, standalone_row],
        payload,
    )

    assert source_table_count == 2
    assert [(view["report_name"], view["caliber"]) for view in views] == [
        ("合并现金流量表", "合并"),
        ("单体现金流量表", "单体"),
    ]
    assert [[cell["text"] for cell in view["rows"][0]] for view in views] == [
        ["Profit for the period before taxation", "318,672"],
        ["Profit for the period before taxation", "170,975"],
    ]


def test_subject_match_preview_ignores_trailing_empty_amount_placeholders() -> None:
    report_name = "合并现金流量表"
    section_row = [
        _cell("cf_r0_c0", "OPERATING ACTIVITIES", page=1, report_name=report_name),
        *[
            _cell(f"cf_r0_c{index}", "", page=1, report_name=report_name)
            for index in range(1, 6)
        ],
    ]
    amount_row = [
        _cell("cf_r1_c0", "Profit before taxation", page=1, report_name=report_name),
        _cell("cf_r1_c1", "318,672", page=1, report_name=report_name),
        _cell("cf_r1_c2", "690,614", page=1, report_name=report_name),
    ]
    payload = {
        "reportList": [],
        "tables": [
            {
                "reportName": report_name,
                "pageAt": 1,
                "rows": [section_row, amount_row],
            }
        ],
    }

    rendered = _render_html([section_row, amount_row], [section_row, amount_row], payload)

    assert rendered.count('<th class="amount">') == 2
    assert "金额 3" not in rendered
    assert "318,672" in rendered
    assert "690,614" in rendered
