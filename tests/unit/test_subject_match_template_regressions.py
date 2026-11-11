from __future__ import annotations

from pathlib import Path

import pytest

from regression_mvp.utils.json_io import load_json
from regression_mvp.utils.runners.financial_subject_match_runner import _load_report_list
from regression_mvp.utils.subject_match_request_builder import (
    build_subject_match_request,
    local_template_config_key,
    load_local_report_list,
    statement_tables_are_english,
)
from src.applications.parsing.schemas.subject_match_schema import Cell, SubjectMatchRequest
from src.applications.parsing.service.subject_match_service import SubjectMatchService
from src.applications.parsing.service.subject_matching.llm_helpers import (
    llm_review_matches,
)


BALANCE_SUBJECT_IDS = {
    "流动资产：": "balance-current-assets-heading",
    "一年内到期的非流动资产": "balance-current-non-current",
    "应收票据及应收账款": "balance-receivables-combined",
    "应收账款": "balance-accounts-receivable",
    "负债和所有者权益(或股东权益)总计": "balance-liabilities-equity-total",
}
CASHFLOW_ENDING_CASH_ID = "cashflow-ending-cash"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGRESSION_ROOT = PROJECT_ROOT / "regression_mvp"


def _reports() -> list[dict]:
    return [
        {
            "reportId": "balance-sheet",
            "reportName": "资产负债表",
            "reportCaliber": 2,
            "subjectList": [
                {
                    "subjectId": BALANCE_SUBJECT_IDS["流动资产："],
                    "subjectName": "流动资产：",
                    "synonymSubject": [],
                    "summaryRelation": [],
                },
                {
                    "subjectId": BALANCE_SUBJECT_IDS["应收票据及应收账款"],
                    "subjectName": "应收票据及应收账款",
                    "synonymSubject": ["应收票据净额"],
                    "summaryRelation": [],
                },
                {
                    "subjectId": BALANCE_SUBJECT_IDS["应收账款"],
                    "subjectName": "应收账款",
                    "synonymSubject": ["应收账款净额", "应收款项", "应收账款"],
                    "summaryRelation": [],
                },
                {
                    "subjectId": BALANCE_SUBJECT_IDS["一年内到期的非流动资产"],
                    "subjectName": "一年内到期的非流动资产",
                    "synonymSubject": ["一年内到期的长期应收款"],
                    "summaryRelation": [],
                },
                {
                    "subjectId": BALANCE_SUBJECT_IDS[
                        "负债和所有者权益(或股东权益)总计"
                    ],
                    "subjectName": "负债和所有者权益(或股东权益)总计",
                    "synonymSubject": ["负债和股东权益总计"],
                    "summaryRelation": [],
                },
            ],
        },
        {
            "reportId": "cash-flow-statement",
            "reportName": "现金流量表",
            "reportCaliber": 2,
            "subjectList": [
                {
                    "subjectId": CASHFLOW_ENDING_CASH_ID,
                    "subjectName": "六、期末现金及现金等价物余额",
                    "synonymSubject": ["年末现金及现金等价物余额"],
                    "summaryRelation": [],
                }
            ],
        },
    ]


def _cell(
    text: str,
    cell_id: str,
    *,
    subject_id: str | None = None,
) -> dict:
    return {
        "cellId": cell_id,
        "subjectId": subject_id,
        "text": text,
        "rowspan": 1,
        "colspan": 1,
        "page": 1,
        "bBox": {},
    }


def _table(report_name: str, cells: list[dict]) -> dict:
    return {
        "reportName": report_name,
        "pageAt": 1,
        "rows": [[cell] for cell in cells],
    }


def _subject_ids_by_text(result) -> dict[str, str | None]:
    return {
        cell.text: cell.subjectId
        for table in result.tables
        for row in table.rows
        for cell in row
    }


@pytest.mark.asyncio
async def test_runtime_template_definitive_matches_take_priority() -> None:
    request = SubjectMatchRequest(
        fileId="runtime-template-priority",
        reportList=_reports(),
        tables=[
            _table(
                "资产负债表",
                [
                    _cell("流动资产：", "balance-category-heading"),
                    _cell(
                        "负债和股东权益",
                        "balance-heading",
                        subject_id=BALANCE_SUBJECT_IDS[
                            "负债和所有者权益(或股东权益)总计"
                        ],
                    ),
                    _cell("一年内到期的长期应收款", "balance-current"),
                    _cell(
                        "应收账款",
                        "balance-receivable",
                        subject_id=BALANCE_SUBJECT_IDS["应收票据及应收账款"],
                    ),
                    _cell("负债和股东权益总计", "balance-total"),
                ],
            ),
            _table(
                "现金流量表",
                [_cell("六、年末现金及现金等价物余额", "cashflow-ending")],
            ),
        ],
    )

    result = await SubjectMatchService(use_llm=False).match_subjects(request)

    assert _subject_ids_by_text(result) == {
        "流动资产：": None,
        "负债和股东权益": None,
        "一年内到期的长期应收款": BALANCE_SUBJECT_IDS[
            "一年内到期的非流动资产"
        ],
        "应收账款": BALANCE_SUBJECT_IDS["应收账款"],
        "负债和股东权益总计": BALANCE_SUBJECT_IDS[
            "负债和所有者权益(或股东权益)总计"
        ],
        "六、年末现金及现金等价物余额": CASHFLOW_ENDING_CASH_ID,
    }


@pytest.mark.asyncio
async def test_ambiguous_runtime_synonym_does_not_override_existing_match() -> None:
    reports = _reports()
    reports[0]["subjectList"].extend(
        [
            {
                "subjectId": "ambiguous-a",
                "subjectName": "测试科目甲",
                "synonymSubject": ["共同别名"],
                "summaryRelation": [],
            },
            {
                "subjectId": "ambiguous-b",
                "subjectName": "测试科目乙",
                "synonymSubject": ["共同别名"],
                "summaryRelation": [],
            },
        ]
    )
    request = SubjectMatchRequest(
        fileId="ambiguous-runtime-template",
        reportList=reports,
        tables=[
            _table(
                "资产负债表",
                [_cell("共同别名", "ambiguous", subject_id="java-existing-id")],
            )
        ],
    )

    result = await SubjectMatchService(use_llm=False).match_subjects(request)

    assert result.tables[0].rows[0][0].subjectId == "java-existing-id"


@pytest.mark.asyncio
async def test_ambiguous_runtime_synonym_is_not_assigned_by_template_order() -> None:
    reports = _reports()
    reports[0]["subjectList"].extend(
        [
            {
                "subjectId": "ambiguous-a",
                "subjectName": "测试科目甲",
                "synonymSubject": ["共同别名"],
                "summaryRelation": [],
            },
            {
                "subjectId": "ambiguous-b",
                "subjectName": "测试科目乙",
                "synonymSubject": ["共同别名"],
                "summaryRelation": [],
            },
        ]
    )
    request = SubjectMatchRequest(
        fileId="ambiguous-runtime-template-without-owner",
        reportList=reports,
        tables=[_table("资产负债表", [_cell("共同别名", "ambiguous")])],
    )

    result = await SubjectMatchService(use_llm=False).match_subjects(request)

    assert result.tables[0].rows[0][0].subjectId is None


@pytest.mark.asyncio
async def test_llm_review_checks_a_single_untrusted_match() -> None:
    class ReviewingModel:
        calls = 0

        async def ainvoke(self, _prompt):
            self.calls += 1
            return type(
                "Response",
                (),
                {"content": '{"reject": [1], "reasons": {"1": "语义不一致"}}'},
            )()

    class ReviewingMatcher:
        model = ReviewingModel()

    cell = Cell(
        cellId="untrusted-cell",
        subjectId="candidate-subject",
        text="源文件科目",
        page=1,
    )
    matcher = ReviewingMatcher()

    await llm_review_matches(
        {"源文件科目": [cell]},
        {"candidate-subject": "候选模板科目"},
        "利润表",
        "income",
        matcher,
        trusted_rule_matches={},
    )

    assert matcher.model.calls == 1
    assert cell.subjectId is None


def test_regression_framework_uses_scraped_platform_template() -> None:
    api_config = load_json(REGRESSION_ROOT / "config.json")["api"]

    class NoTemplateApiCall:
        def get_templates(self) -> list:
            raise AssertionError("local_file template source must not call /templates")

    reports = _load_report_list(NoTemplateApiCall(), REGRESSION_ROOT, api_config)
    direct_reports = load_local_report_list(REGRESSION_ROOT, api_config)

    assert reports == direct_reports
    assert len(reports) == 6
    assert sum(len(report["subjectList"]) for report in reports) == 622
    assert {report["reportCaliber"] for report in reports} == {1, 2}
    assert len({report["reportId"] for report in reports}) == 6

    subjects = {
        subject["subjectName"]: subject
        for report in reports
        for subject in report["subjectList"]
    }
    assert "一年内到期的长期应收款" in subjects[
        "一年内到期的非流动资产"
    ]["synonymSubject"]
    assert subjects["应收账款"]["subjectId"] == "8d3740e5615d464d8ef66f53e132ddb9"
    assert "负债和股东权益总计" in subjects[
        "负债和所有者权益(或股东权益)总计"
    ]["synonymSubject"]
    assert "年末现金及现金等价物余额" in subjects[
        "六、期末现金及现金等价物余额"
    ]["synonymSubject"]

    SubjectMatchRequest(fileId="platform-template-schema", reportList=reports, tables=[])


def test_regression_framework_selects_ifrs_template_for_english_statements(
    monkeypatch,
) -> None:
    api_config = load_json(REGRESSION_ROOT / "config.json")["api"]

    class NoTemplateApiCall:
        def get_templates(self) -> list:
            raise AssertionError("local_file template source must not call /templates")

    org_tables = [
        {
            "text": "Cashflow Statement",
            "caliberLabel": "Consolidated",
            "rows": [
                [
                    {"text": "Item"},
                    {"text": "2024"},
                ],
                [
                    {"text": "Profit before tax"},
                    {"text": "318,672"},
                ],
                [
                    {"text": "Cash flows from operating activities"},
                    {"text": "496,921"},
                ],
            ],
        }
    ]

    monkeypatch.setattr(
        "regression_mvp.utils.runners.financial_subject_match_runner."
        "infer_report_presentations",
        lambda *_args, **_kwargs: {"CF": "INDIRECT"},
    )

    reports = _load_report_list(
        NoTemplateApiCall(),
        REGRESSION_ROOT,
        api_config,
        org_tables=org_tables,
    )

    assert statement_tables_are_english(org_tables) is True
    assert local_template_config_key(org_tables) == "subject_match_english_template_file"
    assert len(reports) == 2
    assert sum(len(report["subjectList"]) for report in reports) == 582
    assert {report["reportCaliber"] for report in reports} == {1, 2}

    subjects = {
        subject["subjectId"]: subject
        for report in reports
        if report["reportCaliber"] == 2
        for subject in report["subjectList"]
    }
    assert subjects["CF_MAI001"]["subjectName"] == "Statement of cash flows (indirect method)"
    assert "CF_DIR001" not in subjects
    assert "CF_EXT001" not in subjects

    SubjectMatchRequest(fileId="ifrs-template-schema", reportList=reports, tables=[])


def test_regression_framework_keeps_chinese_template_for_chinese_statements() -> None:
    org_tables = [
        {
            "text": "现金流量表",
            "rows": [
                [{"text": "项目"}, {"text": "2024年度"}],
                [{"text": "经营活动产生的现金流量净额"}, {"text": "496,921"}],
            ],
        }
    ]

    assert statement_tables_are_english(org_tables) is False
    assert local_template_config_key(org_tables) == "subject_match_template_file"


def test_statement_language_detection_ignores_numeric_cells() -> None:
    org_tables = [
        {
            "rows": [
                [{"text": "Revenue"}, {"text": "2024"}, {"text": "1,000"}],
                [{"text": "Operating profit"}, {"text": "2023"}, {"text": "900"}],
                [{"text": "Income tax expense"}, {"text": "2022"}, {"text": "100"}],
            ]
        }
    ]

    assert statement_tables_are_english(org_tables) is True


def test_subject_match_request_preserves_statement_caliber_in_table_name() -> None:
    request = build_subject_match_request(
        file_id="statement-calibers",
        report_list=_reports(),
        org_tables=[
            {
                "text": "现金流量表",
                "caliberLabel": "单体",
                "pageAt": 155,
                "rows": [[
                    {"text": "取得投资收益收到的现金", "source_page_no": 156},
                    {"text": ""},
                ]],
            },
            {
                "text": "现金流量表",
                "caliberLabel": "合并",
                "pageAt": 149,
                "rows": [],
            },
        ],
    )

    assert [table["reportName"] for table in request["tables"]] == [
        "单体现金流量表",
        "合并现金流量表",
    ]
    assert request["tables"][0]["rows"][0][0]["page"] == 156
    assert request["tables"][0]["rows"][0][1]["page"] == 156


def _minimal_subject(subject_id: str, subject_name: str) -> dict:
    return {
        "subjectId": subject_id,
        "subjectName": subject_name,
        "synonymSubject": [],
        "summaryRelation": [],
    }


@pytest.mark.asyncio
async def test_accounting_equivalence_does_not_depend_on_template_synonyms() -> None:
    reports = [
        {
            "reportId": "profit",
            "reportName": "利润表",
            "reportCaliber": 2,
            "subjectList": [
                _minimal_subject(
                    "derecognition-result",
                    "以摊余成本计量的金融资产终止确认产生的收益(损失以“-”号填列)",
                )
            ],
        },
        {
            "reportId": "cashflow",
            "reportName": "现金流量表",
            "reportCaliber": 2,
            "subjectList": [
                _minimal_subject(
                    "purchase-long-term-assets",
                    "购置固定资产、无形资产及其他长期资产所支付的现金",
                ),
                _minimal_subject(
                    "disposal-investee",
                    "处置子公司及其他营业单位收到的现金净额",
                ),
                _minimal_subject("absorbed-investment", "吸收投资收到的现金"),
                _minimal_subject(
                    "minority-investment",
                    "其中：子公司吸收少数股东投资收到的现金",
                ),
            ],
        },
    ]
    request = SubjectMatchRequest(
        fileId="code-only-accounting-equivalence",
        reportList=reports,
        tables=[
            _table(
                "合并利润表",
                [
                    _cell(
                        "以摊余成本计量的金融资产终止确认损失",
                        "derecognition-source",
                    )
                ],
            ),
            _table(
                "合并现金流量表",
                [
                    _cell(
                        "购建固定资产、无形资产和其他长期资产支付的现金",
                        "purchase-source",
                    ),
                    _cell(
                        "处置合营或联营公司所收到的现金",
                        "disposal-source",
                    ),
                    _cell(
                        "其他权益工具持有者投入的现金",
                        "equity-source",
                    ),
                ],
            ),
        ],
    )

    result = await SubjectMatchService(use_llm=False).match_subjects(request)

    assert _subject_ids_by_text(result) == {
        "以摊余成本计量的金融资产终止确认损失": "derecognition-result",
        "购建固定资产、无形资产和其他长期资产支付的现金": (
            "purchase-long-term-assets"
        ),
        "处置合营或联营公司所收到的现金": "disposal-investee",
        "其他权益工具持有者投入的现金": "absorbed-investment",
    }


@pytest.mark.asyncio
async def test_subject_match_repairs_two_subjects_merged_by_malformed_rowspan() -> None:
    report = {
        "reportId": "profit",
        "reportName": "利润表",
        "reportCaliber": 2,
        "subjectList": [
            _minimal_subject(
                "associate-investment-income",
                "其中：对联营企业和合营企业的投资收益",
            ),
            _minimal_subject(
                "amortized-cost-derecognition",
                "以摊余成本计量的金融资产终止确认产生的收益(损失以“-”号填列)",
            ),
        ],
    }
    request = SubjectMatchRequest(
        fileId="malformed-rowspan-subjects",
        reportList=[report],
        tables=[
            {
                "reportName": "合并利润表",
                "pageAt": 3,
                "rows": [
                    [
                        {
                            **_cell(
                                "其中：对联营企业和合营企业的投资收益/(损失）"
                                "以摊余成本计量的金融资产终止确认损失",
                                "subject-rowspan",
                            ),
                            "rowspan": 2,
                        },
                        _cell("1,277,455", "current-first"),
                        _cell("(685,885)", "prior-first"),
                    ],
                    [
                        _cell("-", "current-second"),
                        _cell("(22,719)", "prior-second"),
                    ],
                ],
            }
        ],
    )

    result = await SubjectMatchService(use_llm=False).match_subjects(request)
    rows = result.tables[0].rows

    assert [[cell.text for cell in row] for row in rows] == [
        [
            "其中：对联营企业和合营企业的投资收益/(损失）",
            "1,277,455",
            "(685,885)",
        ],
        [
            "以摊余成本计量的金融资产终止确认损失",
            "-",
            "(22,719)",
        ],
    ]
    assert rows[0][0].subjectId == "associate-investment-income"
    assert rows[1][0].subjectId == "amortized-cost-derecognition"
    assert rows[0][0].rowSpan == rows[1][0].rowSpan == 1


@pytest.mark.asyncio
async def test_subject_match_keeps_legitimate_rowspan_without_two_template_subjects() -> None:
    report = {
        "reportId": "profit",
        "reportName": "利润表",
        "reportCaliber": 2,
        "subjectList": [_minimal_subject("revenue", "营业收入")],
    }
    request = SubjectMatchRequest(
        fileId="legitimate-rowspan",
        reportList=[report],
        tables=[
            {
                "reportName": "合并利润表",
                "pageAt": 1,
                "rows": [
                    [
                        {**_cell("营业收入", "revenue-rowspan"), "rowspan": 2},
                        _cell("100", "revenue-current"),
                        _cell("90", "revenue-prior"),
                    ],
                    [_cell("80", "detail-current"), _cell("70", "detail-prior")],
                ],
            }
        ],
    )

    result = await SubjectMatchService(use_llm=False).match_subjects(request)

    assert [len(row) for row in result.tables[0].rows] == [3, 2]
    assert result.tables[0].rows[0][0].rowSpan == 2


@pytest.mark.asyncio
async def test_exact_subsidiary_disposal_owns_shared_platform_template_subject() -> None:
    request = SubjectMatchRequest(
        fileId="disposal-subject-ownership",
        reportList=[
            {
                "reportId": "cashflow",
                "reportName": "现金流量表",
                "reportCaliber": 2,
                "subjectList": [
                    {
                        "subjectId": "disposal-investee",
                        "subjectName": "处置子公司及其他营业单位收到的现金净额",
                        "synonymSubject": ["处置联营及合营企业收到的现金净额"],
                        "summaryRelation": [],
                    }
                ],
            }
        ],
        tables=[
            _table(
                "合并现金流量表",
                [
                    _cell(
                        "处置合营或联营公司所收到的现金",
                        "joint-associate-disposal",
                    ),
                    _cell(
                        "处置子公司及其他营业单位收到的现金净额",
                        "subsidiary-disposal",
                    ),
                ],
            )
        ],
    )

    result = await SubjectMatchService(use_llm=False).match_subjects(request)

    assert _subject_ids_by_text(result) == {
        "处置合营或联营公司所收到的现金": None,
        "处置子公司及其他营业单位收到的现金净额": "disposal-investee",
    }


@pytest.mark.asyncio
async def test_equity_instrument_inflow_does_not_duplicate_an_owned_subject() -> None:
    request = SubjectMatchRequest(
        fileId="occupied-accounting-subject",
        reportList=[
            {
                "reportId": "cashflow",
                "reportName": "现金流量表",
                "reportCaliber": 2,
                "subjectList": [
                    _minimal_subject("absorbed-investment", "吸收投资收到的现金")
                ],
            }
        ],
        tables=[
            _table(
                "合并现金流量表",
                [
                    _cell("吸收投资收到的现金", "exact-investment"),
                    _cell(
                        "其他权益工具持有者投入的现金",
                        "additional-equity-inflow",
                    ),
                ],
            )
        ],
    )

    result = await SubjectMatchService(use_llm=False).match_subjects(request)

    assert _subject_ids_by_text(result) == {
        "吸收投资收到的现金": "absorbed-investment",
        "其他权益工具持有者投入的现金": None,
    }


@pytest.mark.asyncio
async def test_future_specific_template_subject_overrides_accounting_fallback() -> None:
    request = SubjectMatchRequest(
        fileId="specific-template-subject",
        reportList=[
            {
                "reportId": "cashflow",
                "reportName": "现金流量表",
                "reportCaliber": 2,
                "subjectList": [
                    _minimal_subject("absorbed-investment", "吸收投资收到的现金"),
                    _minimal_subject(
                        "specific-equity-inflow",
                        "其他权益工具持有者投入的现金",
                    ),
                ],
            }
        ],
        tables=[
            _table(
                "合并现金流量表",
                [
                    _cell(
                        "其他权益工具持有者投入的现金",
                        "specific-equity-source",
                    )
                ],
            )
        ],
    )

    result = await SubjectMatchService(use_llm=False).match_subjects(request)

    assert _subject_ids_by_text(result) == {
        "其他权益工具持有者投入的现金": "specific-equity-inflow"
    }


@pytest.mark.asyncio
async def test_additional_accounting_equivalence_does_not_depend_on_llm() -> None:
    reports = [
        {
            "reportId": "profit",
            "reportName": "利润表",
            "reportCaliber": 2,
            "subjectList": [
                _minimal_subject(
                    "receivable-financing-fair-value",
                    "2.其他债权投资公允价值变动",
                ),
                _minimal_subject(
                    "other-debt-credit-impairment",
                    "4.其他债权投资信用减值准备",
                ),
                _minimal_subject(
                    "foreign-currency-translation",
                    "6.外币财务报表折算差额",
                ),
            ],
        },
        {
            "reportId": "cashflow",
            "reportName": "现金流量表",
            "reportCaliber": 2,
            "subjectList": [
                _minimal_subject(
                    "subsidiary-acquisition",
                    "取得子公司及其他营业单位支付的现金净额",
                ),
                _minimal_subject(
                    "operating-net-cashflow",
                    "经营活动产生的现金流量净额",
                ),
                _minimal_subject(
                    "financing-net-cashflow",
                    "筹资活动产生的现金流量净额",
                ),
                _minimal_subject(
                    "cash-equivalents-net-change",
                    "五、现金及现金等价物净增加额",
                ),
            ],
        },
    ]
    request = SubjectMatchRequest(
        fileId="additional-accounting-equivalence",
        reportList=reports,
        tables=[
            _table(
                "合并利润表",
                [
                    _cell(
                        "应收款项融资公允价值变动",
                        "receivable-financing-source",
                    ),
                    _cell(
                        "应收款项融资信用减值准备",
                        "receivable-financing-credit-impairment-source",
                    ),
                    _cell("外币报表折算差额", "foreign-currency-source"),
                ],
            ),
            _table(
                "合并现金流量表",
                [
                    _cell(
                        "收购子公司及其他营业单位支付的现金净额",
                        "subsidiary-acquisition-source",
                    ),
                    _cell(
                        "经营活动（使用）/产生的现金流量净额",
                        "operating-net-source",
                    ),
                    _cell(
                        "筹资活动产生/(使用)的现金流量净额",
                        "financing-net-source",
                    ),
                    _cell(
                        "五、现金及现金等价物(减少)/增加额",
                        "cash-equivalents-bidirectional-change-source",
                    ),
                ],
            ),
        ],
    )

    result = await SubjectMatchService(use_llm=False).match_subjects(request)

    assert _subject_ids_by_text(result) == {
        "应收款项融资公允价值变动": "receivable-financing-fair-value",
        "应收款项融资信用减值准备": "other-debt-credit-impairment",
        "外币报表折算差额": "foreign-currency-translation",
        "收购子公司及其他营业单位支付的现金净额": (
            "subsidiary-acquisition"
        ),
        "经营活动（使用）/产生的现金流量净额": "operating-net-cashflow",
        "筹资活动产生/(使用)的现金流量净额": "financing-net-cashflow",
        "五、现金及现金等价物(减少)/增加额": (
            "cash-equivalents-net-change"
        ),
    }


@pytest.mark.asyncio
async def test_llm_review_cannot_clear_trusted_accounting_rules(monkeypatch) -> None:
    class RejectingModel:
        async def ainvoke(self, _prompt):
            raise AssertionError("trusted rule matches must not reach LLM review")

    class RejectingMatcher:
        model = RejectingModel()

        async def select_best_match(self, *_args, **_kwargs):
            raise AssertionError("trusted rule matches must not use LLM selection")

    report = {
        "reportId": "cashflow",
        "reportName": "现金流量表",
        "reportCaliber": 2,
        "subjectList": [
            _minimal_subject(
                "purchase-long-term-assets",
                "购置固定资产、无形资产及其他长期资产所支付的现金",
            ),
            _minimal_subject(
                "subsidiary-acquisition",
                "取得子公司及其他营业单位支付的现金净额",
            ),
            _minimal_subject(
                "cash-equivalents-net-change",
                "五、现金及现金等价物净增加额",
            ),
        ],
    }
    request = SubjectMatchRequest(
        fileId="trusted-rules-survive-llm-review",
        reportList=[report],
        tables=[
            _table(
                "合并现金流量表",
                [
                    _cell(
                        "购建固定资产、无形资产和其他长期资产支付的现金",
                        "purchase-source",
                    ),
                    _cell(
                        "收购子公司及其他营业单位支付的现金净额",
                        "acquisition-source",
                    ),
                    _cell(
                        "五、现金及现金等价物净减少额",
                        "net-change-source",
                    ),
                ],
            )
        ],
    )
    service = SubjectMatchService(use_llm=True)

    async def no_llm_normalization(*_args, **_kwargs):
        return {}

    monkeypatch.setattr(service, "_llm_normalize_batch", no_llm_normalization)
    monkeypatch.setattr(service, "_get_llm_matcher", lambda: RejectingMatcher())

    result = await service.match_subjects(request)

    assert _subject_ids_by_text(result) == {
        "购建固定资产、无形资产和其他长期资产支付的现金": (
            "purchase-long-term-assets"
        ),
        "收购子公司及其他营业单位支付的现金净额": (
            "subsidiary-acquisition"
        ),
        "五、现金及现金等价物净减少额": "cash-equivalents-net-change",
    }
