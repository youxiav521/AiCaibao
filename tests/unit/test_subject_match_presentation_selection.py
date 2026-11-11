from __future__ import annotations

import json
from pathlib import Path

import pytest

from regression_mvp.utils import subject_match_presentation as presentation_module
from regression_mvp.utils.runners import financial_subject_match_runner as runner_module
from regression_mvp.utils.subject_match_presentation import (
    PresentationDecision,
    PresentationDecisionResponse,
    build_presentation_options,
    infer_report_presentations,
)
from regression_mvp.utils.subject_match_request_builder import (
    load_local_report_list,
    load_raw_local_report_list,
)


def _subject(
    subject_id: str,
    name: str,
    presentation: str,
    *,
    children: list[str] | None = None,
) -> dict:
    return {
        "subjectId": subject_id,
        "subjectName": name,
        "subjectLevel": 1,
        "presentation": presentation,
        "synonymSubject": [],
        "summaryRelation": children or [],
    }


def _report(report_type: str, subjects: list[dict]) -> dict:
    report_names = {"BS": "资产负债表", "IS": "利润表", "CF": "现金流量表"}
    return {
        "reportId": f"test-{report_type.lower()}",
        "reportName": report_names[report_type],
        "reportType": report_type,
        "subjectList": subjects,
    }


def test_infer_presentations_calls_llm_only_for_ambiguous_reports(monkeypatch) -> None:
    report_list = [
        _report(
            "BS",
            [
                _subject("bs-current", "Current/non-current", "CURT_NONCURT"),
                _subject("bs-liquidity", "Liquidity order", "LIQ_ORDER"),
            ],
        ),
        _report("IS", [_subject("is-function", "By function", "BY_FUNC")]),
    ]
    calls: list[dict] = []

    async def fake_invoke(**kwargs):
        calls.append(kwargs)
        return PresentationDecisionResponse(
            items=[
                PresentationDecision(
                    report_type="BS",
                    presentation="LIQ_ORDER",
                    confidence=0.96,
                    reason="Assets and liabilities are ordered by liquidity.",
                )
            ]
        )

    monkeypatch.setattr(
        presentation_module,
        "ainvoke_llm_with_structured_output",
        fake_invoke,
    )

    result = infer_report_presentations(
        report_list,
        [
            {"text": "Statement of financial position", "rows": [[{"text": "Cash"}]]},
            {"text": "Income statement", "rows": [[{"text": "Cost of sales"}]]},
        ],
        model_id="presentation-model",
        timeout_s=12,
    )

    assert result == {"BS": "LIQ_ORDER", "IS": "BY_FUNC"}
    assert len(calls) == 1
    assert calls[0]["model_id"] == "presentation-model"
    assert calls[0]["timeout_s"] == 12
    assert "CURT_NONCURT" in calls[0]["prompt"]
    assert "LIQ_ORDER" in calls[0]["prompt"]
    assert "IS:" not in calls[0]["prompt"]


def test_infer_presentations_rejects_value_outside_template(monkeypatch) -> None:
    report_list = [
        _report(
            "BS",
            [
                _subject("bs-current", "Current/non-current", "CURT_NONCURT"),
                _subject("bs-liquidity", "Liquidity order", "LIQ_ORDER"),
            ],
        )
    ]

    async def fake_invoke(**_kwargs):
        return PresentationDecisionResponse(
            items=[PresentationDecision(report_type="BS", presentation="OTHER")]
        )

    monkeypatch.setattr(
        presentation_module,
        "ainvoke_llm_with_structured_output",
        fake_invoke,
    )

    with pytest.raises(ValueError, match="模板外值"):
        infer_report_presentations(
            report_list,
            [{"text": "Balance sheet", "rows": [[{"text": "Cash"}]]}],
            model_id="presentation-model",
            timeout_s=12,
        )


def test_infer_presentations_coalesces_identical_duplicate_decisions(monkeypatch) -> None:
    report_list = [
        _report(
            "CF",
            [
                _subject("cf-direct", "Direct method", "DIRECT"),
                _subject("cf-indirect", "Indirect method", "INDIRECT"),
            ],
        )
    ]

    async def fake_invoke(**_kwargs):
        return PresentationDecisionResponse(
            items=[
                PresentationDecision(
                    report_type="CF",
                    presentation="INDIRECT",
                    confidence=0.82,
                    reason="First table block.",
                ),
                PresentationDecision(
                    report_type="CF",
                    presentation="INDIRECT",
                    confidence=0.94,
                    reason="Second table block.",
                ),
            ]
        )

    monkeypatch.setattr(
        presentation_module,
        "ainvoke_llm_with_structured_output",
        fake_invoke,
    )

    result = infer_report_presentations(
        report_list,
        [
            {"text": "Cashflow Statement", "rows": [[{"text": "Profit before tax"}]]},
            {"text": "Cashflow Statement", "rows": [[{"text": "Depreciation"}]]},
        ],
        model_id="presentation-model",
        timeout_s=12,
    )

    assert result == {"CF": "INDIRECT"}


def test_infer_presentations_rejects_conflicting_duplicate_decisions(monkeypatch) -> None:
    report_list = [
        _report(
            "CF",
            [
                _subject("cf-direct", "Direct method", "DIRECT"),
                _subject("cf-indirect", "Indirect method", "INDIRECT"),
            ],
        )
    ]

    async def fake_invoke(**_kwargs):
        return PresentationDecisionResponse(
            items=[
                PresentationDecision(report_type="CF", presentation="DIRECT"),
                PresentationDecision(report_type="CF", presentation="INDIRECT"),
            ]
        )

    monkeypatch.setattr(
        presentation_module,
        "ainvoke_llm_with_structured_output",
        fake_invoke,
    )

    with pytest.raises(ValueError, match="列报方式判断冲突.*DIRECT.*INDIRECT"):
        infer_report_presentations(
            report_list,
            [{"text": "Cashflow Statement", "rows": [[{"text": "Cash received"}]]}],
            model_id="presentation-model",
            timeout_s=12,
        )


def test_infer_presentations_surfaces_model_failure(monkeypatch) -> None:
    report_list = [
        _report(
            "CF",
            [
                _subject("cf-direct", "Direct method", "DIRECT"),
                _subject("cf-indirect", "Indirect method", "INDIRECT"),
            ],
        )
    ]

    async def fail_invoke(**_kwargs):
        raise TimeoutError("model timeout")

    monkeypatch.setattr(
        presentation_module,
        "ainvoke_llm_with_structured_output",
        fail_invoke,
    )

    with pytest.raises(RuntimeError, match="列报方式判断失败"):
        infer_report_presentations(
            report_list,
            [{"text": "Cashflow Statement", "rows": [[{"text": "Profit before tax"}]]}],
            model_id="presentation-model",
            timeout_s=12,
        )


def test_load_local_report_list_filters_presentation_and_relations(tmp_path: Path) -> None:
    template = {
        "reportList": [
            _report(
                "BS",
                [
                    _subject(
                        "bs-current-root",
                        "Current/non-current",
                        "CURT_NONCURT",
                        children=["bs-current-child", "bs-liquidity-root"],
                    ),
                    _subject("bs-current-child", "Current assets", "CURT_NONCURT"),
                    _subject("bs-liquidity-root", "Liquidity order", "LIQ_ORDER"),
                ],
            ),
            _report("IS", [_subject("is-function", "By function", "BY_FUNC")]),
        ]
    }
    template_file = tmp_path / "template.json"
    template_file.write_text(json.dumps(template), encoding="utf-8")

    reports = load_local_report_list(
        tmp_path,
        {
            "subject_match_english_template_file": template_file.name,
            "subject_match_template_report_calibers": [2],
        },
        template_config_key="subject_match_english_template_file",
        presentation_by_report_type={"BS": "CURT_NONCURT"},
    )

    assert len(reports) == 1
    assert [item["subjectId"] for item in reports[0]["subjectList"]] == [
        "bs-current-root",
        "bs-current-child",
    ]
    assert reports[0]["subjectList"][0]["summaryRelation"] == ["bs-current-child"]
    assert "presentation" not in reports[0]["subjectList"][0]


def test_ifrs_template_contains_all_supported_presentation_trees() -> None:
    project_root = Path(__file__).resolve().parents[2]
    regression_root = project_root / "regression_mvp"
    api_config = json.loads((regression_root / "config.json").read_text(encoding="utf-8"))[
        "api"
    ]
    report_list = load_raw_local_report_list(
        regression_root,
        api_config,
        template_config_key="subject_match_english_template_file",
    )

    assert build_presentation_options(report_list, {"BS", "IS", "CF"}) == {
        "BS": [
            ("CURT_NONCURT", "Statement of financial position, current/non-current"),
            ("LIQ_ORDER", "Statement of financial position, liquidity order"),
        ],
        "CF": [
            ("DIRECT", "Statement of cash flows (direct method)"),
            ("INDIRECT", "Statement of cash flows (indirect method)"),
            (
                "INDIRECT_FI",
                "Statement of cash flows (indirect method - extended FI)",
            ),
        ],
        "IS": [
            ("BY_FUNC", "Statement of comprehensive income, by function of expense"),
            ("BY_NATURE", "Statement of comprehensive income, by nature of expense"),
        ],
    }


def test_chinese_local_template_does_not_call_presentation_classifier(
    monkeypatch,
    tmp_path: Path,
) -> None:
    template_file = tmp_path / "zh.json"
    template_file.write_text(
        json.dumps(
            {
                "reportList": [
                    _report("BS", [_subject("cash", "货币资金", "unused")])
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    api_config = {
        "subject_match_template_source": "local_file",
        "subject_match_template_file": template_file.name,
        "subject_match_template_report_calibers": [2],
    }

    def reject_classifier(*_args, **_kwargs):
        raise AssertionError("Chinese statements must not call the presentation classifier")

    monkeypatch.setattr(
        runner_module,
        "infer_report_presentations",
        reject_classifier,
    )

    reports = runner_module._load_report_list(
        object(),
        tmp_path,
        api_config,
        org_tables=[
            {"text": "资产负债表", "rows": [[{"text": "货币资金"}, {"text": "100"}]]}
        ],
    )

    assert reports[0]["subjectList"][0]["subjectName"] == "货币资金"
