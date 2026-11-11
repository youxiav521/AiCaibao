from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regression_mvp.utils.mineru_assessment import assess_mineru_regression


def check_identical_content_passes() -> None:
    items = [
        {"type": "text", "page": 1, "value": "Revenue was 100"},
        {"type": "table", "page": 1, "rows": [[{"text": "Revenue"}, {"text": "100"}]]},
    ]

    result = assess_mineru_regression(items, items, [])

    assert result["status"] == "passed"
    assert result["contract"]["passed"]
    assert result["content"]["passed"]


def check_missing_meaningful_page_fails_contract() -> None:
    expected = [
        {"type": "text", "page": 1, "value": "A" * 80},
        {"type": "text", "page": 2, "value": "B" * 80},
    ]
    actual = [{"type": "text", "page": 1, "value": "A" * 80}]

    result = assess_mineru_regression(expected, actual, [])

    assert result["status"] == "failed"
    assert any(item["code"] == "meaningful_page_missing" for item in result["contract"]["issues"])


def check_numeric_change_is_suspected_content_regression() -> None:
    expected = [{"type": "table", "page": 1, "rows": [[{"text": "Revenue"}, {"text": "100"}]]}]
    actual = [{"type": "table", "page": 1, "rows": [[{"text": "Revenue"}, {"text": "500"}]]}]
    diffs = [{"type": "financial_numeric_changed", "severity": "critical", "path": "items[0~0].rows[0~0][1].text"}]

    result = assess_mineru_regression(expected, actual, diffs)

    assert result["status"] == "suspected"
    assert result["should_fail"]


def check_text_split_is_warning_when_content_is_preserved() -> None:
    expected = [{"type": "text", "page": 1, "value": "Total current assets and liabilities"}]
    actual = [
        {"type": "text", "page": 1, "value": "Total current assets"},
        {"type": "text", "page": 1, "value": "and liabilities"},
    ]
    diffs = [{"type": "text_split_merged", "severity": "minor", "path": "items[0~0]"}]

    result = assess_mineru_regression(expected, actual, diffs)

    assert result["status"] == "warning"
    assert not result["should_fail"]


def main() -> None:
    check_identical_content_passes()
    check_missing_meaningful_page_fails_contract()
    check_numeric_change_is_suspected_content_regression()
    check_text_split_is_warning_when_content_is_preserved()
    print("mineru assessment checks passed")


if __name__ == "__main__":
    main()
