from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regression_mvp.utils.case_exclusions import load_compare_exclusions
from regression_mvp.utils.json_io import dump_json
from regression_mvp.web_app import case_edit_actions


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        cases_dir = Path(temp_dir) / "cases"
        case_dir = cases_dir / "mineru_to_json" / "demo"
        case_dir.mkdir(parents=True)
        (case_dir / "case.yaml").write_text(
            'case_name: "demo"\nendpoint_type: "mineru_to_json"\nexpected_file: "expected_items.json"\n',
            encoding="utf-8",
        )
        dump_json(case_dir / "expected_items.json", [])
        form = {
            "page": "2",
            "item_type": "image",
            "bbox": '{"l":10,"t":20,"r":120,"b":180}',
            "scope": "item",
            "reason": "原始图片模糊，人工确认不作为稳定基线",
            "run_id": "demo_run",
            "diff_path": "items[3]",
        }
        with patch.object(case_edit_actions, "CASES_DIR", cases_dir), patch.object(
            case_edit_actions,
            "_try_upload_case",
            lambda _: None,
        ):
            rule = case_edit_actions.save_compare_exclusion("mineru_to_json", "demo", form)

        rules = load_compare_exclusions(case_dir)
        assert len(rules) == 1
        assert rules[0]["id"] == rule["id"]
        assert rules[0]["selector"]["page"] == 2
        assert rules[0]["selector"]["bbox"]["r"] == 120.0
        assert 'compare_exclusions_file: "compare_exclusions.json"' in (case_dir / "case.yaml").read_text(encoding="utf-8")
    print("case exclusion persistence checks passed")


if __name__ == "__main__":
    main()
