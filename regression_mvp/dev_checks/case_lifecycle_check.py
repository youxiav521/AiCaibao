from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regression_mvp.utils.case_loader import load_enabled_cases
from regression_mvp.web_app import case_actions, data


ROOT_DIR = PROJECT_ROOT / "regression_mvp"
CONFIG = {"paths": {"case_dir": "cases"}}


def main() -> None:
    case_dir = ROOT_DIR / "cases" / "file_read" / "codex_lifecycle_check"
    legacy_case_dir = ROOT_DIR / "cases" / "file_read" / "codex_lifecycle_legacy_dir"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    if legacy_case_dir.exists():
        shutil.rmtree(legacy_case_dir)
    case_dir.mkdir(parents=True)
    legacy_case_dir.mkdir(parents=True)
    try:
        (case_dir / "case.yaml").write_text(
            "\n".join(
                [
                    'case_name: "codex_lifecycle_check"',
                    'endpoint_type: "file_read"',
                    'enabled: "true"',
                    'input_file: "input.pdf"',
                    'expected_file: "expected_items.json"',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (case_dir / "input.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
        (case_dir / "expected_items.json").write_text("[]\n", encoding="utf-8")

        assert any(case.case_name == "codex_lifecycle_check" for case in load_enabled_cases(ROOT_DIR, CONFIG))
        case_actions.archive_case("file_read", "codex_lifecycle_check")
        assert not any(case.case_name == "codex_lifecycle_check" for case in load_enabled_cases(ROOT_DIR, CONFIG))
        deleted_rows = [row for row in data.all_cases() if row["case_name"] == "codex_lifecycle_check"]
        assert deleted_rows and deleted_rows[0]["deleted"]
        case_actions.restore_case("file_read", "codex_lifecycle_check")
        assert any(case.case_name == "codex_lifecycle_check" for case in load_enabled_cases(ROOT_DIR, CONFIG))

        (legacy_case_dir / "case.yaml").write_text(
            "\n".join(
                [
                    'case_name: "codex_lifecycle_legacy_name"',
                    'endpoint_type: "file_read"',
                    'enabled: "true"',
                    'input_file: "input.pdf"',
                    'expected_file: "expected_items.json"',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (legacy_case_dir / "input.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
        (legacy_case_dir / "expected_items.json").write_text("[]\n", encoding="utf-8")

        case_actions.archive_case("file_read", "codex_lifecycle_legacy_name")
        deleted_legacy_rows = [row for row in data.all_cases() if row["case_name"] == "codex_lifecycle_legacy_name"]
        assert deleted_legacy_rows and deleted_legacy_rows[0]["deleted"]
        case_actions.restore_case("file_read", "codex_lifecycle_legacy_name")
        assert any(case.case_name == "codex_lifecycle_legacy_name" for case in load_enabled_cases(ROOT_DIR, CONFIG))
    finally:
        if case_dir.exists():
            shutil.rmtree(case_dir, ignore_errors=True)
        if legacy_case_dir.exists():
            shutil.rmtree(legacy_case_dir, ignore_errors=True)

    print("case_lifecycle_check passed")


if __name__ == "__main__":
    main()
