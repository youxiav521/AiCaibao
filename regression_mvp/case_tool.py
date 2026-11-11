from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regression_mvp.utils.case_manager import generate_candidate, promote_candidate
from regression_mvp.utils.json_io import load_json
from regression_mvp.storage import minio_sync


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.json"

# =============================================================================
# PyCharm Run 配置区
#
# 不想写命令行时，直接改下面这些值，然后在 PyCharm 里运行本文件。
# 当脚本没有收到命令行参数时，会使用这里的配置。
# 当脚本收到 generate/promote 等命令行参数时，会忽略这里的配置。
# =============================================================================
RUN_FROM_CONFIG = True

# generate: 从本地文件生成候选用例
# promote: 将人工确认后的候选用例固化到 cases/
ACTION = "promote"

# file_read /
# mineru_to_json /
# mineru_upload_async /
# ade_jobs_async /
# financial_metadata /
# financial_three_statements /
# financial_merge_tables /
# financial_subject_match
GENERATE_ENDPOINT = "financial_merge_tables"
GENERATE_INPUT = r"C:\Users\23829\Documents\WXWork\1688856991780343\Cache\File\2026-06\OQEP\OQEP_2024_Yearly (Audited)_e\3_OQEP_IncomeStatement_27022025_52_U.pdf"
GENERATE_CASE_NAME = "case_001"
GENERATE_DESCRIPTION = "Sample PDF for {} regression".format(GENERATE_ENDPOINT)

PROMOTE_CANDIDATE = r"D:\dev\yusys_ai\regression_mvp\case_candidates\20260622_171137_case_001"
PROMOTE_FORCE = True


def main() -> None:
    if len(sys.argv) == 1 and RUN_FROM_CONFIG:
        _run_from_top_config()
        return

    parser = argparse.ArgumentParser(description="Manage regression test cases.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate a candidate case from an input file.")
    generate_parser.add_argument(
        "--endpoint",
        default="file_read",
        choices=[
            "file_read",
            "mineru_to_json",
            "mineru_upload_async",
            "ade_jobs_async",
            "financial_metadata",
            "financial_three_statements",
            "financial_org_tables",
            "financial_merge_tables",
            "financial_subject_match",
        ],
    )
    generate_parser.add_argument("--input", required=True, help="Path to the input file.")
    generate_parser.add_argument("--case-name", required=True, help="Case name, for example case_002_file_read.")
    generate_parser.add_argument("--description", default="", help="Human-readable case description.")

    promote_parser = subparsers.add_parser("promote", help="Promote a reviewed candidate into formal cases/.")
    promote_parser.add_argument("--candidate", required=True, help="Path to a candidate directory.")
    promote_parser.add_argument("--force", action="store_true", help="Overwrite an existing formal case directory.")

    args = parser.parse_args()
    config = load_json(CONFIG_PATH)

    if args.command == "generate":
        _generate(config, args.endpoint, Path(args.input), args.case_name, args.description)
        return

    if args.command == "promote":
        _promote(Path(args.candidate), force=args.force)
        return


def _run_from_top_config() -> None:
    config = load_json(CONFIG_PATH)
    action = ACTION.strip().lower()
    if action == "generate":
        _generate(config, GENERATE_ENDPOINT, Path(GENERATE_INPUT), GENERATE_CASE_NAME, GENERATE_DESCRIPTION)
        return
    if action == "promote":
        _promote(Path(PROMOTE_CANDIDATE), force=PROMOTE_FORCE)
        return
    raise ValueError("ACTION must be 'generate' or 'promote'")


def _generate(config: dict[str, Any], endpoint: str, input_file: Path, case_name: str, description: str) -> None:
    candidate_dir = generate_candidate(
        ROOT_DIR,
        config,
        endpoint,
        input_file,
        case_name,
        description,
    )
    print("candidate created:")
    print(candidate_dir)
    print("")
    print("open these files for review:")
    for input_path in sorted(candidate_dir.glob("input.*")):
        print(input_path)
    print(candidate_dir / "preview.html")
    overlay = candidate_dir / "overlay.html"
    if overlay.exists():
        print(overlay)
    print(candidate_dir / "metrics.json")
    _try_upload_candidate(candidate_dir)


def _promote(candidate_dir: Path, *, force: bool = False) -> None:
    case_dir = promote_candidate(ROOT_DIR, candidate_dir, force=force)
    _try_upload_candidate(candidate_dir)
    _try_upload_case(case_dir)
    print("case promoted:")
    print(case_dir)


def _try_upload_candidate(candidate_dir: Path) -> None:
    try:
        minio_sync.upload_candidate_dir(candidate_dir)
    except Exception as exc:
        print(f"MinIO candidate sync failed: {exc}")


def _try_upload_case(case_dir: Path) -> None:
    try:
        minio_sync.upload_case_dir(case_dir)
    except Exception as exc:
        print(f"MinIO case sync failed: {exc}")


if __name__ == "__main__":
    main()
