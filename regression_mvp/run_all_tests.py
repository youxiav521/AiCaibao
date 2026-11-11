from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ROOT_DIR = Path(__file__).resolve().parent

# =============================================================================
# PyCharm Run 配置区
#
# 直接运行本文件即可执行当前已经接入的所有接口回归测试。
# 如果只想临时跑某几个接口，把对应路径从 TEST_TARGETS 中删掉即可。
# =============================================================================
TEST_TARGETS = [
    ROOT_DIR / "tests" / "test_file_read.py",
    ROOT_DIR / "tests" / "test_mineru_to_json.py",
    ROOT_DIR / "tests" / "test_mineru_upload_async.py",
    ROOT_DIR / "tests" / "test_ade_jobs_async.py",
    ROOT_DIR / "tests" / "test_financial_metadata.py",
    ROOT_DIR / "tests" / "test_financial_three_statements.py",
    ROOT_DIR / "tests" / "test_financial_org_tables.py",
    ROOT_DIR / "tests" / "test_financial_merge_tables.py",
    ROOT_DIR / "tests" / "test_financial_subject_match.py",
]

PYTEST_ARGS = [
    "-v",
]


def main() -> int:
    args = [str(path) for path in TEST_TARGETS] + PYTEST_ARGS
    return pytest.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
