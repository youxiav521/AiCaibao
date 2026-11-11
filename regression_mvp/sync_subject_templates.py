from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regression_mvp.utils.api_clients.subject_match_client import SubjectMatchApiClient
from regression_mvp.utils.json_io import dump_json, load_json


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.json"


def main() -> int:
    config = load_json(CONFIG_PATH)
    api_config = config["api"]
    output_path = _template_file_path(api_config)

    try:
        templates = SubjectMatchApiClient(api_config).get_templates()
    except Exception as exc:
        print("同步失败：无法从 /api/parsing/subject-match/templates 获取标准科目模板。")
        print(f"原因：{exc}")
        print("请先确认业务服务的 evaluator/subject_templates.json 已生成并且接口可访问。")
        return 1

    real_templates = [
        item for item in templates
        if isinstance(item, dict) and item.get("reportId") != "__placeholder__"
    ]
    if not real_templates:
        print("同步失败：接口返回为空或只包含 placeholder 模板。")
        return 1

    dump_json(output_path, real_templates)
    subject_count = sum(
        len(item.get("subjectList") or [])
        for item in real_templates
        if isinstance(item, dict)
    )
    print(f"同步成功：{output_path}")
    print(f"报表模板数：{len(real_templates)}")
    print(f"标准科目数：{subject_count}")
    return 0


def _template_file_path(api_config: dict[str, Any]) -> Path:
    path = Path(str(api_config.get("subject_match_template_file") or "templates/subject_match_report_list.json"))
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


if __name__ == "__main__":
    raise SystemExit(main())
