from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RegressionCase:
    case_name: str
    endpoint_type: str
    input_file: Path
    expected_file: Path
    folder_id: str | None = None
    input_sha256: str | None = None
    description: str = ""


def load_enabled_cases(
    root_dir: Path,
    config: dict[str, Any],
    *,
    endpoint_type: str | None = None,
) -> list[RegressionCase]:
    case_dir = root_dir / config["paths"]["case_dir"]
    if not case_dir.exists():
        return []
    selected_case_name = str(os.environ.get("REGRESSION_MVP_CASE_NAME") or "").strip()

    cases: list[RegressionCase] = []
    for config_path in sorted(case_dir.glob("**/case.yaml")):
        raw = _load_simple_yaml(config_path)
        lifecycle_status = str(raw.get("lifecycle_status") or "active").strip().lower()
        if lifecycle_status == "deleted":
            continue
        enabled = str(raw.get("enabled", "true")).strip().lower() in {"true", "1", "yes", "on"}
        if not enabled:
            continue

        resolved_endpoint_type = _resolve_endpoint_type(case_dir, config_path, raw)
        if endpoint_type and resolved_endpoint_type != endpoint_type:
            continue

        case_name = str(raw.get("case_name") or config_path.parent.name)
        if selected_case_name and case_name != selected_case_name and config_path.parent.name != selected_case_name:
            continue
        input_file = _resolve_case_path(config_path.parent, raw.get("input_file") or raw.get("file_name"))
        expected_file = _resolve_case_path(config_path.parent, raw.get("expected_file") or "expected_items.json")
        folder_id = raw.get("folder_id")
        input_sha256 = raw.get("input_sha256")
        cases.append(
            RegressionCase(
                case_name=case_name,
                endpoint_type=resolved_endpoint_type,
                input_file=input_file,
                expected_file=expected_file,
                folder_id=str(folder_id) if folder_id else None,
                input_sha256=str(input_sha256).lower() if input_sha256 else None,
                description=str(raw.get("description") or ""),
            )
        )
    return cases


def _resolve_endpoint_type(case_dir: Path, config_path: Path, raw: dict[str, str]) -> str:
    if raw.get("endpoint_type"):
        return str(raw["endpoint_type"]).strip()

    try:
        relative_parent = config_path.parent.relative_to(case_dir)
    except ValueError:
        return "file_read"

    parts = relative_parent.parts
    if len(parts) >= 2:
        return parts[0]
    return "file_read"


def _resolve_case_path(case_dir: Path, value: Any) -> Path:
    if not value:
        raise ValueError(f"missing path value in case: {case_dir}")
    path = Path(str(value))
    if not path.is_absolute():
        path = case_dir / path
    return path.resolve()


def _load_simple_yaml(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip('"').strip("'")
        result[key.strip()] = value
    return result
