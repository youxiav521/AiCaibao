import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _get_first(data: dict[str, Any], *keys: str, default=None):
    """Return first existing non-null value from candidate keys."""
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def _as_list(value: Any) -> list[Any]:
    """Return list value or empty list for shape-safe iteration."""
    if isinstance(value, list):
        return value
    return []


def _normalize_element_ids(value: Any) -> list[int]:
    """Normalize multiple alias shapes into a list of integer element ids."""
    if value is None or value == "":
        return []
    if isinstance(value, list):
        normalized: list[int] = []
        for item in value:
            if isinstance(item, int):
                normalized.append(item)
            elif isinstance(item, str) and item.isdigit():
                normalized.append(int(item))
        return normalized
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",")]
        return [int(token) for token in tokens if token.isdigit()]
    return []


def _normalize_requirement(requirement: dict[str, Any]) -> dict[str, Any]:
    """Normalize requirement object across alias variants."""
    source_location = _get_first(requirement, "sourceLocation", default={}) or {}
    if not isinstance(source_location, dict):
        source_location = {}

    return {
        "requirement_id": _get_first(requirement, "requirementId", "requirement_id", default=""),
        "title": _get_first(requirement, "title", default=""),
        "description": _get_first(requirement, "description", default=""),
        "source_text": _get_first(requirement, "sourceText", "source_text", default=""),
        "file_id": _get_first(source_location, "fileId", "file_id", default=""),
        "page_no": _get_first(source_location, "pageNo", "page_no", "page", default=None),
        "element_ids": _normalize_element_ids(
            _get_first(source_location, "elementIds", "elementId", "element_id", "chunkId", "chunk_id", default=[])
        ),
        "controls": _as_list(requirement.get("controls")),
    }


def _normalize_control(control: dict[str, Any]) -> dict[str, Any]:
    """Normalize control object across alias variants."""
    source_location = _get_first(control, "sourceLocation", default={}) or {}
    if not isinstance(source_location, dict):
        source_location = {}

    return {
        "control_id": _get_first(control, "controlId", "control_id", default=""),
        "content": _get_first(control, "content", "statement", "title", default=""),
        "control_type": _get_first(control, "controlType", "control_type", default=""),
        "risk": _get_first(control, "risk", default=""),
        "source_text": _get_first(control, "sourceText", "source_text", default=""),
        "file_id": _get_first(source_location, "fileId", "file_id", default=""),
        "page_no": _get_first(source_location, "pageNo", "page_no", "page", default=None),
        "element_ids": _normalize_element_ids(
            _get_first(source_location, "elementIds", "elementId", "element_id", "chunkId", "chunk_id", default=[])
        ),
    }


def _section_key(requirement: dict[str, Any]) -> str:
    """Build section key used for per-section summary aggregation."""
    return requirement["file_id"] or "UNKNOWN_FILE"


def build_review_summary(payload: dict[str, Any], source_file: str) -> dict[str, Any]:
    """Build quality/review summary from framework extraction output."""
    requirements = [_normalize_requirement(item) for item in _as_list(payload.get("requirements")) if isinstance(item, dict)]

    framework_name = _get_first(payload, "name", "document_title", default="")
    framework_id = _get_first(payload, "frameworkId", "framework_id", default="")
    file_id = _get_first(payload, "fileId", "file_id", "document_id", default="")
    status = _get_first(payload, "status", default="")
    description = _get_first(payload, "description", "summary", default="")

    section_map: dict[str, dict[str, Any]] = {}
    control_type_counter: Counter[str] = Counter()
    risk_counter: Counter[str] = Counter()
    requirement_title_counter: Counter[str] = Counter()
    control_title_counter: Counter[str] = Counter()
    suspicious_items: list[dict[str, Any]] = []

    for requirement in requirements:
        section_name = _section_key(requirement)
        section_entry = section_map.setdefault(
            section_name,
            {
                "section_title": section_name,
                "element_ids": set(),
                "requirement_count": 0,
                "control_count": 0,
                "requirement_previews": [],
            },
        )

        section_entry["requirement_count"] += 1
        for element_id in requirement["element_ids"]:
            section_entry["element_ids"].add(element_id)
        if len(section_entry["requirement_previews"]) < 3:
            section_entry["requirement_previews"].append(
                {
                    "requirementId": requirement["requirement_id"],
                    "title": requirement["title"],
                    "description": requirement["description"],
                    "controlCount": len(requirement["controls"]),
                }
            )

        if not requirement["controls"]:
            # Flag requirements that are likely incomplete for rule execution.
            suspicious_items.append(
                {
                    "type": "requirement_without_controls",
                    "requirementId": requirement["requirement_id"],
                    "title": requirement["title"],
                    "section": section_name,
                }
            )

        requirement_title_counter[requirement["title"]] += 1

        seen_element_ids = set()
        for control in requirement["controls"]:
            if not isinstance(control, dict):
                continue
            normalized_control = _normalize_control(control)
            section_entry["control_count"] += 1
            for element_id in normalized_control["element_ids"]:
                section_entry["element_ids"].add(element_id)

            if normalized_control["control_type"]:
                control_type_counter[normalized_control["control_type"]] += 1
            if normalized_control["risk"]:
                risk_counter[normalized_control["risk"]] += 1
            if normalized_control["content"]:
                control_title_counter[normalized_control["content"]] += 1

            content = normalized_control["content"].strip()
            if len(content) >= 120:
                # Long control text often indicates over-merged extraction output.
                suspicious_items.append(
                    {
                        "type": "long_control_content",
                        "controlId": normalized_control["control_id"],
                        "content": normalized_control["content"],
                        "length": len(content),
                        "section": section_name,
                    }
                )

            duplicate_element_ids = [
                element_id for element_id in normalized_control["element_ids"] if element_id in seen_element_ids
            ]
            if duplicate_element_ids:
                # Multiple controls mapped to the same source elements may indicate duplication.
                suspicious_items.append(
                    {
                        "type": "duplicate_control_source_element",
                        "controlId": normalized_control["control_id"],
                        "content": normalized_control["content"],
                        "section": section_name,
                        "elementIds": duplicate_element_ids,
                    }
                )
            for element_id in normalized_control["element_ids"]:
                seen_element_ids.add(element_id)

    for title, count in requirement_title_counter.items():
        if title and count > 1:
            suspicious_items.append(
                {
                    "type": "duplicate_requirement_title",
                    "title": title,
                    "count": count,
                }
            )

    for title, count in control_title_counter.items():
        if title and count > 1:
            suspicious_items.append(
                {
                    "type": "duplicate_control_content",
                    "content": title,
                    "count": count,
                }
            )

    sections = []
    for section_name, section in sorted(section_map.items(), key=lambda item: item[0]):
        sections.append(
            {
                "sectionName": section_name,
                "sectionTitle": section["section_title"],
                "elementCount": len(section["element_ids"]),
                "requirementCount": section["requirement_count"],
                "controlCount": section["control_count"],
                "requirementPreviews": section["requirement_previews"],
            }
        )

    total_controls = sum(len(requirement["controls"]) for requirement in requirements)
    empty_sections = [section["sectionName"] for section in sections if section["controlCount"] == 0]

    return {
        "sourceFile": source_file,
        "framework": {
            "frameworkId": framework_id,
            "fileId": file_id,
            "name": framework_name,
            "status": status,
            "description": description,
        },
        "overview": {
            "sectionCount": len(sections),
            "requirementCount": len(requirements),
            "controlCount": total_controls,
            "requirementsWithoutControls": sum(1 for item in requirements if not item["controls"]),
            "emptySectionCount": len(empty_sections),
        },
        "distributions": {
            "controlTypes": dict(sorted(control_type_counter.items())),
            "riskLevels": dict(sorted(risk_counter.items())),
        },
        "emptySections": empty_sections,
        "sections": sections,
        "suspiciousItems": suspicious_items,
    }


def default_output_path(input_path: str | Path) -> Path:
    """Return default summary output path for a given input path."""
    input_file = Path(input_path)
    if input_file.suffix.lower() == ".json":
        return input_file.with_name(f"{input_file.stem}_review_summary.json")
    return input_file / "review_summary.json"


def main() -> None:
    """CLI entry point for generating review summary from framework output."""
    parser = argparse.ArgumentParser(description="Analyze policy extractor output and generate a review summary.")
    parser.add_argument("--input", required=True, help="Path to framework output json.")
    parser.add_argument("--output", default=None, help="Optional review summary output path.")
    args = parser.parse_args()

    input_path = Path(args.input)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    summary = build_review_summary(payload, str(input_path))

    output_path = Path(args.output) if args.output else default_output_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved review summary to: {output_path}")
    print(json.dumps(summary["overview"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
