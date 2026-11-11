import json
import re
from pathlib import Path
from typing import Any, cast

from src.ade.knowledge_graph.evaluate.schema import (
    ConfidenceLevel,
    EvaluationEntity,
    EvaluationGraph,
    EvaluationRelation,
    GraphFormat,
)

_DATA_ROW_RE = re.compile(r"^\s*(\d+)\s+(.+)$")
_SECTION_ENTITIES = "ENTITIES"
_SECTION_RELATIONS = "RELATIONS"
_CONFIDENCE_RANK = {"Low": 0, "Medium": 1, "High": 2}


def parse_graph(path: Path, file_format: GraphFormat = "auto") -> EvaluationGraph:
    fmt = _resolve_format(path, file_format)
    if fmt == "txt":
        return parse_txt_graph(path)
    if fmt == "json":
        return parse_json_graph(path)
    raise ValueError(f"不支持的文件格式: {file_format}")


def parse_txt_graph(path: Path) -> EvaluationGraph:
    warnings: list[str] = []
    entity_lines: list[tuple[int, str]] = []
    relation_lines: list[tuple[int, str]] = []
    section: str | None = None
    entity_has_confidence = False
    relation_has_confidence = False
    relation_has_attributes = False

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if _SECTION_ENTITIES in stripped:
            section = "entities"
            continue
        if _SECTION_RELATIONS in stripped:
            section = "relations"
            continue
        if section is None:
            continue
        if "序号" in stripped:
            if section == "entities":
                entity_has_confidence = "confidence" in stripped
            elif section == "relations":
                relation_has_confidence = "confidence" in stripped
                relation_has_attributes = "attributes" in stripped
            continue
        if stripped.startswith("━") or stripped.startswith("="):
            continue
        if section == "entities":
            if _DATA_ROW_RE.match(line):
                entity_lines.append((line_no, line))
            else:
                warnings.append(f"实体段未识别行 | line={line_no}: {stripped}")
        elif section == "relations":
            if _DATA_ROW_RE.match(line):
                relation_lines.append((line_no, line))
            else:
                warnings.append(f"关系段未识别行 | line={line_no}: {stripped}")

    return EvaluationGraph(
        entities=_parse_txt_entities(entity_lines, warnings, entity_has_confidence),
        relations=_parse_txt_relations(
            relation_lines,
            warnings,
            relation_has_attributes,
            relation_has_confidence,
        ),
        warnings=warnings,
    )


def parse_json_graph(path: Path) -> EvaluationGraph:
    warnings: list[str] = []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return EvaluationGraph(warnings=[f"JSON 根节点必须是对象: {path}"])

    entities_raw = data.get("entities")
    relations_raw = data.get("relations")
    if not isinstance(entities_raw, list):
        warnings.append("JSON 缺少 entities 数组")
        entities_raw = []
    if not isinstance(relations_raw, list):
        warnings.append("JSON 缺少 relations 数组")
        relations_raw = []

    entities: list[EvaluationEntity] = []
    for idx, raw in enumerate(entities_raw, start=1):
        if not isinstance(raw, dict):
            warnings.append(f"JSON 实体不是对象 | index={idx}")
            continue
        missing = [key for key in ("name", "type", "type_en") if not raw.get(key)]
        if missing:
            warnings.append(f"JSON 实体缺少字段 {missing} | index={idx}")
            continue
        attrs = raw.get("attributes") or {}
        if not isinstance(attrs, dict):
            warnings.append(f"JSON 实体 attributes 不是对象，已忽略 | index={idx}")
            attrs = {}
        entities.append(
            EvaluationEntity(
                name=str(raw["name"]),
                type=str(raw["type"]),
                type_en=str(raw["type_en"]),
                attributes={str(k): str(v) for k, v in attrs.items()},
                confidence=_json_confidence(raw, warnings, f"JSON 实体 index={idx}"),
            )
        )

    relations: list[EvaluationRelation] = []
    for idx, raw in enumerate(relations_raw, start=1):
        if not isinstance(raw, dict):
            warnings.append(f"JSON 关系不是对象 | index={idx}")
            continue
        missing = [key for key in ("from_entity", "type", "to_entity") if not raw.get(key)]
        if missing:
            warnings.append(f"JSON 关系缺少字段 {missing} | index={idx}")
            continue
        relations.append(
            EvaluationRelation(
                from_entity=str(raw["from_entity"]),
                type=str(raw["type"]),
                type_en=str(raw.get("type_en") or ""),
                to_entity=str(raw["to_entity"]),
                attributes=_normalize_relation_attributes(
                    raw.get("attributes"),
                    warnings,
                    f"JSON 关系 index={idx}",
                ),
                confidence=_json_confidence(raw, warnings, f"JSON 关系 index={idx}"),
            )
        )

    return EvaluationGraph(entities=entities, relations=relations, warnings=warnings)


def parse_attributes(text: str) -> dict[str, str]:
    attributes: dict[str, str] = {}
    if not text or not text.strip():
        return attributes
    for part in text.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, _, value = part.partition("=")
        key, value = key.strip(), value.strip()
        if key:
            attributes[key] = value
    return attributes


def parse_relation_attributes(text: str) -> dict[str, list[str]]:
    attributes: dict[str, list[str]] = {}
    if not text or not text.strip():
        return attributes
    for part in text.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, _, value = part.partition("=")
        key, value = key.strip(), value.strip()
        if not key or not value:
            continue
        values = attributes.setdefault(key, [])
        if value not in values:
            values.append(value)
    return attributes


def _normalize_relation_attributes(
    raw: Any,
    warnings: list[str],
    context: str,
) -> dict[str, list[str]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        warnings.append(f"关系 attributes 不是对象，已忽略 | {context}")
        return {}

    attributes: dict[str, list[str]] = {}
    for raw_key, raw_value in raw.items():
        key = str(raw_key).strip()
        if not key:
            continue
        raw_values = raw_value if isinstance(raw_value, list) else [raw_value]
        values = attributes.setdefault(key, [])
        for raw_item in raw_values:
            if raw_item is None or isinstance(raw_item, (dict, list)):
                continue
            value = str(raw_item).strip()
            if value and value not in values:
                values.append(value)
        if not values:
            attributes.pop(key)
    return attributes


def _resolve_format(path: Path, file_format: GraphFormat) -> str:
    if file_format != "auto":
        return file_format
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return "txt"
    if suffix == ".json":
        return "json"
    raise ValueError(f"无法根据后缀识别文件格式: {path}")


def _split_columns(line_body: str) -> list[str]:
    return [col.strip() for col in re.split(r"\s{2,}", line_body.strip()) if col.strip()]


def _parse_txt_entities(
    lines: list[tuple[int, str]],
    warnings: list[str],
    has_confidence: bool,
) -> list[EvaluationEntity]:
    entities: list[EvaluationEntity] = []
    for line_no, line in lines:
        match = _DATA_ROW_RE.match(line)
        if not match:
            continue
        columns = _split_columns(match.group(2))
        if len(columns) < 3:
            warnings.append(f"实体行解析失败（列数不足）| line={line_no}: {line.strip()}")
            continue
        confidence = None
        if has_confidence and len(columns) > 3 and (
            columns[-1] in _CONFIDENCE_RANK or "=" not in columns[-1]
        ):
            confidence = _parse_confidence(
                columns.pop(), warnings, f"TXT 实体 line={line_no}"
            )
        entities.append(
            EvaluationEntity(
                name=columns[0],
                type=columns[1],
                type_en=columns[2],
                attributes=parse_attributes(columns[3] if len(columns) > 3 else ""),
                confidence=confidence,
            )
        )
    return entities


def _parse_txt_relations(
    lines: list[tuple[int, str]],
    warnings: list[str],
    has_attributes: bool,
    has_confidence: bool,
) -> list[EvaluationRelation]:
    relations: list[EvaluationRelation] = []
    for line_no, line in lines:
        match = _DATA_ROW_RE.match(line)
        if not match:
            continue
        columns = _split_columns(match.group(2))
        if len(columns) < 4:
            warnings.append(f"关系行解析失败（列数不足）| line={line_no}: {line.strip()}")
            continue
        confidence = None
        if has_confidence and len(columns) > 4 and (
            columns[-1] in _CONFIDENCE_RANK or "=" not in columns[-1]
        ):
            confidence = _parse_confidence(
                columns.pop(), warnings, f"TXT 关系 line={line_no}"
            )
        attributes = (
            parse_relation_attributes(columns[4])
            if has_attributes and len(columns) > 4
            else {}
        )
        relations.append(
            EvaluationRelation(
                from_entity=columns[0],
                type=columns[1],
                type_en=columns[2],
                to_entity=columns[3],
                attributes=attributes,
                confidence=confidence,
            )
        )
    return relations


def _json_confidence(
    raw: dict[str, Any], warnings: list[str], context: str
) -> ConfidenceLevel | None:
    if "confidence" in raw:
        return _parse_confidence(raw.get("confidence"), warnings, context)
    references = raw.get("references")
    if not isinstance(references, list):
        return None
    valid = [
        reference.get("confidence")
        for reference in references
        if isinstance(reference, dict)
        and reference.get("confidence") in _CONFIDENCE_RANK
    ]
    return cast(
        ConfidenceLevel | None,
        max(valid, key=_CONFIDENCE_RANK.get, default=None),
    )


def _parse_confidence(
    raw: Any, warnings: list[str], context: str
) -> ConfidenceLevel | None:
    if raw is None or not str(raw).strip():
        return None
    value = str(raw).strip()
    if value not in _CONFIDENCE_RANK:
        warnings.append(f"置信度无效，已按未标注处理 | {context}: {value}")
        return None
    return cast(ConfidenceLevel, value)
