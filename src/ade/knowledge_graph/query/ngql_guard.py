"""Validation and anchor compilation for LLM-generated nGQL."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence

from src.ade.knowledge_graph.nebula.repository import _vid_literal

_START_KEYWORDS = frozenset({"GO", "MATCH", "LOOKUP", "FETCH", "FIND"})
_WRITE_KEYWORDS = frozenset(
    """
    INSERT UPDATE UPSERT DELETE DROP CREATE ALTER REBUILD SUBMIT
    ADD REMOVE GRANT REVOKE CHANGE BALANCE COMPACT FLUSH SIGN
    INGEST DOWNLOAD USE SHOW SPACE HOSTS CONFIGS SNAPSHOT SESSION JOB
    ROLE USER PASSWORD KILL CLEAR STOP RECOVER RESTART RENAME OVERWRITE
    """.split()
)
_ANCHOR_PATTERN = re.compile(r"__ANCHOR_(\d+)_VIDS__")
_VID_PATTERN = re.compile(r"\bent_[0-9a-f]{16}\b")
_VALID_VID_PATTERN = re.compile(r"^ent_[0-9a-f]{16}$")
_STRING_PATTERN = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', re.DOTALL)
_FOLDER_SCOPE_PATTERN = re.compile(
    r"(?:`?[a-z][a-z0-9_]*`?\s*\.\s*)?`?folder_key`?\s*==\s*\$folder_key\b",
    re.IGNORECASE,
)
_AGGREGATE_PATTERN = re.compile(r"\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(", re.IGNORECASE)
_FINAL_LIMIT_PATTERN = re.compile(r"(?:\|\s*)?LIMIT\s+(\d+)\s*;?\s*$", re.IGNORECASE)
_INTERNAL_IDENTIFIER = (
    r"_(?:vid[a-z0-9_]*|edge\d+_(?:src|dst|type|rank))"
)
DEFAULT_MAX_HOPS = 10


class NGQLGuardError(ValueError):
    """Raised when generated nGQL violates the V3 query contract."""


def extract_statement(raw_output: str) -> str:
    """Extract the first nGQL statement from common LLM response wrappers."""
    fenced = re.search(
        r"```(?:ngql|nql|cypher)?\s*(.*?)```",
        raw_output,
        re.IGNORECASE | re.DOTALL,
    )
    candidate = fenced.group(1) if fenced else raw_output
    start = re.search(
        r"(?im)^[ \t]*(?:GO|MATCH|LOOKUP|FETCH|FIND)\b",
        candidate,
    )
    if start is None:
        raise NGQLGuardError("未找到可执行的 nGQL 语句")
    return candidate[start.start() :].strip()


def _without_comments(statement: str) -> str:
    statement = re.sub(r"/\*.*?\*/", " ", statement, flags=re.DOTALL)
    return re.sub(r"(?m)(?:--|//).*$", " ", statement)


def _normalize_internal_identifiers(statement: str) -> str:
    statement = re.sub(
        rf"\bAS\s+({_INTERNAL_IDENTIFIER})\b",
        r"AS `\1`",
        statement,
        flags=re.IGNORECASE,
    )
    return re.sub(
        rf"\$-\.\s*({_INTERNAL_IDENTIFIER})\b",
        r"$-.`\1`",
        statement,
        flags=re.IGNORECASE,
    )


def _single_statement(statement: str) -> str:
    cleaned = _without_comments(statement).strip()
    parts = [part.strip() for part in cleaned.split(";")]
    nonempty = [part for part in parts if part]
    if len(nonempty) != 1:
        raise NGQLGuardError("只允许一条非空 nGQL 语句")
    if len(parts) > 2 or (len(parts) == 2 and parts[1]):
        raise NGQLGuardError("只允许一条非空 nGQL 语句")
    return nonempty[0]


def _is_aggregate(statement: str) -> bool:
    return bool(_AGGREGATE_PATTERN.search(statement))


def _max_rows() -> int:
    try:
        value = int(os.environ.get("KG_NGQL_MAX_ROWS", "200"))
    except ValueError as error:
        raise NGQLGuardError("KG_NGQL_MAX_ROWS 必须是整数") from error
    if value <= 0:
        raise NGQLGuardError("KG_NGQL_MAX_ROWS 必须大于 0")
    return value


def _validate_shape(statement: str, max_hops: int) -> str:
    if max_hops <= 0:
        raise NGQLGuardError("最大查询跳数必须大于 0")
    cleaned = _single_statement(_normalize_internal_identifiers(statement))
    first = re.match(r"([A-Za-z]+)\b", cleaned)
    if first is None or first.group(1).upper() not in _START_KEYWORDS:
        raise NGQLGuardError("nGQL 首关键词不在只读白名单中")

    for keyword in _WRITE_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", cleaned, re.IGNORECASE):
            raise NGQLGuardError(f"禁止的 nGQL 关键词: {keyword}")

    for match in re.finditer(
        r"\bGO\s+(\d+)(?:\s+TO\s+(\d+))?\s+STEPS?\b",
        cleaned,
        re.IGNORECASE,
    ):
        if int(match.group(2) or match.group(1)) > max_hops:
            raise NGQLGuardError(f"查询跳数不得超过 {max_hops}")
    if first.group(1).upper() == "FIND":
        find_bound = re.search(
            r"\bUPTO\s+(\d+)\s+STEPS?\b",
            cleaned,
            re.IGNORECASE,
        )
        if find_bound is None:
            raise NGQLGuardError("FIND 路径查询必须包含 UPTO 最大跳数")
        if int(find_bound.group(1)) > max_hops:
            raise NGQLGuardError(f"FIND 查询跳数不得超过 {max_hops}")
    for match in re.finditer(r"\[[^\]]*\*([^\]]*)\]", cleaned):
        bound = match.group(1).strip()
        fixed = re.fullmatch(r"(\d+)", bound)
        ranged = re.fullmatch(r"(?:\d+)?\s*\.\.\s*(\d+)", bound)
        if fixed:
            match_max_hops = int(fixed.group(1))
        elif ranged:
            match_max_hops = int(ranged.group(1))
        else:
            raise NGQLGuardError("MATCH 变长路径必须包含最大跳数")
        if match_max_hops > max_hops:
            raise NGQLGuardError(f"MATCH 查询跳数不得超过 {max_hops}")
    if re.search(r"\bMATCH\b[^|;]*\)\s*,\s*\(", cleaned, re.IGNORECASE | re.DOTALL):
        raise NGQLGuardError("禁止无锚点的笛卡尔积 MATCH")
    if re.search(r"\battributes_json\b", cleaned, re.IGNORECASE):
        raise NGQLGuardError("禁止在 nGQL 中直接使用 attributes_json")
    if re.search(r"\bAS\s+_[a-z0-9_]*\b", cleaned, re.IGNORECASE):
        raise NGQLGuardError("内部列别名必须符合契约并使用反引号")
    if re.search(r"\$-\._[a-z0-9_]*\b", cleaned, re.IGNORECASE):
        raise NGQLGuardError("管道中的内部列引用必须符合契约并使用反引号")
    for alias in re.findall(r"\bAS\s+`(_[a-z0-9_]+)`", cleaned, re.IGNORECASE):
        if not re.fullmatch(_INTERNAL_IDENTIFIER, alias, re.IGNORECASE):
            raise NGQLGuardError(f"非法内部列别名: {alias}")

    if not _is_aggregate(cleaned):
        limit = _FINAL_LIMIT_PATTERN.search(cleaned)
        if limit is None:
            raise NGQLGuardError("非聚合查询必须在最终输出阶段包含 LIMIT")
        if int(limit.group(1)) > _max_rows():
            raise NGQLGuardError("LIMIT 超过 KG_NGQL_MAX_ROWS")
    return cleaned


def validate_before_compile(
    statement: str,
    anchor_tokens: set[str],
    max_hops: int = DEFAULT_MAX_HOPS,
) -> str:
    """Validate raw LLM output before trusted VIDs are inserted."""
    cleaned = _validate_shape(statement, max_hops)
    if "'" in cleaned or '"' in cleaned:
        raise NGQLGuardError("编译前禁止字符串字面量")
    if _VID_PATTERN.search(cleaned):
        raise NGQLGuardError("编译前禁止真实 VID")

    referenced = {match.group(0) for match in _ANCHOR_PATTERN.finditer(cleaned)}
    unknown = referenced - anchor_tokens
    if unknown:
        raise NGQLGuardError(f"引用了未知锚点: {', '.join(sorted(unknown))}")
    if not referenced and not (
        _is_aggregate(cleaned) and _FOLDER_SCOPE_PATTERN.search(cleaned)
    ):
        raise NGQLGuardError("查询必须使用锚点，或使用精确 folder_key 条件的聚合查询")
    return cleaned


def validate_after_compile(
    statement: str,
    injected_vids: set[str],
    max_hops: int = DEFAULT_MAX_HOPS,
) -> str:
    """Validate compiled output and prove all VID literals came from the application."""
    cleaned = _validate_shape(statement, max_hops)
    literals = _STRING_PATTERN.findall(cleaned)
    remainder = _STRING_PATTERN.sub("", cleaned)
    if "'" in remainder or '"' in remainder:
        raise NGQLGuardError("编译后存在无法识别的字符串字面量")

    literal_vids: set[str] = set()
    for literal in literals:
        value = literal[1:-1].replace(r"\"", '"').replace(r"\\", "\\")
        if not _VALID_VID_PATTERN.fullmatch(value):
            raise NGQLGuardError("编译后只允许应用注入的 VID 字符串")
        literal_vids.add(value)
    if literal_vids != injected_vids:
        raise NGQLGuardError("编译后的 VID 集合与应用注入集合不一致")
    if _ANCHOR_PATTERN.search(cleaned):
        raise NGQLGuardError("编译后仍存在锚点占位符")
    return cleaned


def compile_statement(
    statement: str,
    anchors: Mapping[int, Sequence[str]],
    max_hops: int = DEFAULT_MAX_HOPS,
) -> str:
    """Replace validated anchor tokens with validated VID literals."""
    token_values = {
        f"__ANCHOR_{index}_VIDS__": list(vids)
        for index, vids in anchors.items()
    }
    compiled = validate_before_compile(
        statement,
        set(token_values),
        max_hops,
    )
    injected: set[str] = set()

    for token in {match.group(0) for match in _ANCHOR_PATTERN.finditer(compiled)}:
        vids = token_values[token]
        if not vids:
            raise NGQLGuardError(f"锚点没有候选 VID: {token}")
        if any(not _VALID_VID_PATTERN.fullmatch(vid) for vid in vids):
            raise NGQLGuardError(f"锚点包含非法 VID: {token}")
        injected.update(vids)
        compiled = compiled.replace(
            token,
            ", ".join(_vid_literal(vid) for vid in vids),
        )

    return validate_after_compile(compiled, injected, max_hops)
