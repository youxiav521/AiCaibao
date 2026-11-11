from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from loguru import logger

from .parent_summary_cache import (
    ParentSummaryCachePayload,
    load_parent_summary_cache,
    store_parent_summary_available_cache,
    store_parent_summary_unavailable_cache,
)
from .summary_service_utils import coerce_int, normalize_file_ids, normalize_nonempty_string
from .vectorstore import VectorStoresService


PARENT_SUMMARY_QUERY_LIMIT = 10000


@dataclass(frozen=True)
class ParentSummaryRecord:
    """表示单条父切片摘要记录。"""

    parent_id: str
    file_id: str
    text: str
    summary: str
    source_element_ids: list[int]
    heading_path: str
    heading_node_ids: list[int]
    metadata: dict[str, Any]

    @property
    def min_source_element_id(self) -> int | None:
        """返回来源元素 ID 中的最小值。

        Returns:
            int | None: 最小来源元素 ID；若不存在来源元素，则返回 `None`。
        """
        if not self.source_element_ids:
            return None
        return min(self.source_element_ids)

    @property
    def sort_key_within_file(self) -> tuple[int | float, str]:
        """生成文件内稳定排序键。

        Returns:
            tuple[int | float, str]: 先按最小来源元素 ID，再按 `parent_id` 排序的键。
        """
        min_source_element_id = self.min_source_element_id
        return (
            min_source_element_id if min_source_element_id is not None else float("inf"),
            self.parent_id,
        )

    @property
    def requires_summary(self) -> bool:
        """仅正文非空的父切片才要求存在摘要。"""
        return bool(str(self.text or "").strip())


@dataclass(frozen=True)
class ParentSummaryFetchResult:
    """表示父切片摘要查询结果。"""

    available: bool
    records: list[ParentSummaryRecord]
    reason: str | None = None
    missing_file_ids: list[str] = field(default_factory=list)
    blank_summary_file_ids: list[str] = field(default_factory=list)

    def is_query_error(self) -> bool:
        """返回当前结果是否为底层查询失败。"""
        return self.reason == "query_error"

    def is_parse_error(self) -> bool:
        """返回当前结果是否为查询结果解析失败。"""
        return self.reason == "parse_error"

    def has_query_failure(self) -> bool:
        """返回当前结果是否属于不可直接使用的查询失败。"""
        return self.is_query_error() or self.is_parse_error()

    def raise_if_query_failed(self, *, phase: str) -> None:
        """在查询或解析失败时抛出异常。

        Args:
            phase: 当前所处阶段，用于补充异常上下文。

        Raises:
            RuntimeError: 当结果表示查询失败或解析失败时抛出。
        """
        if self.has_query_failure():
            raise RuntimeError(f"{phase}父切片摘要失败: {self.reason}")

    def repair_file_ids(self, file_ids: Sequence[str]) -> list[str]:
        """筛选需要重建父切片摘要的文件 ID。

        Args:
            file_ids: 待检查的文件 ID 序列。

        Returns:
            list[str]: 当前结果中缺失摘要或摘要为空的文件 ID 列表。
        """
        repair_targets: list[str] = []
        missing_file_ids = set(self.missing_file_ids)
        blank_summary_file_ids = set(self.blank_summary_file_ids)
        for file_id in normalize_file_ids(file_ids):
            if file_id in missing_file_ids or file_id in blank_summary_file_ids:
                repair_targets.append(file_id)
        return repair_targets


def _normalize_int_list(value: Any) -> list[int]:
    """将输入值规整为升序、去重的整数列表。

    Args:
        value: 原始值，可以是单值或列表、元组、集合。

    Returns:
        list[int]: 转换成功后的整数列表。
    """
    if isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        raw_values = [value]

    normalized: list[int] = []
    seen: set[int] = set()
    for raw_value in raw_values:
        parsed = coerce_int(raw_value)
        if parsed is None or parsed in seen:
            continue
        seen.add(parsed)
        normalized.append(parsed)
    return sorted(normalized)


def _normalize_heading_path(value: Any) -> str:
    """将标题路径规整为统一的斜杠分隔字符串。

    Args:
        value: 原始标题路径，支持字符串或字符串列表。

    Returns:
        str: 规整后的标题路径。
    """
    if isinstance(value, list):
        parts = [str(part).strip() for part in value if str(part or "").strip()]
        return "/".join(parts)
    return str(value or "").strip()


def _parse_parent_summary_row(row: Mapping[str, Any]) -> ParentSummaryRecord:
    """将向量库中的原始记录解析为父切片摘要对象。

    Args:
        row: 向量库返回的单条原始记录。

    Returns:
        ParentSummaryRecord: 规整后的父切片摘要记录。

    Raises:
        ValueError: 当关键字段缺失时抛出。
    """
    metadata = dict(row.get("metadata") or row)
    text = str(row.get("text") or metadata.get("text") or "")
    summary = str(row.get("summary") or metadata.get("summary") or "")
    file_id = normalize_nonempty_string(row.get("file_id") or metadata.get("file_id"))
    parent_id = normalize_nonempty_string(row.get("parent_id") or metadata.get("parent_id"))
    file_name = normalize_nonempty_string(row.get("file_name") or metadata.get("file_name"))
    folder_id = normalize_nonempty_string(row.get("folder_id") or metadata.get("folder_id"))

    if file_id is None:
        raise ValueError("parent summary row missing file_id")
    if parent_id is None:
        raise ValueError("parent summary row missing parent_id")

    source_element_ids = _normalize_int_list(metadata.get("source_element_ids", []))
    heading_node_ids = _normalize_int_list(metadata.get("heading_node_ids", []))
    heading_path = _normalize_heading_path(metadata.get("heading_path"))

    normalized_metadata = dict(metadata)
    normalized_metadata["file_id"] = file_id
    normalized_metadata["parent_id"] = parent_id
    normalized_metadata["source_element_ids"] = list(source_element_ids)
    normalized_metadata["heading_node_ids"] = list(heading_node_ids)
    normalized_metadata["heading_path"] = heading_path
    if file_name is not None:
        normalized_metadata["file_name"] = file_name
    if folder_id is not None:
        normalized_metadata["folder_id"] = folder_id
    if summary:
        normalized_metadata["summary"] = summary

    return ParentSummaryRecord(
        parent_id=parent_id,
        file_id=file_id,
        text=text,
        summary=summary,
        source_element_ids=source_element_ids,
        heading_path=heading_path,
        heading_node_ids=heading_node_ids,
        metadata=normalized_metadata,
    )


async def _query_parent_summary_rows(
    file_ids: list[str]
) -> list[dict[str, Any]]:
    """从父摘要集合中查询指定文件的记录。

    Args:
        file_ids: 待查询的文件 ID 列表。

    Returns:
        list[dict[str, Any]]: 向量库返回的原始记录列表。
    """
    return await VectorStoresService.read_parent_summary_hits(file_ids, limit=PARENT_SUMMARY_QUERY_LIMIT)


async def load_parent_summaries_for_files(
    file_ids: Sequence[str] | None,
) -> ParentSummaryFetchResult:
    """加载并校验多个文件的父切片摘要。

    Args:
        file_ids: 待加载的文件 ID 序列。

    Returns:
        ParentSummaryFetchResult: 包含摘要记录、缺失文件和空摘要文件信息的结果对象。
    """
    stable_file_ids = normalize_file_ids(file_ids)
    if not stable_file_ids:
        return ParentSummaryFetchResult(
            available=False,
            records=[],
            reason="empty_file_ids",
        )

    try:
        rows = await _query_parent_summary_rows(stable_file_ids)
    except Exception as error:
        logger.warning("加载父切片摘要失败: {}", error)
        return ParentSummaryFetchResult(
            available=False,
            records=[],
            reason="query_error",
        )

    if len(rows) >= PARENT_SUMMARY_QUERY_LIMIT:
        logger.warning(
            "父切片摘要查询达到上限: 文件 ID 列表={}, 上限={}",
            stable_file_ids,
            PARENT_SUMMARY_QUERY_LIMIT,
        )

    file_order = {file_id: index for index, file_id in enumerate(stable_file_ids)}
    files_with_records: set[str] = set()
    files_with_blank_summaries: set[str] = set()
    records: list[ParentSummaryRecord] = []

    try:
        for row in rows or []:
            record = _parse_parent_summary_row(row)
            if record.file_id not in file_order:
                continue
            files_with_records.add(record.file_id)
            if record.requires_summary and not record.summary.strip():
                files_with_blank_summaries.add(record.file_id)
            records.append(record)
    except Exception as error:
        logger.warning("解析父切片摘要记录失败: {}", error)
        return ParentSummaryFetchResult(
            available=False,
            records=[],
            reason="parse_error",
        )

    records.sort(key=lambda record: (file_order[record.file_id], *record.sort_key_within_file))
    missing_file_ids = [file_id for file_id in stable_file_ids if file_id not in files_with_records]
    blank_summary_file_ids = [file_id for file_id in stable_file_ids if file_id in files_with_blank_summaries]
    reason_parts: list[str] = []
    if missing_file_ids:
        reason_parts.append(f"missing_files:{','.join(missing_file_ids)}")
    if blank_summary_file_ids:
        reason_parts.append(f"blank_summary:{','.join(blank_summary_file_ids)}")
    return ParentSummaryFetchResult(
        available=not missing_file_ids and not blank_summary_file_ids,
        records=records,
        reason=";".join(reason_parts) if reason_parts else None,
        missing_file_ids=missing_file_ids,
        blank_summary_file_ids=blank_summary_file_ids,
    )


async def load_usable_parent_summaries_by_file(
    file_ids: Sequence[str] | None,
) -> dict[str, list[ParentSummaryRecord]]:
    """在向量存储前使用 Redis 缓存加载可用的父摘要记录"""
    stable_file_ids = normalize_file_ids(file_ids)
    if not stable_file_ids:
        return {}

    available_records: dict[str, list[ParentSummaryRecord]] = {}
    uncached_file_ids: list[str] = []
    for file_id in stable_file_ids:
        cached_payload = await load_parent_summary_cache(file_id)
        if cached_payload is None:
            uncached_file_ids.append(file_id)
            continue
        if cached_payload.state == "unavailable":
            continue
        cached_records = [
            ParentSummaryRecord(**record.model_dump())
            for record in cached_payload.records
        ]
        if cached_records:
            available_records[file_id] = cached_records

    if not uncached_file_ids:
        return available_records

    result = await load_parent_summaries_for_files(uncached_file_ids)
    if result.is_query_error():
        return available_records
    if result.is_parse_error():
        for file_id in uncached_file_ids:
            single_result = await load_parent_summaries_for_files([file_id])
            if single_result.has_query_failure():
                continue
            file_records = [
                record for record in single_result.records if record.file_id == file_id
            ]
            if (
                file_id in single_result.missing_file_ids
                or file_id in single_result.blank_summary_file_ids
                or not file_records
            ):
                await store_parent_summary_unavailable_cache(file_id)
                continue
            await store_parent_summary_available_cache(file_id, file_records)
            available_records[file_id] = file_records
        return available_records

    grouped: dict[str, list[ParentSummaryRecord]] = {}
    for record in result.records:
        grouped.setdefault(record.file_id, []).append(record)

    for file_id in uncached_file_ids:
        file_records = grouped.get(file_id, [])
        if (
            file_id in result.missing_file_ids
            or file_id in result.blank_summary_file_ids
            or not file_records
        ):
            await store_parent_summary_unavailable_cache(file_id)
            continue
        await store_parent_summary_available_cache(file_id, file_records)
        available_records[file_id] = file_records

    return available_records
