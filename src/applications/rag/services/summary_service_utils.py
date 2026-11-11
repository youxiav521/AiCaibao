from __future__ import annotations

from typing import Any, Mapping, Sequence


def get_field_value(source: Any, field_name: str) -> Any:
    """从映射或对象中读取指定字段的值。

    Args:
        source: 待读取的对象，可以是 `dict`、`Mapping` 或普通对象。
        field_name: 需要读取的字段名。

    Returns:
        Any: 字段对应的值；若字段不存在，则返回 `None`。
    """
    if isinstance(source, Mapping):
        return source.get(field_name)
    return getattr(source, field_name, None)


def normalize_nonempty_string(value: Any) -> str | None:
    """将任意值规整为非空字符串。

    Args:
        value: 待规整的原始值。

    Returns:
        str | None: 去除首尾空白后的字符串；若结果为空，则返回 `None`。
    """
    text = str(value or "").strip()
    return text or None


def normalize_file_ids(file_ids: Sequence[Any] | None) -> list[str]:
    """规整并去重文件 ID 列表。

    Args:
        file_ids: 原始文件 ID 序列，允许包含空值或重复值。

    Returns:
        list[str]: 按原始顺序保留的非空、去重文件 ID 列表。
    """
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_file_id in file_ids or ():
        file_id = normalize_nonempty_string(raw_file_id)
        if file_id is None or file_id in seen:
            continue
        seen.add(file_id)
        normalized.append(file_id)
    return normalized


def coerce_int(value: Any) -> int | None:
    """将常见输入值转换为整数。

    Args:
        value: 待转换的值，支持 `int`、整数型 `float` 和数字字符串。

    Returns:
        int | None: 转换后的整数；若无法安全转换，则返回 `None`。
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
    return None


def extract_origin_content_file_ids(origin_content: Sequence[Any]) -> list[str]:
    """从原始内容中提取有序且去重的文件 ID。

    Args:
        origin_content: 原始内容项序列，每项需可读取 `file_id` 字段。

    Returns:
        list[str]: 按首次出现顺序排列的文件 ID 列表。
    """
    ordered_file_ids: list[str] = []
    seen: set[str] = set()
    for item in origin_content or ():
        file_id = normalize_nonempty_string(get_field_value(item, "file_id"))
        if file_id is None or file_id in seen:
            continue
        seen.add(file_id)
        ordered_file_ids.append(file_id)
    return ordered_file_ids


def group_origin_content_by_file_id(origin_content: Sequence[Any]) -> tuple[list[str], dict[str, list[Any]]]:
    """按文件 ID 对原始内容进行分组。

    Args:
        origin_content: 原始内容项序列，每项需可读取 `file_id` 字段。

    Returns:
        tuple[list[str], dict[str, list[Any]]]:
            第一个返回值为按出现顺序收集的文件 ID 列表；
            第二个返回值为 `file_id -> 内容项列表` 的映射。
    """
    ordered_file_ids: list[str] = []
    file_to_items: dict[str, list[Any]] = {}
    for item in origin_content or ():
        file_id = normalize_nonempty_string(get_field_value(item, "file_id"))
        if file_id is None:
            continue
        if file_id not in file_to_items:
            ordered_file_ids.append(file_id)
            file_to_items[file_id] = []
        file_to_items[file_id].append(item)
    return ordered_file_ids, file_to_items
