from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO, Mapping, TextIO

import orjson

JSON_MIME_TYPE = "application/json"


def is_json_content_type(content_type: str | None) -> bool:
    if not content_type:
        return False
    normalized = str(content_type).split(";", 1)[0].strip().lower()
    return normalized == JSON_MIME_TYPE or normalized.endswith("+json")


def loads(data: bytes | bytearray | memoryview | str) -> Any:
    if isinstance(data, memoryview):
        return orjson.loads(data.tobytes())
    return orjson.loads(data)


def load(source: str | Path | BinaryIO | TextIO, *, encoding: str = "utf-8") -> Any:
    if isinstance(source, (str, Path)):
        return loads(Path(source).read_bytes())

    raw = source.read()
    if isinstance(raw, str):
        raw = raw.encode(encoding)
    return loads(raw)


def dumps_bytes(
    value: Any,
    *,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
    default: Any | None = None,
    indent: int | None = None,
) -> bytes:
    _ = ensure_ascii
    options = 0
    if sort_keys:
        options |= orjson.OPT_SORT_KEYS
    if indent:
        options |= orjson.OPT_INDENT_2
    return orjson.dumps(value, option=options, default=default)


def dumps(
    value: Any,
    *,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
    default: Any | None = None,
    indent: int | None = None,
) -> str:
    return dumps_bytes(
        value,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys,
        default=default,
        indent=indent,
    ).decode("utf-8")


def dump(
    value: Any,
    target: str | Path | BinaryIO | TextIO,
    *,
    encoding: str = "utf-8",
    ensure_ascii: bool = False,
    sort_keys: bool = False,
    default: Any | None = None,
    indent: int | None = None,
) -> None:
    payload = dumps_bytes(
        value,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys,
        default=default,
        indent=indent,
    )
    if isinstance(target, (str, Path)):
        Path(target).write_bytes(payload)
        return

    try:
        target.write(payload)
    except TypeError:
        target.write(payload.decode(encoding))


def response_json(response: Any) -> Any:
    content = getattr(response, "content", None)
    if content is None:
        text = getattr(response, "text", "")
        content = text.encode("utf-8") if isinstance(text, str) else text
    return loads(content or b"null")


def request_content(
    payload: Any,
    headers: Mapping[str, str] | None = None,
) -> tuple[bytes, dict[str, str]]:
    request_headers = dict(headers or {})
    request_headers.setdefault("Content-Type", JSON_MIME_TYPE)
    return dumps_bytes(payload), request_headers
