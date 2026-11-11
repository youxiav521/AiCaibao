import asyncio
import re
import unicodedata
from collections.abc import Mapping
from difflib import SequenceMatcher
from typing import Any, Callable, List, Optional, TypeVar

import json_repair
from loguru import logger
from pydantic import BaseModel, ValidationError

from src.ade.knowledge_graph.client.kg_client import KGClient
from src.ade.knowledge_graph.client.schema.extraction_schema import (
    ENTITY_TYPE_MAP,
    RELATION_TYPE_MAP,
    ExtractedAttribute,
    ExtractedEntity,
    ExtractedRelation,
    KGExtractionResult,
)
from src.ade.knowledge_graph.client.prompt import (
    CONFIDENCE_LEVEL_LIST,
    ENTITY_TYPE_LIST,
    RELATION_TYPE_LIST,
)
from src.ade.knowledge_graph.schema.kg_schema import ElementMeta, Entity, Piece, Reference, Relation
from src.utils.json_utils import dumps as json_dumps

_INVALID_ENTITY_NAMES = set(ENTITY_TYPE_LIST) | set(RELATION_TYPE_LIST)
_GENERIC_ENTITY_NAMES = {
    "本公司",
    "该公司",
    "我司",
    "本机构",
    "本单位",
    "本集团",
    "该集团",
    "本集团公司",
    "该集团公司",
}
_CONFIDENCE_LEVELS = set(CONFIDENCE_LEVEL_LIST)
_LLM_MAX_504_RETRIES = 2
_LLM_MAX_FATAL_ATTEMPTS = 5
_LLM_RETRY_DELAY_SECONDS = 2
_METADATA_ENTITY_CONFIDENCE_REASON = "证据不足：实体来源于元数据。"
_METADATA_RELATION_CONFIDENCE_REASON = "证据不足：头实体或尾实体来源于元数据。"
_HASH_LIKE_METADATA_PATTERN = re.compile(
    r"^(?:[0-9a-f]{16,}|[0-9]{8,})$", re.IGNORECASE
)
_GENERIC_METADATA_PATTERN = re.compile(
    r"^(?:附件|文件|文档|材料|扫描件|新建文件夹|未命名|"
    r"attachment|document|file)(?:\d+)?(?:最终版|最新版|final)?$",
    re.IGNORECASE,
)
_FILE_EXTENSION_PATTERN = re.compile(
    r"\.(?:pdf|docx?|xlsx?|pptx?|csv|txt|md|html?|png|jpe?g|zip|rar|7z)$",
    re.IGNORECASE,
)
_ModelT = TypeVar("_ModelT", bound=BaseModel)


class FatalKGExtractionError(RuntimeError):
    """当前切片的结构化输出不可恢复。"""


class KGExtractionOutputTruncatedError(RuntimeError):
    """当前切片的模型输出达到长度上限。"""


MatchStats = dict[str, int]
_MATCH_STAT_KEYS = (
    "total",
    "matched",
    "source_mismatch",
    "element_not_found",
    "empty_source",
)


def create_match_stats() -> MatchStats:
    """创建单次知识图谱生成任务独享的溯源统计。"""
    return {key: 0 for key in _MATCH_STAT_KEYS}


def _merge_match_stats(target: MatchStats, source: MatchStats) -> None:
    for key in _MATCH_STAT_KEYS:
        target[key] += source[key]


def _clean_html(s: str) -> str:
    return s.replace("<br/>", "\n").replace("<br>", "\n").replace("&nbsp;", " ")


def _render_piece_input(piece: Piece) -> str:
    """为 LLM 标记每段原文的 element_id，不修改 Piece.content。"""
    return "\n\n".join(
        f'<element element_id="{element.element_id}">\n{element.md}\n</element>'
        for element in piece.elements
    )


def _error_summary(exc: BaseException) -> str:
    return " ".join(str(exc).split())[:500]


def _is_504_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 504:
        return True

    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) == 504:
        return True

    message = _error_summary(exc).lower()
    return "504" in message or "gateway timeout" in message


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return " ".join(s.split()).strip()


def _clean_metadata_value(value: str, *, is_file_name: bool = False) -> str:
    """只过滤确定无语义的元数据，保留可能有业务含义的名称。"""
    normalized = _normalize(value or "")
    if not normalized:
        return ""

    normalized = re.split(r"[/\\]", normalized)[-1].strip()
    if is_file_name:
        normalized = _FILE_EXTENSION_PATTERN.sub("", normalized).strip()

    compact = re.sub(r"[\s._-]+", "", normalized)
    if (
        not compact
        or _HASH_LIKE_METADATA_PATTERN.fullmatch(compact)
        or _GENERIC_METADATA_PATTERN.fullmatch(compact)
    ):
        return ""
    return normalized


def _build_document_metadata(folder_name: str, file_name: str) -> dict[str, str]:
    metadata = {
        "folder_name": _clean_metadata_value(folder_name),
        "file_name": _clean_metadata_value(file_name, is_file_name=True),
    }
    return {key: value for key, value in metadata.items() if value}


def _name_is_metadata_derived(
    name: str,
    piece_content: str,
    document_metadata: Mapping[str, str],
) -> bool:
    normalized_name = _normalize(name)
    if not normalized_name or normalized_name in _normalize(piece_content):
        return False
    return any(
        normalized_name in _normalize(value)
        for value in document_metadata.values()
    )


def _apply_metadata_confidence_cap(
    confidence: Optional[str],
    confidence_reason: Optional[str],
    *,
    metadata_derived: bool,
    reason: str,
) -> tuple[Optional[str], Optional[str]]:
    if not metadata_derived:
        return confidence, confidence_reason
    if confidence == "High":
        confidence = "Medium"
    return confidence, reason


def _is_invalid_entity_name(name: str) -> bool:
    normalized_name = name.strip()
    return (
        normalized_name in _INVALID_ENTITY_NAMES
        or normalized_name in _GENERIC_ENTITY_NAMES
    )


def _lcs_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    sm = SequenceMatcher(None, a, b)
    m = sm.find_longest_match(0, len(a), 0, len(b))
    return m.size / min(len(a), len(b))


def _match_table_row(source: str, elem: ElementMeta) -> bool:
    """表格溯源匹配：仅处理水平数据表（首行为列名表头，第 2 行起为数据行）。"""
    if elem.type != "table" or not elem.rows:
        return False

    # 从第一行构建列名映射
    headers: dict[int, str] = {}
    for cell in elem.rows[0]:
        cs = cell.get("colstart")
        if isinstance(cs, int) and cs >= 0 and cs not in headers:
            headers[cs] = cell.get("text", "").strip()
    if not headers:
        return False

    # 解析 source 中的 key|value 对
    parts = [p.strip() for p in source.split("|")]
    source_pairs: list[tuple[str, str]] = []
    i = 0
    while i + 1 < len(parts):
        if parts[i] and parts[i + 1]:
            source_pairs.append((parts[i], parts[i + 1]))
        i += 2
    if not source_pairs:
        return False

    # 在数据行中匹配，任一数据行全部 pair 命中即成功
    for row in elem.rows[1:]:
        row_dict: dict[str, str] = {}
        for cell in row:
            cs = cell.get("colstart")
            h = headers.get(cs, "")
            if h:
                row_dict[h] = cell.get("text", "").strip()

        if all(
            sv in (rv := row_dict.get(sk, "")) or sv == rv
            for sk, sv in source_pairs
        ):
            return True

    return False
    

def _match_source(source: str, element: ElementMeta) -> bool:
    """验证 source 是否属于指定 element，不负责查找或修正 element_id。"""
    if not source:
        return False

    source = _clean_html(source)

    # L1: exact substring
    if source in element.md:
        return True

    # L2: normalized substring
    src_n = _normalize(source)
    if src_n in _normalize(element.md):
        return True

    # L3: 分段匹配（LLM 用 ... 省略中间内容时）
    if "..." in source:
        parts = [p.strip() for p in source.split("...") if p.strip()]
        if len(parts) >= 2 and all(part in element.md for part in parts):
            return True
                    
    # P1: table row matching
    if _match_table_row(source, element):
        return True

    # # L3: LCS
    # if _lcs_ratio(source, element.md) >= 0.5:
    #     return True

    return False


def _resolve_reference(
    source: str,
    element_id: int,
    elements_by_id: dict[int, ElementMeta],
    match_stats: MatchStats,
) -> tuple[int, Optional[int], bool]:
    """按 LLM 返回的 ID 确定位置，并仅校验该 element 的原文。"""
    match_stats["total"] += 1
    element = elements_by_id.get(element_id)
    if element is None:
        match_stats["element_not_found"] += 1
        return 0, None, False

    if not source:
        match_stats["empty_source"] += 1
        return element.element_id, element.page, False

    source_matched = _match_source(source, element)
    match_stats["matched" if source_matched else "source_mismatch"] += 1
    return element.element_id, element.page, source_matched


def _parse_confidence(
    raw_confidence: str, raw_reason: str, record_type: str
) -> tuple[Optional[str], Optional[str]]:
    confidence = raw_confidence.strip()
    reason = raw_reason.strip()

    if confidence not in _CONFIDENCE_LEVELS:
        logger.warning("ExtractorService | 无效置信度 | record_type={} confidence={}", record_type, confidence)
        return None, None

    if confidence == "High":
        if reason:
            logger.warning("ExtractorService | High置信度原因应为空 | record_type={}", record_type)
        return confidence, None

    if not reason:
        logger.warning("ExtractorService | 置信度原因缺失 | record_type={} confidence={}", record_type, confidence)
        return confidence, None

    return confidence, reason


def _split_relation_types(raw_type: str) -> list[str]:
    relation_types: list[str] = []
    seen: set[str] = set()
    for item in raw_type.replace("，", ",").split(","):
        relation_type = item.strip()
        if relation_type and relation_type not in seen:
            seen.add(relation_type)
            relation_types.append(relation_type)
    return relation_types


def _group_relation_attributes(
    attributes: list[ExtractedAttribute],
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for attribute in attributes:
        name = attribute.name.strip()
        value = attribute.value.strip()
        if not name or not value:
            continue
        values = grouped.setdefault(name, [])
        if value not in values:
            values.append(value)
    return grouped


def _record_name(record: Any) -> str:
    if not isinstance(record, Mapping):
        return "?"
    return str(record.get("name") or record.get("from_entity") or "?")[:200]


def _validate_records(
    records: list[Any], model_type: type[_ModelT], scope: str
) -> tuple[list[_ModelT], list[dict[str, Any]]]:
    validated: list[_ModelT] = []
    issues: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        try:
            validated.append(model_type.model_validate(record))
        except ValidationError as exc:
            logger.exception(
                "ExtractorService | LLM记录校验失败，跳过 | scope={} index={} name={}",
                scope,
                index,
                _record_name(record),
            )
            issues.append(
                {
                    "scope": scope,
                    "index": index,
                    "name": _record_name(record),
                    "errors": [
                        {
                            "loc": list(error["loc"]),
                            "type": error["type"],
                            "msg": error["msg"],
                        }
                        for error in exc.errors(include_url=False, include_input=False)
                    ],
                }
            )
    return validated, issues


def _exception_payload(exc: BaseException | None) -> dict[str, str] | None:
    if exc is None:
        return None
    return {"type": type(exc).__name__, "message": _error_summary(exc)}


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _finish_reason(raw: Any) -> str | None:
    metadata = (
        raw.get("response_metadata")
        if isinstance(raw, Mapping)
        else getattr(raw, "response_metadata", None)
    )
    if not isinstance(metadata, Mapping):
        return None
    finish_reason = metadata.get("finish_reason")
    return finish_reason if isinstance(finish_reason, str) else None


def _raw_tool_args(raw: Any) -> Mapping[str, Any] | None:
    tool_calls = getattr(raw, "tool_calls", None)
    if not isinstance(tool_calls, list):
        return None

    for tool_call in tool_calls:
        if not isinstance(tool_call, Mapping):
            continue
        if tool_call.get("name") != KGExtractionResult.__name__:
            continue
        args = tool_call.get("args")
        return args if isinstance(args, Mapping) else None
    return None


def _repair_invalid_tool_args(
    raw: Any,
) -> tuple[KGExtractionResult | None, dict[str, str] | None, str | None]:
    invalid_tool_calls = getattr(raw, "invalid_tool_calls", None)
    if not isinstance(invalid_tool_calls, list):
        return None, None, None

    for tool_call in invalid_tool_calls:
        if not isinstance(tool_call, Mapping):
            continue
        if tool_call.get("name") != KGExtractionResult.__name__:
            continue

        args = tool_call.get("args")
        if not isinstance(args, str):
            return (
                None,
                {
                    "type": "TypeError",
                    "message": "KGExtractionResult invalid tool call args 必须是字符串",
                },
                None,
            )

        repair_action: str | None = None
        try:
            repaired = json_repair.loads(args)
            if not isinstance(repaired, Mapping):
                return (
                    None,
                    {
                        "type": "TypeError",
                        "message": "JSON repair 结果必须是对象",
                    },
                    None,
                )

            relations_keys = list(re.finditer(r'"relations"\s*:', args))
            relations_tail = (
                args[relations_keys[0].end() :].strip()
                if len(relations_keys) == 1
                else None
            )
            if relations_tail == "[":
                return (
                    None,
                    {
                        "type": "ValueError",
                        "message": "JSON repair relations 数组未闭合",
                    },
                    None,
                )
            if (
                relations_tail == ""
                and isinstance(repaired.get("entities"), list)
                and repaired.get("relations") == ""
            ):
                repaired = dict(repaired)
                repaired["relations"] = []
                repair_action = "complete_empty_relations_at_eof"

            invalid_fields = [
                field
                for field in ("entities", "relations")
                if field not in repaired or not isinstance(repaired[field], list)
            ]
            if invalid_fields:
                return (
                    None,
                    {
                        "type": "ValueError",
                        "message": (
                            "JSON repair 顶层数组缺失或类型错误: "
                            f"{', '.join(invalid_fields)}"
                        ),
                    },
                    repair_action,
                )

            return (
                KGExtractionResult.model_validate(repaired, strict=True),
                None,
                repair_action,
            )
        except ValidationError as exc:
            errors = exc.errors(include_url=False, include_input=False)
            first_error = errors[0] if errors else {}
            return (
                None,
                {
                    "type": type(exc).__name__,
                    "message": (
                        f"loc={list(first_error.get('loc', []))}, "
                        f"msg={first_error.get('msg', '严格校验失败')}"
                    ),
                },
                repair_action,
            )
        except Exception as exc:
            return (
                None,
                {
                    "type": type(exc).__name__,
                    "message": "JSON repair 解析失败",
                },
                repair_action,
            )

    return None, None, None


class ExtractorService:
    """实体和关系提取服务"""

    # 注意：不能在类定义时实例化 KGClient。
    # KGClient.__init__ 会调用 ModelFactory.get_model 创建 ChatOpenAI → AsyncOpenAI → httpx.AsyncClient，
    # 在 uvicorn --reload 模式下旧 loop 关闭后 GC 触发 AsyncAPIClient.__del__ 中的
    # create_task(aclose()) 会抛 RuntimeError: Event loop is closed。
    # 改为首次使用时通过 _get_client() 懒加载。
    _client: KGClient | None = None

    @classmethod
    def _get_client(cls) -> KGClient:
        if cls._client is None:
            cls._client = KGClient()
        return cls._client

    @classmethod
    async def extract(
        cls,
        piece: Piece,
        folder_name: str = "",
        project_name: str = "",
        raw_response_callback: Callable[[Piece, str], None] | None = None,
        match_stats: MatchStats | None = None,
    ) -> tuple[List[Entity], List[Relation]]:
        """从切片中提取实体和关系

        Args:
            piece: 文档切片，携带file_id/file_name/elements用于溯源
            folder_name: 原始文件夹名称，仅作为文档元数据
            project_name: 项目节点名称；为空时兼容地使用folder_name
            raw_response_callback: LLM诊断包络回调，包含原始响应、校验结果与错误
            match_stats: 当前知识图谱生成任务的溯源匹配统计
        """
        project_name = project_name or folder_name
        max_attempts = _LLM_MAX_FATAL_ATTEMPTS
        for attempt in range(1, max_attempts + 1):
            attempt_match_stats = create_match_stats()
            try:
                result = await cls._extract_once(
                    piece=piece,
                    folder_name=folder_name,
                    project_name=project_name,
                    raw_response_callback=raw_response_callback,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    match_stats=attempt_match_stats,
                )
            except FatalKGExtractionError as exc:
                if attempt >= max_attempts:
                    raise
                logger.warning(
                    "ExtractorService | piece结构化输出致命失败，准备重试 | piece_id={} file_name={} attempt={}/{} next_attempt={} delay_seconds={} error_summary={}",
                    piece.id,
                    piece.file_name,
                    attempt,
                    max_attempts,
                    attempt + 1,
                    _LLM_RETRY_DELAY_SECONDS,
                    _error_summary(exc),
                )
                await asyncio.sleep(_LLM_RETRY_DELAY_SECONDS)
                continue

            if attempt > 1:
                logger.info(
                    "ExtractorService | piece结构化输出重试成功 | piece_id={} file_name={} attempt={}/{}",
                    piece.id,
                    piece.file_name,
                    attempt,
                    max_attempts,
                )
            if match_stats is not None:
                _merge_match_stats(match_stats, attempt_match_stats)
            return result

        raise RuntimeError("ExtractorService | 致命错误重试流程异常结束")

    @classmethod
    async def _extract_once(
        cls,
        piece: Piece,
        folder_name: str,
        project_name: str,
        raw_response_callback: Callable[[Piece, str], None] | None,
        attempt: int,
        max_attempts: int,
        match_stats: MatchStats,
    ) -> tuple[List[Entity], List[Relation]]:
        """执行一次LLM调用、校验和转换。"""
        client = cls._get_client()
        document_metadata = _build_document_metadata(
            folder_name, piece.file_name
        )
        max_504_attempts = _LLM_MAX_504_RETRIES + 1
        for retry_504_attempt in range(1, max_504_attempts + 1):
            try:
                response = await client.extract(
                    content=_render_piece_input(piece),
                    project_name=project_name,
                    document_metadata=document_metadata,
                )
                break
            except Exception as exc:
                if not _is_504_error(exc):
                    raise

                logger.warning(
                    "ExtractorService | LLM提取504失败 | piece_id={} file_name={} attempt={}/{} error_type={} error_summary={}",
                    piece.id,
                    piece.file_name,
                    retry_504_attempt,
                    max_504_attempts,
                    type(exc).__name__,
                    _error_summary(exc),
                )
                if retry_504_attempt >= max_504_attempts:
                    raise
                await asyncio.sleep(_LLM_RETRY_DELAY_SECONDS)

        raw = response.get("raw")
        parsed = response.get("parsed")
        parsing_error = response.get("parsing_error")
        finish_reason = _finish_reason(raw)
        output_truncated = finish_reason == "length"
        used_raw_fallback = False
        used_json_repair = False
        json_repair_error: dict[str, str] | None = None
        json_repair_action: str | None = None
        validation_errors: list[dict[str, Any]] = []
        conversion_errors: list[dict[str, Any]] = []
        result: KGExtractionResult | None = None
        entities: List[Entity] = []
        relations: List[Relation] = []
        candidate_count = 0
        valid_count = 0
        fatal_error: str | None = None

        if output_truncated:
            parsed_payload: Mapping[str, Any] | None = None
        elif isinstance(parsed, KGExtractionResult):
            parsed_payload = parsed.model_dump(mode="python")
        elif isinstance(parsed, Mapping):
            parsed_payload = parsed
        elif parsing_error is not None:
            parsed_payload = _raw_tool_args(raw)
            used_raw_fallback = parsed_payload is not None
        else:
            parsed_payload = None

        if parsed_payload is None and not output_truncated:
            repaired_result, json_repair_error, json_repair_action = (
                _repair_invalid_tool_args(raw)
            )
            if repaired_result is not None:
                parsed_payload = repaired_result.model_dump(mode="python")
                used_json_repair = True

        if output_truncated:
            fatal_error = "LLM输出达到长度上限"
            validation_errors.append(
                {
                    "scope": "top_level",
                    "index": None,
                    "name": "finish_reason",
                    "errors": [
                        {
                            "loc": ["finish_reason"],
                            "type": "output_truncated",
                            "msg": fatal_error,
                        }
                    ],
                }
            )
        elif parsed_payload is None:
            if json_repair_error is not None:
                fatal_error = "LLM tool call arguments JSON 修复或严格校验失败"
            elif parsing_error is not None:
                fatal_error = "LLM结构化输出解析失败"
            else:
                fatal_error = "LLM未返回可用的 tool call 对象"
            validation_errors.append(
                {
                    "scope": "top_level",
                    "index": None,
                    "name": "parsed",
                    "errors": [{"loc": [], "type": "mapping_type", "msg": fatal_error}],
                }
            )
        else:
            invalid_fields = [
                field
                for field in ("entities", "relations")
                if field not in parsed_payload
                or not isinstance(parsed_payload[field], list)
            ]
            if invalid_fields:
                fatal_error = "LLM tool call 顶层数组缺失或类型错误"
                validation_errors.extend(
                    {
                        "scope": "top_level",
                        "index": None,
                        "name": field,
                        "errors": [
                            {
                                "loc": [field],
                                "type": "list_type",
                                "msg": f"{field} 必须是数组",
                            }
                        ],
                    }
                    for field in invalid_fields
                )
            else:
                raw_entities = parsed_payload["entities"]
                raw_relations = parsed_payload["relations"]
                candidate_count = len(raw_entities) + len(raw_relations)
                valid_entities, entity_errors = _validate_records(
                    raw_entities, ExtractedEntity, "entity"
                )
                valid_relations, relation_errors = _validate_records(
                    raw_relations, ExtractedRelation, "relation"
                )
                validation_errors.extend(entity_errors)
                validation_errors.extend(relation_errors)
                valid_count = len(valid_entities) + len(valid_relations)
                result = KGExtractionResult(
                    entities=valid_entities,
                    relations=valid_relations,
                )

                if candidate_count > 0 and valid_count == 0:
                    fatal_error = "LLM返回的非空候选全部校验失败"
                else:
                    entities, relations, conversion_errors = cls._build(
                        result,
                        piece,
                        match_stats,
                        document_metadata,
                    )
                    if (
                        valid_count > 0
                        and len(conversion_errors) == valid_count
                        and not entities
                        and not relations
                    ):
                        fatal_error = "所有已校验候选均在转换阶段失败"

        diagnostic = {
            "attempt": attempt,
            "max_attempts": max_attempts,
            "status": "fatal" if fatal_error else "ok",
            "raw": _jsonable(raw),
            "parsed": _jsonable(parsed),
            "finish_reason": finish_reason,
            "output_truncated": output_truncated,
            "used_raw_fallback": used_raw_fallback,
            "used_json_repair": used_json_repair,
            "json_repair_error": json_repair_error,
            "json_repair_action": json_repair_action,
            "validated": result.model_dump(mode="json") if result is not None else None,
            "parsing_error": _exception_payload(parsing_error),
            "validation_errors": validation_errors,
            "conversion_errors": conversion_errors,
            "fatal_error": fatal_error,
        }
        if raw_response_callback is not None:
            try:
                raw_response_callback(
                    piece,
                    json_dumps(diagnostic, indent=2, default=str),
                )
            except Exception:
                logger.exception(
                    "ExtractorService | 诊断回调失败，继续生成 | piece_id={} file_name={}",
                    piece.id,
                    piece.file_name,
                )

        if output_truncated:
            raise KGExtractionOutputTruncatedError(
                f"{fatal_error} | piece_id={piece.id} | attempt={attempt}/{max_attempts}"
            )

        if fatal_error:
            raise FatalKGExtractionError(
                f"{fatal_error} | piece_id={piece.id} | attempt={attempt}/{max_attempts}"
            )

        logger.info(
            "ExtractorService | 提取完成 | piece_id={} candidates={} valid={} entities={} relations={}",
            piece.id,
            candidate_count,
            valid_count,
            len(entities),
            len(relations),
        )
        return entities, relations

    @classmethod
    def _build(
        cls,
        result: KGExtractionResult,
        piece: Piece,
        match_stats: MatchStats,
        document_metadata: Mapping[str, str] | None = None,
    ) -> tuple[List[Entity], List[Relation], list[dict[str, Any]]]:
        """转换已校验记录，返回实体、关系和单条转换错误。"""
        entities: List[Entity] = []
        relations: List[Relation] = []
        conversion_errors: list[dict[str, Any]] = []
        elements_by_id = {element.element_id: element for element in piece.elements}
        metadata = document_metadata or {}
        metadata_entity_names: set[str] = set()

        for index, item in enumerate(result.entities):
            try:
                entity_name = item.name.strip()
                entity_type = item.type.strip()
                if not entity_name:
                    continue
                if _is_invalid_entity_name(entity_name):
                    logger.warning("ExtractorService | 丢弃实体 | name={}", entity_name)
                    continue

                entity_type_en = ENTITY_TYPE_MAP.get(entity_type)
                if entity_type_en is None:
                    logger.warning(
                        "ExtractorService | 丢弃未知实体类型 | type={} name={}",
                        entity_type,
                        entity_name,
                    )
                    continue

                confidence, confidence_reason = _parse_confidence(
                    item.confidence or "", item.confidence_reason or "", "entity"
                )
                metadata_derived = (
                    item.element_id == 0
                    or item.confidence_reason == _METADATA_ENTITY_CONFIDENCE_REASON
                    or _name_is_metadata_derived(
                        entity_name, piece.content, metadata
                    )
                )
                if metadata_derived:
                    metadata_entity_names.add(entity_name)
                confidence, confidence_reason = _apply_metadata_confidence_cap(
                    confidence,
                    confidence_reason,
                    metadata_derived=metadata_derived,
                    reason=_METADATA_ENTITY_CONFIDENCE_REASON,
                )
                attributes = {
                    attr.name.strip(): attr.value.strip()
                    for attr in item.attributes
                    if attr.name and attr.name.strip()
                }
                source_text = item.source.strip()
                if metadata_derived:
                    element_id, page, source_matched = 0, None, False
                else:
                    element_id, page, source_matched = _resolve_reference(
                        source_text,
                        item.element_id,
                        elements_by_id,
                        match_stats,
                    )

                entities.append(
                    Entity(
                        type=entity_type,
                        name=entity_name,
                        type_en=entity_type_en,
                        attributes=attributes,
                        references=[
                            Reference(
                                source=source_text,
                                element_id=element_id,
                                file_id=piece.file_id,
                                page=page,
                                source_matched=source_matched,
                                confidence=confidence,
                                confidence_reason=confidence_reason,
                            )
                        ],
                        degree=0,
                    )
                )
            except Exception as exc:
                logger.exception(
                    "ExtractorService | 实体记录转换失败，跳过 | index={} name={}",
                    index,
                    getattr(item, "name", "?"),
                )
                conversion_errors.append(
                    {
                        "scope": "entity",
                        "index": index,
                        "name": getattr(item, "name", "?"),
                        "error": _exception_payload(exc),
                    }
                )

        for index, item in enumerate(result.relations):
            try:
                source_entity = item.from_entity.strip()
                raw_relation_type = item.type.strip()
                relation_types = _split_relation_types(raw_relation_type)
                target_entity = item.to_entity.strip()
                if not source_entity or not target_entity or not relation_types:
                    continue
                if _is_invalid_entity_name(source_entity) or _is_invalid_entity_name(
                    target_entity
                ):
                    logger.warning(
                        "ExtractorService | 丢弃关系 | from_entity={} to_entity={}",
                        source_entity,
                        target_entity,
                    )
                    continue

                if "," in raw_relation_type or "，" in raw_relation_type:
                    logger.warning(
                        "ExtractorService | 组合关系已拆分 | raw_type={} types={}",
                        raw_relation_type,
                        relation_types,
                    )

                confidence, confidence_reason = _parse_confidence(
                    item.confidence or "", item.confidence_reason or "", "relation"
                )
                metadata_derived = (
                    item.confidence_reason == _METADATA_RELATION_CONFIDENCE_REASON
                    or source_entity in metadata_entity_names
                    or target_entity in metadata_entity_names
                    or _name_is_metadata_derived(
                        source_entity, piece.content, metadata
                    )
                    or _name_is_metadata_derived(
                        target_entity, piece.content, metadata
                    )
                )
                confidence, confidence_reason = _apply_metadata_confidence_cap(
                    confidence,
                    confidence_reason,
                    metadata_derived=metadata_derived,
                    reason=_METADATA_RELATION_CONFIDENCE_REASON,
                )
                attributes = _group_relation_attributes(item.attributes)
                source_text = item.source.strip()
                element_id, page, source_matched = _resolve_reference(
                    source_text,
                    item.element_id,
                    elements_by_id,
                    match_stats,
                )

                for relation_type in relation_types:
                    relation_type_en = RELATION_TYPE_MAP.get(relation_type)
                    if relation_type_en is None:
                        logger.warning("ExtractorService | 未知关系类型 | type={}", relation_type)
                        relation_type_en = relation_type

                    relations.append(
                        Relation(
                            type=relation_type,
                            type_en=relation_type_en,
                            from_entity=source_entity,
                            from_entity_id="",
                            to_entity=target_entity,
                            to_entity_id="",
                            attributes=attributes,
                            references=(
                                [
                                    Reference(
                                        source=source_text,
                                        element_id=element_id,
                                        file_id=piece.file_id,
                                        page=page,
                                        source_matched=source_matched,
                                        confidence=confidence,
                                        confidence_reason=confidence_reason,
                                    )
                                ]
                                if source_text
                                else []
                            ),
                        )
                    )
            except Exception as exc:
                logger.exception(
                    "ExtractorService | 关系记录转换失败，跳过 | index={} from_entity={} to_entity={}",
                    index,
                    getattr(item, "from_entity", "?"),
                    getattr(item, "to_entity", "?"),
                )
                conversion_errors.append(
                    {
                        "scope": "relation",
                        "index": index,
                        "name": getattr(item, "from_entity", "?"),
                        "error": _exception_payload(exc),
                    }
                )

        return entities, relations, conversion_errors
