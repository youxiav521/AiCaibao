"""Knowledge-graph query V3: mention anchors, generated nGQL, and hydration."""

from __future__ import annotations

import os
import re
from contextlib import AsyncExitStack, ExitStack
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from src.ade.knowledge_graph.client.schema.definitions import RELATION_TYPES
from src.ade.knowledge_graph.nebula.client import NebulaGraphError
from src.ade.knowledge_graph.nebula.repository import EdgeKey, NebulaGraphRepository
from src.ade.knowledge_graph.nebula.schema import (
    RELATION_TYPE_BY_EDGE,
    folder_key,
    render_schema_for_prompt,
)
from src.ade.knowledge_graph.query.ngql_guard import (
    NGQLGuardError,
    compile_statement,
    extract_statement,
)
from src.ade.knowledge_graph.query.prompt import (
    GRAPH_FAIL_RESPONSE,
    QUERY_FAIL_RESPONSE,
    TUPLE_DELIMITER,
)
from src.ade.knowledge_graph.schema.kg_schema import Entity, QueryData, Relation
from src.ade.knowledge_graph.service.entity_vector_store import (
    AnchorHit,
    EntityVectorStore,
)
from src.ade.knowledge_graph.service.graph_storage_service import GraphStorageService
from src.applications.rag.services.vectorstore import VectorStoresService
from src.base.distributed_lock import DistributedLock
from src.utils.model_factory import ModelFactory
from src.utils.stopwatch import StopWatch, TaskContext
from src.utils.tokenizer import Tokenizer

_EDGE_COLUMN_PATTERN = re.compile(r"^_edge(\d+)_(src|dst|type|rank)$")
# 每个称呼只保留排序最高的一个锚点：多个同名关联企业等权进入 __ANCHOR_N_VIDS__
# 会把几家公司的事实混进同一个答案，精度优先于召回。
_ANCHOR_TOP_K = 1
_ASSOCIATION_ENTITY_TYPES = ["企业主体", "自然人"]
_ASSOCIATION_EDGE_TYPES = (
    "shareholder_of|controller_of|executive_of|kinship_of|spouse_of"
)
_ASSOCIATION_MAX_HOPS = 4
_ASSOCIATION_MENTION = "关联方"
_ASSOCIATION_SUBJECT_REQUIRED = (
    "请明确提供一个或两个企业或自然人主体后再查询关联关系。"
)
_PROJECT_ANCHOR_MENTION = "本项目"
_PROJECT_ROLE_MENTIONS = frozenset(
    definition.name
    for definition in RELATION_TYPES
    if "项目" in definition.target_types
) | frozenset({"借款人", "借款企业", "申请人", "本次授信", "本次授信业务"})
_PROJECT_QUERY_TERMS = _PROJECT_ROLE_MENTIONS | frozenset(
    {"本项目", "作担保", "提供担保"}
)
_NON_ENTITY_MENTIONS = frozenset(
    definition.name for definition in RELATION_TYPES
) | frozenset(
    {
        "关联关系",
        "供应商",
        "法人",
        "股东",
        "高管",
    }
)

_MENTION_PROMPT = """你是知识图谱查询的实体锚点抽取器。
从用户问题中按出现顺序抽取需要在图中定位的实体称呼。

规则：
- 抽取明确专名，例如“比亚迪”“王传福”。
- 不抽取业务角色、属性名、关系名或疑问词，例如“借款人”“保证人”“股东”“法定代表人”“哪些”。
- “本项目”“本次授信”等上下文指代不作为实体称呼抽取，由应用绑定当前项目。
- 项目级纯统计问题可以返回空列表。
- 不改写、不扩展用户原文。
- 当问题包含“关联”且明确给出一个或两个企业/自然人主体时，同时写入 association_subjects；每项包含原文称呼 mention 和预期类型 expected_types。
- 如果同一称呼在自关联问题中出现两次，association_subjects 也必须保留两项，不得去重。
- expected_types 只能包含“企业主体”或“自然人”；类型明确时只写一个，确实无法区分时可同时写两个。
- 仅查询是否关联、关联方或关联路径时，needs_association_enrichment=false；还要求高管、供应商、担保、投资等信息时为 true。
- 不要把“借款人”“保证人”“该企业”等上下文角色放入 association_subjects。
- 以 JSON object 输出，字段格式为 {{"mentions": ["实体称呼"], "association_subjects": [{{"mention": "主体称呼", "expected_types": ["企业主体"]}}], "needs_association_enrichment": false}}。
- 只输出 JSON object，不要 Markdown 围栏、解释或其他文字。

用户问题：
{query}
"""

_NGQL_SYSTEM_PROMPT = """你是 NebulaGraph 3.8 的只读 nGQL 生成器。
只输出一条可执行 nGQL，不要 Markdown 围栏、解释或注释。
只能使用给定的物理 schema，不得臆造 Tag、Edge 或属性。
"""

_ANSWER_SYSTEM_PROMPT = f"""你是专业的知识图谱问答助手，只能依据给定查询结果回答。

输入包含：
- <查询语句>：本次执行的 nGQL。
- <查询结果>：业务标量列；“（空）”表示查询成功但没有结果。
- <实体>：格式为 (名称{TUPLE_DELIMITER}类型{TUPLE_DELIMITER}属性)。
- <关系>：格式为 (源实体{TUPLE_DELIMITER}关系类型{TUPLE_DELIMITER}目标实体)。
- <原始文本>：实体和关系的来源。

规则：
1. 禁止编造或使用输入之外的信息。
2. 只有 <查询结果> 明确为“（空）”时，才回答“当前图谱未检索到相关数据”。
   “命中 N 行（无业务标量列）”表示查询已命中，必须依据 <实体> 和 <关系> 回答。
3. <关系> 严格按“源实体 → 关系类型 → 目标实体”解释，不得颠倒方向。
   例如“甲 → 股东 → 乙”表示甲是乙的股东；“甲 → 主借款人 → 项目”表示甲是该项目的借款人。
4. 数据存在时直接、简洁回答；数据不足时如实说明。
5. 不使用 Markdown 标记。
"""


class _AssociationSubject(BaseModel):
    mention: str
    expected_types: list[Literal["企业主体", "自然人"]]

    @field_validator("mention")
    @classmethod
    def normalize_mention(cls, value: str) -> str:
        return value.strip()

    @field_validator("expected_types")
    @classmethod
    def normalize_expected_types(
        cls,
        values: list[Literal["企业主体", "自然人"]],
    ) -> list[Literal["企业主体", "自然人"]]:
        return list(dict.fromkeys(values))


class _MentionExtraction(BaseModel):
    mentions: list[str] = Field(default_factory=list)
    association_subjects: list[_AssociationSubject] = Field(default_factory=list)
    needs_association_enrichment: bool = False

    @field_validator("mentions")
    @classmethod
    def normalize_mentions(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = value.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        return result


class _SnapshotChanged(RuntimeError):
    pass


class _SnapshotUnavailable(NebulaGraphError):
    pass


class QueryServiceV3:
    DEFAULT_MAX_TOKENS = 8000

    @classmethod
    async def query(
        cls,
        folder_id: str,
        query_text: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        return_context: bool = False,
    ):
        stopwatch = StopWatch(f"QueryServiceV3-{folder_id}")
        with ExitStack() as stack, stopwatch.task("query") as root:
            stack.callback(stopwatch.print)
            with root.task("get_graph"):
                graph = await GraphStorageService.get(folder_id)
                if graph is None:
                    return cls._result(
                        QueryData(answer=GRAPH_FAIL_RESPONSE),
                        "",
                        return_context,
                    )

            with root.task("extract_mentions"):
                extraction = await cls._extract_mentions(query_text)
                mentions = cls._normalize_mentions(query_text, extraction.mentions)
                association_subjects = [
                    subject
                    for subject in extraction.association_subjects
                    if "关联" in query_text
                    and subject.mention in mentions
                    and subject.expected_types
                ][:2]
                if "关联" in query_text and not association_subjects:
                    return cls._result(
                        QueryData(answer=_ASSOCIATION_SUBJECT_REQUIRED),
                        "",
                        return_context,
                    )
                association_subject_types: dict[str, list[str]] = {}
                for subject in association_subjects:
                    expected_types = association_subject_types.setdefault(
                        subject.mention,
                        [],
                    )
                    expected_types.extend(
                        type_name
                        for type_name in subject.expected_types
                        if type_name not in expected_types
                    )
                association_subject_names = [
                    subject.mention for subject in association_subjects
                ]

            for snapshot_attempt in range(2):
                with root.task("resolve_anchors"):
                    anchors, hits, snapshot_version = await cls._resolve_anchors(
                        folder_id,
                        mentions,
                        association_subject_types,
                    )
                if mentions and any(not hits[index] for index in range(len(mentions))):
                    return cls._result(
                        QueryData(answer=QUERY_FAIL_RESPONSE),
                        "",
                        return_context,
                    )

                association = None
                association_context = ""
                association_answer = ""
                if association_subject_names:
                    with root.task("query_association"):
                        association = await cls._execute_association(
                            folder_id=folder_id,
                            subjects=association_subject_names,
                            mentions=mentions,
                            hits=hits,
                            snapshot_version=snapshot_version,
                        )
                    if association == "snapshot_changed":
                        if snapshot_attempt == 1:
                            raise NebulaGraphError(
                                "Knowledge graph snapshot changed repeatedly during query"
                            )
                        await EntityVectorStore.invalidate_cache(folder_id)
                        continue
                    association_answer = cls._association_answer(
                        association,
                        association_subject_names,
                    )
                    association_context = cls._association_context(
                        association,
                        association_answer,
                    )
                    if (
                        association["out_of_scope"]
                        or not association["paths"]
                        or not extraction.needs_association_enrichment
                    ):
                        result = QueryData(
                            answer=association_answer,
                            entities=association["entities"],
                            relations=association["relations"],
                        )
                        return cls._result(
                            result,
                            association_context,
                            return_context,
                        )
                    cls._append_association_anchor(
                        mentions,
                        anchors,
                        hits,
                        association,
                        snapshot_version,
                    )

                execution = await cls._generate_execute_hydrate(
                    timer=root,
                    folder_id=folder_id,
                    query_text=query_text,
                    mentions=mentions,
                    anchors=anchors,
                    hits=hits,
                    snapshot_version=snapshot_version,
                    association_context=association_answer,
                )
                if execution is None:
                    if association:
                        result = QueryData(
                            answer=association_answer,
                            entities=association["entities"],
                            relations=association["relations"],
                        )
                        return cls._result(
                            result,
                            association_context,
                            return_context,
                        )
                    return cls._result(
                        QueryData(answer=QUERY_FAIL_RESPONSE),
                        "",
                        return_context,
                    )
                if execution == "snapshot_changed":
                    if snapshot_attempt == 1:
                        raise NebulaGraphError(
                            "Knowledge graph snapshot changed repeatedly during query"
                        )
                    await EntityVectorStore.invalidate_cache(folder_id)
                    continue

                statement, rows, entities, relations = execution
                if association:
                    entities = cls._merge_by_id(
                        association["entities"],
                        entities,
                    )
                    relations = cls._merge_by_id(
                        association["relations"],
                        relations,
                    )
                with root.task("assemble_context"):
                    context, sources = await cls._assemble_context(
                        statement,
                        rows,
                        entities,
                        relations,
                        query_text,
                        max_tokens,
                    )
                    if association_context:
                        context = f"{association_context}\n\n{context}"
                with root.task("generate_answer"):
                    answer = await cls._generate_answer(context, query_text)
                    if association_answer:
                        answer = f"{association_answer}\n\n{answer}"
                    answer = (
                        f"{answer}\n\n"
                        f"{cls._format_references(entities, relations, sources)}"
                    )
                result = QueryData(
                    answer=answer,
                    entities=entities,
                    relations=relations,
                )
                root.detail(f"folder_id={folder_id}")
                logger.info(
                    "QueryServiceV3 | 查询完成 | folder_id={} mentions={} rows={} entities={} relations={}",
                    folder_id,
                    len(mentions),
                    len(rows),
                    len(entities),
                    len(relations),
                )
                return cls._result(result, context, return_context)

        raise NebulaGraphError("Knowledge graph query did not reach a terminal state")

    @classmethod
    async def _extract_mentions(cls, query_text: str) -> _MentionExtraction:
        response = await ModelFactory.get_model().ainvoke(
            [HumanMessage(content=_MENTION_PROMPT.format(query=query_text))]
        )
        raw = response.content if hasattr(response, "content") else str(response)
        if not isinstance(raw, str):
            raise TypeError(f"Unexpected mention extraction content: {type(raw)!r}")
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Mention extraction did not return a JSON object")
        result = _MentionExtraction.model_validate_json(raw[start : end + 1])
        return result

    @classmethod
    def _normalize_mentions(
        cls,
        query_text: str,
        mentions: list[str],
    ) -> list[str]:
        normalized = [
            mention
            for mention in mentions
            if mention not in _PROJECT_ROLE_MENTIONS
            and mention not in _NON_ENTITY_MENTIONS
        ]
        if any(term in query_text for term in _PROJECT_QUERY_TERMS):
            normalized.append(_PROJECT_ANCHOR_MENTION)
        return _MentionExtraction(mentions=normalized).mentions

    @classmethod
    async def _resolve_anchors(
        cls,
        folder_id: str,
        mentions: list[str],
        association_subject_types: dict[str, list[str]] | None = None,
    ) -> tuple[dict[int, list[str]], dict[int, list[AnchorHit]], str]:
        meta = await EntityVectorStore.metadata(folder_id)
        snapshot_version = str(meta["snapshot_version"])
        hits_by_index: dict[int, list[AnchorHit]] = {}
        anchors: dict[int, list[str]] = {}
        for index, mention in enumerate(mentions):
            if mention == _PROJECT_ANCHOR_MENTION:
                hits = [
                    AnchorHit(
                        vid=str(item["id"]),
                        name=str(item["name"]),
                        type=str(item["type"]),
                        score=1.0,
                        match_type="project_context",
                        snapshot_version=snapshot_version,
                    )
                    for item in meta.get("items", [])
                    if item.get("type") == "项目"
                ][:_ANCHOR_TOP_K]
            else:
                hits = await EntityVectorStore.search(
                    folder_id,
                    mention,
                    _ANCHOR_TOP_K,
                    entity_types=(association_subject_types or {}).get(mention),
                )
            if any(hit.snapshot_version != snapshot_version for hit in hits):
                raise NebulaGraphError("Entity vector snapshot changed during resolution")
            hits_by_index[index] = hits
            anchors[index] = [hit.vid for hit in hits]
        return anchors, hits_by_index, snapshot_version

    @classmethod
    async def _execute_association(
        cls,
        *,
        folder_id: str,
        subjects: list[str],
        mentions: list[str],
        hits: dict[int, list[AnchorHit]],
        snapshot_version: str,
    ) -> dict[str, Any] | str:
        if len(subjects) == 2 and subjects[0] == subjects[1]:
            return {
                "statements": [],
                "paths": [],
                "entities": [],
                "relations": [],
                "associated_vids": set(),
                "out_of_scope": False,
            }
        subject_hits = [hits[mentions.index(subject)] for subject in subjects]
        specs = cls._association_query_specs(subject_hits)
        if len(subject_hits) == 2 and not specs:
            return {
                "statements": [],
                "paths": [],
                "entities": [],
                "relations": [],
                "associated_vids": set(),
                "out_of_scope": True,
            }

        statements: list[str] = []
        paths: list[dict[str, Any]] = []
        async with DistributedLock.acquire(f"kg:{folder_id}"):
            snapshot = await NebulaGraphRepository.get_snapshot(folder_id)
            if snapshot is None or not snapshot.ready:
                raise _SnapshotUnavailable("Knowledge graph snapshot is not ready")
            if snapshot.snapshot_version != snapshot_version:
                return "snapshot_changed"

            for source_vids, target_vids, enterprise_target in specs:
                statement = cls._association_statement(
                    pairwise=len(subject_hits) == 2,
                    enterprise_target=enterprise_target,
                )
                local_anchors = {0: source_vids}
                if target_vids:
                    local_anchors[1] = target_vids
                compiled = compile_statement(
                    statement,
                    local_anchors,
                    max_hops=_ASSOCIATION_MAX_HOPS,
                )
                rows = await NebulaGraphRepository.execute_read(
                    folder_id,
                    compiled,
                    {"folder_key": folder_key(folder_id)},
                )
                statements.append(compiled)
                paths.extend(
                    value
                    for row in rows
                    for value in [row.get("association_path")]
                    if cls._is_path_value(value)
                )
            paths = cls._select_association_paths(
                paths,
                pairwise=len(subject_hits) == 2,
            )
            entities, relations, _ = await cls._hydrate(
                folder_id,
                [{"association_path": path} for path in paths],
                {},
            )
        associated_vids = {
            str(path["nodes"][-1]["vid"])
            for path in paths
            if path.get("nodes")
        }
        return {
            "statements": statements,
            "paths": paths,
            "entities": entities,
            "relations": relations,
            "associated_vids": associated_vids,
            "out_of_scope": False,
        }

    @staticmethod
    def _association_query_specs(
        subject_hits: list[list[AnchorHit]],
    ) -> list[tuple[list[str], list[str], bool]]:
        by_type = [
            {
                type_name: [hit.vid for hit in values if hit.type == type_name]
                for type_name in _ASSOCIATION_ENTITY_TYPES
            }
            for values in subject_hits
        ]
        if len(by_type) == 1:
            specs = []
            if by_type[0]["企业主体"]:
                specs.append((by_type[0]["企业主体"], [], False))
            if by_type[0]["自然人"]:
                specs.append((by_type[0]["自然人"], [], True))
            return specs

        specs = []
        right_all = by_type[1]["企业主体"] + by_type[1]["自然人"]
        if by_type[0]["企业主体"] and right_all:
            specs.append((by_type[0]["企业主体"], right_all, False))
        if by_type[0]["自然人"] and by_type[1]["企业主体"]:
            specs.append(
                (
                    by_type[0]["自然人"],
                    by_type[1]["企业主体"],
                    False,
                )
            )
        return specs

    @staticmethod
    def _association_statement(*, pairwise: bool, enterprise_target: bool) -> str:
        target = "(dst:enterprise)" if enterprise_target else "(dst)"
        if pairwise:
            target_filter = "id(dst) IN [__ANCHOR_1_VIDS__]"
            limit = 1
        else:
            target_filter = "id(dst) NOT IN [__ANCHOR_0_VIDS__]"
            limit = int(os.environ.get("KG_NGQL_MAX_ROWS", "200"))
        return f"""MATCH p = shortestPath(
  (src)-[:{_ASSOCIATION_EDGE_TYPES}*1..{_ASSOCIATION_MAX_HOPS}]-{target}
)
WHERE id(src) IN [__ANCHOR_0_VIDS__]
  AND {target_filter}
  AND id(src) != id(dst)
RETURN p AS association_path,
       id(src) AS `_vid_subject`,
       id(dst) AS `_vid_associated`
LIMIT {limit}"""

    @staticmethod
    def _select_association_paths(
        paths: list[dict[str, Any]],
        *,
        pairwise: bool,
    ) -> list[dict[str, Any]]:
        if not paths:
            return []
        if pairwise:
            return [min(paths, key=lambda path: len(path.get("edges", [])))]
        shortest: dict[str, dict[str, Any]] = {}
        for path in paths:
            nodes = path.get("nodes", [])
            if not nodes:
                continue
            endpoint = str(nodes[-1].get("vid"))
            current = shortest.get(endpoint)
            if current is None or len(path.get("edges", [])) < len(
                current.get("edges", [])
            ):
                shortest[endpoint] = path
        return sorted(
            shortest.values(),
            key=lambda path: (
                len(path.get("edges", [])),
                str(path["nodes"][-1].get("vid")),
            ),
        )

    @classmethod
    def _association_answer(
        cls,
        association: dict[str, Any],
        subjects: list[str],
    ) -> str:
        if association["out_of_scope"]:
            return "自然人与自然人之间不属于本关联关系的判定范围。"
        paths = association["paths"]
        if not paths:
            if len(subjects) == 2:
                return (
                    f"根据当前知识图谱，在最多 {_ASSOCIATION_MAX_HOPS} 跳范围内，"
                    f"未发现{subjects[0]}与{subjects[1]}存在关联关系。"
                )
            return (
                f"根据当前知识图谱，在最多 {_ASSOCIATION_MAX_HOPS} 跳范围内，"
                f"未发现{subjects[0]}的关联方。"
            )

        entity_by_id = {
            entity.id: entity for entity in association["entities"]
        }
        rendered = [cls._format_association_path(path, entity_by_id) for path in paths]
        if len(subjects) == 2:
            return (
                f"{subjects[0]}与{subjects[1]}存在关联关系。"
                f"最短路径：{rendered[0]}。"
            )
        return "关联方及最短路径：\n" + "\n".join(
            f"- {path}" for path in rendered
        )

    @staticmethod
    def _format_association_path(
        path: dict[str, Any],
        entity_by_id: dict[str, Entity],
    ) -> str:
        nodes = path.get("nodes", [])
        edges = path.get("edges", [])
        if not nodes:
            return str(path.get("path_str", ""))
        first_vid = str(nodes[0].get("vid"))
        first_name = (
            entity_by_id[first_vid].name if first_vid in entity_by_id else first_vid
        )
        parts = [first_name]
        for edge, node in zip(edges, nodes[1:]):
            relation_type = RELATION_TYPE_BY_EDGE.get(
                str(edge.get("type")),
                str(edge.get("type")),
            )
            vid = str(node.get("vid"))
            name = entity_by_id[vid].name if vid in entity_by_id else vid
            parts.extend((f"—[{relation_type}]—", name))
        return "".join(parts)

    @staticmethod
    def _association_context(
        association: dict[str, Any],
        answer: str,
    ) -> str:
        statements = "\n\n".join(association["statements"]) or "（未执行）"
        return f"<关联查询语句>\n{statements}\n\n<关联判定>\n{answer}"

    @staticmethod
    def _append_association_anchor(
        mentions: list[str],
        anchors: dict[int, list[str]],
        hits: dict[int, list[AnchorHit]],
        association: dict[str, Any],
        snapshot_version: str,
    ) -> None:
        associated_vids = association["associated_vids"]
        associated_entities = [
            entity
            for entity in association["entities"]
            if entity.id in associated_vids
        ]
        index = len(mentions)
        mentions.append(_ASSOCIATION_MENTION)
        anchors[index] = [entity.id for entity in associated_entities]
        hits[index] = [
            AnchorHit(
                vid=entity.id,
                name=entity.name,
                type=entity.type,
                score=1.0,
                match_type="association_path",
                snapshot_version=snapshot_version,
            )
            for entity in associated_entities
        ]

    @staticmethod
    def _merge_by_id(first: list[Any], second: list[Any]) -> list[Any]:
        merged = {item.id: item for item in first}
        merged.update((item.id, item) for item in second)
        return list(merged.values())

    @classmethod
    async def _generate_execute_hydrate(
        cls,
        *,
        timer: TaskContext,
        folder_id: str,
        query_text: str,
        mentions: list[str],
        anchors: dict[int, list[str]],
        hits: dict[int, list[AnchorHit]],
        snapshot_version: str,
        association_context: str = "",
    ) -> (
        tuple[str, list[dict[str, Any]], list[Entity], list[Relation]]
        | str
        | None
    ):
        feedback = ""
        model = ModelFactory.get_model(key="model_deepseek_v4_flash")
        for attempt in range(1, 3):
            with timer.task(f"generate_ngql[attempt={attempt}]"):
                prompt = cls._ngql_prompt(
                    query_text,
                    mentions,
                    hits,
                    feedback,
                    association_context,
                )
                response = await model.ainvoke(
                    [
                        SystemMessage(content=_NGQL_SYSTEM_PROMPT),
                        HumanMessage(content=prompt),
                    ]
                )
                raw = (
                    response.content
                    if hasattr(response, "content")
                    else str(response)
                )
                statement = str(raw)
            try:
                statement = extract_statement(statement)
                compiled = compile_statement(statement, anchors, max_hops=10)
                async with AsyncExitStack() as lock_stack:
                    with timer.task(f"acquire_lock[attempt={attempt}]"):
                        await lock_stack.enter_async_context(
                            DistributedLock.acquire(f"kg:{folder_id}")
                        )
                    with timer.task(f"execute_ngql[attempt={attempt}]"):
                        snapshot = await NebulaGraphRepository.get_snapshot(folder_id)
                        if snapshot is None or not snapshot.ready:
                            raise _SnapshotUnavailable(
                                "Knowledge graph snapshot is not ready"
                            )
                        if snapshot.snapshot_version != snapshot_version:
                            raise _SnapshotChanged
                        rows = await NebulaGraphRepository.execute_read(
                            folder_id,
                            compiled,
                            {"folder_key": folder_key(folder_id)},
                        )
                    with timer.task(f"hydrate[attempt={attempt}]"):
                        entities, relations, business_rows = await cls._hydrate(
                            folder_id,
                            rows,
                            anchors,
                        )
                return compiled, business_rows, entities, relations
            except _SnapshotChanged:
                return "snapshot_changed"
            except _SnapshotUnavailable:
                raise
            except (NGQLGuardError, NebulaGraphError, KeyError, TypeError, ValueError) as error:
                if cls._is_connection_error(error):
                    raise
                feedback = str(error)
                logger.warning(
                    "QueryServiceV3 | nGQL 生成失败 | folder_id={} attempt={} reason={} stmt={}",
                    folder_id,
                    attempt,
                    feedback,
                    statement,
                )
        return None

    @classmethod
    def _ngql_prompt(
        cls,
        query_text: str,
        mentions: list[str],
        hits: dict[int, list[AnchorHit]],
        feedback: str,
        association_context: str = "",
    ) -> str:
        anchor_lines = []
        for index, mention in enumerate(mentions):
            candidates = ", ".join(
                f"{hit.name.replace(chr(10), ' ')}<{hit.type},{hit.match_type}>"
                for hit in hits.get(index, [])
            )
            anchor_lines.append(
                f"mention[{index}]={mention} → __ANCHOR_{index}_VIDS__ "
                f"(候选: {candidates or '无'})"
            )
        anchors_text = "\n".join(anchor_lines) or "无具名实体锚点"
        retry_text = (
            f"\n上一次语句被拒绝或执行失败，错误如下，请修正：\n{feedback}\n"
            if feedback
            else ""
        )
        association_text = (
            "\n## 已确定的关联关系\n"
            f"{association_context}\n"
            "关联关系已由应用的四跳专用查询确定。关联方锚点只用于查询题目要求的补充信息，禁止重新判定或扩大关联范围。\n"
            if association_context
            else ""
        )
        return f"""## 物理 schema
{render_schema_for_prompt()}

## 锚点
{anchors_text}

## 强制规则
- 起点顶点只能使用提供的 __ANCHOR_N_VIDS__；禁止写实体名或真实 VID。
- 无锚点时，只允许使用 `Tag.folder_key == $folder_key` 的项目级聚合查询。
- 比较必须用 ==，MATCH 属性必须写 `v.tag.property`。
- schema 中 source → target 是入库方向；锚点属于 target 且要找 source 时必须使用 REVERSELY。
- 多跳查询必须显式给出最大跳数，且不得超过 10。管道中的下一跳会过滤掉没有该边的上一跳结果，不要添加问题不需要的跳数。
- kinship_of 和 spouse_of 必须使用 BIDIRECT。
- “作担保”同时覆盖 is_guarantor 和 is_warrantor，查询时应同时包含两种边。
- 同一起点、方向相同的多种边优先写 `OVER edge1, edge2`，不要拆成 UNION。
- 必须使用 UNION 时，两侧输出列的数量、名称和顺序必须完全一致。
- `IN` 右侧必须是集合；锚点占位符应写成 `[__ANCHOR_N_VIDS__]`。
- 非聚合查询最终必须带 `| LIMIT {os.environ.get("KG_NGQL_MAX_ROWS", "200")}` 或更小值。
- 禁止查询或过滤 attributes_json；业务属性只输出实体 `_vid`，由应用回填。
- 所有结果顶点列以 `_vid` 开头，以下划线开头的别名必须加反引号，例如 ``AS `_vid_person` ``。
- 每条结果边输出完整编号列：`_edgeN_src/_edgeN_dst/_edgeN_type/_edgeN_rank`。
- 管道引用内部列时也必须加反引号，例如 ``$-.`_edge1_src` ``。
- 需要传给下一段 GO 的工作列使用不以下划线开头的普通别名；不要臆造 `_legal_rep` 等内部列。
- 聚合可以只输出标量业务列。其他业务标量列不得以下划线开头。
- 只能生成读取语句，最大跳数 10。
{retry_text}
{association_text}
## 方言示例
“比亚迪的法定代表人的亲属有哪些”：
GO FROM __ANCHOR_0_VIDS__ OVER legal_representative REVERSELY
YIELD src(edge) AS legal_rep,
      src(edge) AS `_vid_legal_rep`,
      src(edge) AS `_edge1_src`, dst(edge) AS `_edge1_dst`,
      type(edge) AS `_edge1_type`, rank(edge) AS `_edge1_rank`
| GO FROM $-.legal_rep OVER kinship_of BIDIRECT
YIELD $-.`_edge1_src` AS `_edge1_src`, $-.`_edge1_dst` AS `_edge1_dst`,
      $-.`_edge1_type` AS `_edge1_type`, $-.`_edge1_rank` AS `_edge1_rank`,
      src(edge) AS `_edge2_src`, dst(edge) AS `_edge2_dst`,
      type(edge) AS `_edge2_type`, rank(edge) AS `_edge2_rank`,
      properties($$).name AS name
| LIMIT {os.environ.get("KG_NGQL_MAX_ROWS", "200")}

“借款人的注册资本是多少”：
GO FROM __ANCHOR_0_VIDS__ OVER is_borrower REVERSELY
YIELD src(edge) AS `_vid`,
      src(edge) AS `_edge1_src`, dst(edge) AS `_edge1_dst`,
      type(edge) AS `_edge1_type`, rank(edge) AS `_edge1_rank`
| LIMIT {os.environ.get("KG_NGQL_MAX_ROWS", "200")}

“有人在甲公司和乙公司同时担任高管吗”：
GO FROM __ANCHOR_0_VIDS__ OVER executive_of REVERSELY
YIELD src(edge) AS executive,
      src(edge) AS `_vid_executive`,
      src(edge) AS `_edge1_src`, dst(edge) AS `_edge1_dst`,
      type(edge) AS `_edge1_type`, rank(edge) AS `_edge1_rank`
| GO FROM $-.executive OVER executive_of
WHERE dst(edge) IN [__ANCHOR_1_VIDS__]
YIELD $-.`_vid_executive` AS `_vid_executive`,
      $-.`_edge1_src` AS `_edge1_src`, $-.`_edge1_dst` AS `_edge1_dst`,
      $-.`_edge1_type` AS `_edge1_type`, $-.`_edge1_rank` AS `_edge1_rank`,
      src(edge) AS `_edge2_src`, dst(edge) AS `_edge2_dst`,
      type(edge) AS `_edge2_type`, rank(edge) AS `_edge2_rank`
| LIMIT {os.environ.get("KG_NGQL_MAX_ROWS", "200")}

“保证人的实际控制人及股东”：
GO FROM __ANCHOR_0_VIDS__ OVER is_guarantor REVERSELY
YIELD src(edge) AS guarantor,
      src(edge) AS `_vid_guarantor`,
      src(edge) AS `_edge1_src`, dst(edge) AS `_edge1_dst`,
      type(edge) AS `_edge1_type`, rank(edge) AS `_edge1_rank`
| GO FROM $-.guarantor OVER controller_of, shareholder_of REVERSELY
YIELD $-.`_vid_guarantor` AS `_vid_guarantor`,
      $-.`_edge1_src` AS `_edge1_src`, $-.`_edge1_dst` AS `_edge1_dst`,
      $-.`_edge1_type` AS `_edge1_type`, $-.`_edge1_rank` AS `_edge1_rank`,
      src(edge) AS `_vid_related`,
      src(edge) AS `_edge2_src`, dst(edge) AS `_edge2_dst`,
      type(edge) AS `_edge2_type`, rank(edge) AS `_edge2_rank`
| LIMIT {os.environ.get("KG_NGQL_MAX_ROWS", "200")}

“本项目共涉及几家企业”：
LOOKUP ON enterprise WHERE enterprise.folder_key == $folder_key
YIELD id(vertex) AS `_vid`
| YIELD COUNT(*) AS n

## 用户问题
{query_text}
"""

    @classmethod
    async def _hydrate(
        cls,
        folder_id: str,
        rows: list[dict[str, Any]],
        anchors: dict[int, list[str]],
    ) -> tuple[list[Entity], list[Relation], list[dict[str, Any]]]:
        vids = {
            vid
            for anchor_vids in anchors.values()
            for vid in anchor_vids
        }
        edge_keys: set[EdgeKey] = set()
        business_rows: list[dict[str, Any]] = []

        for row in rows:
            edge_columns: dict[int, dict[str, Any]] = {}
            business: dict[str, Any] = {}
            for name, value in row.items():
                if name.startswith("_vid"):
                    if value is not None:
                        vids.add(str(value))
                    continue
                if cls._is_path_value(value):
                    for node in value["nodes"]:
                        vids.add(str(node["vid"]))
                    for edge in value["edges"]:
                        edge_keys.add(
                            (
                                str(edge["src"]),
                                str(edge["dst"]),
                                str(edge["type"]),
                                int(edge["rank"]),
                            )
                        )
                    business[name] = str(value.get("path_str", ""))
                    business[f"{name}_hops"] = len(value["edges"])
                    continue
                edge_match = _EDGE_COLUMN_PATTERN.fullmatch(name)
                if edge_match:
                    edge_columns.setdefault(int(edge_match.group(1)), {})[
                        edge_match.group(2)
                    ] = value
                    continue
                if (
                    not name.startswith("_")
                    and (
                        value is None
                        or isinstance(value, (bool, int, float, str))
                    )
                ):
                    business[name] = value

            for number, columns in edge_columns.items():
                if set(columns) != {"src", "dst", "type", "rank"}:
                    raise NebulaGraphError(
                        f"Incomplete edge hydration columns for _edge{number}"
                    )
                source = str(columns["src"])
                target = str(columns["dst"])
                vids.update((source, target))
                edge_keys.add(
                    (
                        source,
                        target,
                        str(columns["type"]),
                        int(columns["rank"]),
                    )
                )
            business_rows.append(business)

        entities = await NebulaGraphRepository.fetch_entities(
            folder_id,
            sorted(vids),
        )
        relations = await NebulaGraphRepository.fetch_relations(
            folder_id,
            sorted(edge_keys),
        )
        return entities, relations, business_rows

    @staticmethod
    def _is_path_value(value: Any) -> bool:
        return (
            isinstance(value, dict)
            and isinstance(value.get("nodes"), list)
            and isinstance(value.get("edges"), list)
            and all(
                isinstance(node, dict) and node.get("vid") is not None
                for node in value["nodes"]
            )
            and all(
                isinstance(edge, dict)
                and {"src", "dst", "type", "rank"}.issubset(edge)
                for edge in value["edges"]
            )
        )

    @classmethod
    async def _assemble_context(
        cls,
        statement: str,
        rows: list[dict[str, Any]],
        entities: list[Entity],
        relations: list[Relation],
        query_text: str,
        max_tokens: int,
    ) -> tuple[str, list[str]]:
        tokenizer = Tokenizer(
            model_name=ModelFactory.resolved_model_name_for_key(None)
        )
        parts = [
            f"<查询语句>\n{statement}",
            cls._format_rows(rows),
            cls._format_entities(entities, max_tokens // 3, tokenizer),
            cls._format_relations(relations, max_tokens // 3, tokenizer),
        ]
        source_context, sources = await cls._format_sources(
            entities,
            relations,
            query_text,
            max_tokens // 3,
            tokenizer,
        )
        parts.append(source_context)
        return "\n\n".join(part for part in parts if part), sources

    @staticmethod
    def _format_rows(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "<查询结果>\n（空）"
        if not any(rows):
            return (
                f"<查询结果>\n命中 {len(rows)} 行（无业务标量列；"
                "请结合 <实体> 和 <关系> 回答）"
            )
        lines = [
            ", ".join(f"{name}={value}" for name, value in row.items())
            for row in rows
        ]
        return "<查询结果>\n" + "\n".join(lines)

    @staticmethod
    def _format_entities(
        entities: list[Entity],
        max_tokens: int,
        tokenizer: Tokenizer,
    ) -> str:
        lines: list[str] = []
        used = 0
        for entity in entities:
            attributes = " ".join(
                f"{name}={value}"
                for name, value in entity.attributes.items()
            )
            line = (
                f"({entity.name}{TUPLE_DELIMITER}{entity.type}"
                f"{TUPLE_DELIMITER}{attributes})"
            )
            tokens = tokenizer(line)
            if lines and used + tokens > max_tokens:
                break
            lines.append(line)
            used += tokens
        return "<实体>\n" + "\n".join(lines) if lines else "<实体>\n（空）"

    @staticmethod
    def _format_relations(
        relations: list[Relation],
        max_tokens: int,
        tokenizer: Tokenizer,
    ) -> str:
        lines: list[str] = []
        used = 0
        for relation in relations:
            line = (
                f"({relation.from_entity}{TUPLE_DELIMITER}{relation.type}"
                f"{TUPLE_DELIMITER}{relation.to_entity})"
            )
            tokens = tokenizer(line)
            if lines and used + tokens > max_tokens:
                break
            lines.append(line)
            used += tokens
        return "<关系>\n" + "\n".join(lines) if lines else "<关系>\n（空）"

    @staticmethod
    async def _format_sources(
        entities: list[Entity],
        relations: list[Relation],
        query_text: str,
        max_tokens: int,
        tokenizer: Tokenizer,
    ) -> tuple[str, list[str]]:
        sources: list[str] = []
        seen: set[str] = set()
        for item in [*entities, *relations]:
            for reference in item.references:
                if reference.source not in seen:
                    seen.add(reference.source)
                    sources.append(reference.source)

        try:
            reranker = VectorStoresService.get_reranker()
            if reranker and sources:
                ranking = await reranker.arerank(
                    query_text,
                    sources,
                    top_k=len(sources),
                )
                ranked = [
                    sources[int(item["index"])]
                    for item in ranking
                    if 0 <= int(item["index"]) < len(sources)
                ]
                if ranked:
                    sources = ranked
        except Exception:
            logger.exception("QueryServiceV3 | 原始文本rerank失败，使用原始顺序")

        lines: list[str] = []
        used = 0
        for index, source in enumerate(sources, 1):
            line = f"[{index}] {source}"
            tokens = tokenizer(line)
            if lines and used + tokens > max_tokens:
                break
            lines.append(line)
            used += tokens
        source_context = (
            "<原始文本>\n" + "\n".join(lines)
            if lines
            else "<原始文本>\n（空）"
        )
        return source_context, sources

    @staticmethod
    def _format_references(
        entities: list[Entity],
        relations: list[Relation],
        sources: list[str],
    ) -> str:
        entity_lines: list[str] = []
        for entity in entities:
            attributes = " ".join(
                f"{name}={value}" for name, value in entity.attributes.items()
            )
            entity_lines.append(
                f"({entity.name}{TUPLE_DELIMITER}{entity.type}"
                f"{TUPLE_DELIMITER}{attributes})"
            )
        relation_lines = [
            (
                f"({relation.from_entity}{TUPLE_DELIMITER}{relation.type}"
                f"{TUPLE_DELIMITER}{relation.to_entity})"
            )
            for relation in relations
        ]
        source_lines = [
            f"[{index}] {source}" for index, source in enumerate(sources, 1)
        ]
        blocks = [
            ("参考实体", entity_lines),
            ("参考关系", relation_lines),
            ("参考原文", source_lines),
        ]
        return "\n\n".join(
            f"{title}\n" + ("\n".join(lines) if lines else "（空）")
            for title, lines in blocks
        )

    @staticmethod
    async def _generate_answer(context: str, query_text: str) -> str:
        response = await ModelFactory.get_model().ainvoke(
            [
                SystemMessage(content=_ANSWER_SYSTEM_PROMPT),
                HumanMessage(content=f"{context}\n\n用户问题：\n{query_text}"),
            ]
        )
        return response.content if hasattr(response, "content") else str(response)

    @staticmethod
    def _is_connection_error(error: Exception) -> bool:
        message = str(error)
        return any(
            marker in message
            for marker in (
                "not configured",
                "credentials are not configured",
                "Unable to initialize",
            )
        )

    @staticmethod
    def _result(
        result: QueryData,
        context: str,
        return_context: bool,
    ):
        return (result, context) if return_context else result
