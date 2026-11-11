"""QueryServiceV2 — Agent 驱动的四阶段图查询流水线

Phase 1: LLM 意图理解 → QueryIntent
Phase 2: 结构化图操作（向量搜索 + 属性提取 + 多跳遍历 + 聚合统计）
Phase 3: 按意图类型组装精准上下文
Phase 4: LLM 生成自然语言回答
"""

from typing import Optional
import numpy as np
from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage
from src.ade.knowledge_graph.schema.kg_schema import Entity, QueryData, Relation
from src.ade.knowledge_graph.query.query_intent import QueryIntent
from src.ade.knowledge_graph.service.graph_storage_service import GraphStorageService
from src.ade.knowledge_graph.service.embed_service import EmbedService
from src.ade.knowledge_graph.query.graph_operations import (
    aggregate_count_by_type,
    attribute_extract,
    collect_entities_from_paths,
    entity_lookup,
    expand_relations,
    graph_traverse,
    merge_relations_unique,
    relation_search,
)
from src.ade.knowledge_graph.query.prompt import (
    ENTITY_TYPES_TABLE,
    GRAPH_FAIL_RESPONSE,
    QUERY_FAIL_RESPONSE,
    RELATION_TYPES_TABLE,
    RELATION_TYPE_LIST,
    TUPLE_DELIMITER,
)
from src.utils.model_factory import ModelFactory
from src.utils.tokenizer import Tokenizer
from src.utils.stopwatch import StopWatch
from src.applications.rag.services.vectorstore import VectorStoresService

# Phase 1: 意图理解提示词
QUERY_INTENT_PROMPT = f"""你是一个知识图谱查询意图分析器。根据用户问题，分析查询意图并输出结构化结果。

## 可用实体类型
{ENTITY_TYPES_TABLE}

## 可用关系类型
{RELATION_TYPES_TABLE}

## 查询类型判定规则

| 类型 | 判定条件 | 示例 |
|------|----------|------|
| attribute_query | 询问某实体的具体属性值（注册资本、身份证号、成立日期、注册地址、企业类型、统一社会信用代码等） | "借款人注册资本是多少"、"法定代表人的姓名和年龄" |
| relation_query | 询问实体间的关系，单层即可回答（担保人是谁、股东是谁、法定代表人是谁等） | "担保人是谁"、"股东有哪些" |
| multi_hop | 需要关系穿透/多步关联（股权穿透、关联关系图谱、担保圈等） | "股权穿透信息"、"关联关系图谱" |
| aggregation | 需要统计计数 | "共有几家企业、几个自然人" |
| rag | 以上都不匹配的复杂分析或综合查询 | "企业生产经营是否正常"、"该笔授信的风险缓释措施是否充足" |

## 字段说明
- target_entity_types: 问题涉及的目标实体类型（从可用实体类型中选择）
- target_relation_types: 问题涉及的关系类型（从可用关系类型中选择）
- requested_attributes: 用户明确询问的属性名列表（仅 attribute_query）
- max_hops: 穿透层数（multi_hop 时填 2-3，其他填 0）
- aggregation: 聚合方式（aggregation 时填 count_by_type，其他填 null）

## 重要规则
- 问题中无明确主语时（如"股东概况""保证人信息"），问的就是“项目“实体或“本项目的主借款人“实体。
- 只输出 JSON，不要其他内容

## 用户问题
{{query}}

请以 JSON 格式输出 QueryIntent。"""

# Phase 4: 回答生成提示词
ANSWER_PROMPT = """

## 数据
{context}

## 规则
- 所有回答严格基于提供的数据，禁止编造
- 数据不足时如实告知
- 回答简洁、直接
- 如果用户问题意图不清晰（指代不明、可作多种解读），反问用户

## 用户问题
{query}

请直接回答："""


class QueryServiceV2:
    """新架构查询服务 — 四阶段流水线"""

    DEFAULT_MAX_TOKENS = 8000

    @classmethod
    async def query(
        cls,
        folder_id: str,
        query_text: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        return_context: bool = False,
    ):
        stopwatch = StopWatch(f"QueryServiceV2-{folder_id}")

        with stopwatch.task("query") as main_task:
            # 获取图
            with main_task.task("get_graph"):
                graph = await GraphStorageService.get(folder_id)
                if graph is None:
                    fail_result = QueryData(answer=GRAPH_FAIL_RESPONSE)
                    if return_context:
                        return fail_result, ""
                    return fail_result

            # Phase 1: 意图理解
            with main_task.task("understand_query"):
                intent = await cls._understand_query(query_text)
                logger.info(
                    "QueryServiceV2 | 意图理解完成 | type={}, entity_types={}, relation_types={}, attrs={}, hops={}",
                    intent.query_type,
                    intent.target_entity_types,
                    intent.target_relation_types,
                    intent.requested_attributes,
                    intent.max_hops,
                )

            # 嵌入查询文本
            with main_task.task("embed_query"):
                query_vecs = await EmbedService.embed_texts([query_text])
                query_vec = np.array(query_vecs[0])

            # Phase 2: 图操作
            with main_task.task("graph_operations"):
                graph_result = cls._execute_graph_ops(intent, query_vec, graph)

            # 检查是否有结果
            if not graph_result.get("entities") and not graph_result.get("relations") and not graph_result.get("counts"):
                fail_result = QueryData(answer=QUERY_FAIL_RESPONSE)
                if return_context:
                    return fail_result, ""
                return fail_result

            # Phase 3: 上下文组装
            with main_task.task("assemble_context"):
                context = await cls._assemble_context(
                    graph_result,
                    intent,
                    query_text,
                    max_tokens,
                )

            # Phase 4: LLM 回答
            with main_task.task("llm_answer"):
                answer = await cls._generate_answer(context, query_text)

            main_task.detail(f"folder_id={folder_id}")
            logger.info("QueryServiceV2 | 查询完成 | folder_id={}", folder_id)

        stopwatch.print()

        query_result = QueryData(
            answer=answer,
            entities=graph_result.get("entities", []),
            relations=graph_result.get("relations", []),
        )

        if return_context:
            return query_result, context
        return query_result

    # ========== Phase 1 ==========

    @classmethod
    async def _understand_query(cls, query_text: str) -> QueryIntent:
        """LLM 意图理解，输出结构化 QueryIntent"""
        model = ModelFactory.get_model()
        relation_types_str = "\n".join(f"- {t}" for t in RELATION_TYPE_LIST)
        prompt = QUERY_INTENT_PROMPT.format(
            relation_types=relation_types_str,
            query=query_text,
        )

        try:
            if ModelFactory.is_deepseek_model_for_key(None):
                structured = model.with_structured_output(QueryIntent, method="json_mode")
            else:
                structured = model.with_structured_output(QueryIntent)
            intent = await structured.ainvoke([HumanMessage(content=prompt)])
            if not isinstance(intent, QueryIntent):
                raise ValueError(f"Unexpected type: {type(intent)}")
            return intent
        except Exception:
            logger.exception("QueryServiceV2 | 意图理解失败，回退到 rag")
            return QueryIntent(query_type="rag")

    # ========== Phase 2 ==========

    @classmethod
    def _execute_graph_ops(
        cls,
        intent: QueryIntent,
        query_vec: np.ndarray,
        graph,
    ) -> dict:
        """按 query_type 路由到对应图操作"""
        result: dict = {"entities": [], "relations": []}

        entities = entity_lookup(query_vec, graph.entities, intent.target_entity_types or None)
        result["entities"] = entities

        if intent.query_type == "attribute_query":
            # 精准属性提取
            attrs_map: dict[str, dict] = {}
            for entity in entities:
                attrs = attribute_extract(entity, intent.requested_attributes)
                if attrs:
                    attrs_map[entity.name] = attrs
            result["attributes"] = attrs_map
            # 收集这些实体的来源文本
            result["relations"] = expand_relations(entities, graph.relations)

        elif intent.query_type == "relation_query":
            vec_relations = relation_search(query_vec, graph.relations, intent.target_relation_types or None)
            expanded = expand_relations(entities, graph.relations, intent.target_relation_types or None)
            result["relations"] = merge_relations_unique(vec_relations, expanded)

        elif intent.query_type == "multi_hop":
            paths = graph_traverse(
                entities,
                graph.relations,
                intent.target_relation_types or None,
                max_hops=intent.max_hops,
            )
            result["paths"] = paths
            # 收集路径涉及的实体和关系
            path_entities = collect_entities_from_paths(paths, graph.entities)
            # 合并原有实体和路径实体
            seen_ids = {e.id for e in entities}
            for e in path_entities:
                if e.id not in seen_ids:
                    seen_ids.add(e.id)
                    entities.append(e)
            result["entities"] = entities
            # 收集关系
            vec_relations = relation_search(query_vec, graph.relations, intent.target_relation_types or None)
            expanded = expand_relations(entities, graph.relations, intent.target_relation_types or None)
            result["relations"] = merge_relations_unique(vec_relations, expanded)

        elif intent.query_type == "aggregation":
            counts = aggregate_count_by_type(graph.entities)
            result["counts"] = counts


        else:  # rag
            vec_relations = relation_search(query_vec, graph.relations)
            expanded = expand_relations(entities, graph.relations)
            result["relations"] = merge_relations_unique(vec_relations, expanded)

        return result

    # ========== Phase 3 ==========

    @classmethod
    async def _assemble_context(
        cls,
        graph_result: dict,
        intent: QueryIntent,
        query_text: str,
        max_tokens: int,
    ) -> str:
        """按意图类型组装精准上下文"""
        tokenizer = Tokenizer(model_name=ModelFactory.resolved_model_name_for_key(None))
        context_parts: list[str] = []

        if intent.query_type == "attribute_query":
            context_parts.append(cls._format_attribute_context(graph_result))
        elif intent.query_type == "multi_hop":
            context_parts.append(cls._format_path_context(graph_result))
            context_parts.append(cls._format_entity_context(graph_result.get("entities", []), max_tokens // 4, tokenizer))
            context_parts.append(cls._format_relation_context(graph_result.get("relations", []), max_tokens // 4, tokenizer))
        elif intent.query_type == "aggregation":
            context_parts.append(cls._format_aggregation_context(graph_result))
        else:
            # relation_query / rag: 实体 + 关系
            context_parts.append(cls._format_entity_context(graph_result.get("entities", []), max_tokens // 3, tokenizer))
            context_parts.append(cls._format_relation_context(graph_result.get("relations", []), max_tokens // 3, tokenizer))

        # 收集并 rerank 原始文本
        source_context = await cls._build_source_context(
            graph_result.get("entities", []),
            graph_result.get("relations", []),
            query_text,
            max_tokens // 2,
            tokenizer,
        )
        if source_context:
            context_parts.append(source_context)

        return "\n\n".join(p for p in context_parts if p)

    @classmethod
    def _format_attribute_context(cls, graph_result: dict) -> str:
        """格式化属性查询的上下文"""
        attrs_map = graph_result.get("attributes", {})
        if not attrs_map:
            return ""

        lines = ["<目标实体>"]
        entities = graph_result.get("entities", [])
        name_to_type = {e.name: e.type for e in entities}

        for name, attrs in attrs_map.items():
            entity_type = name_to_type.get(name, "")
            lines.append(f"实体名称: {name}")
            lines.append(f"类型: {entity_type}")
            lines.append("属性:")
            for k, v in attrs.items():
                lines.append(f"  {k}: {v}")
            lines.append("")

        return "\n".join(lines)

    @classmethod
    def _format_path_context(cls, graph_result: dict) -> str:
        """格式化多跳穿透路径"""
        paths = graph_result.get("paths", [])
        if not paths:
            return ""

        lines = ["<穿透路径>"]
        for i, path in enumerate(paths, 1):
            segments = [f"{src} -[{rel}]-> {dst}" for src, rel, dst in path]
            lines.append(f"路径{i}: {' → '.join(segments)}")

        return "\n".join(lines)

    @classmethod
    def _format_aggregation_context(cls, graph_result: dict) -> str:
        """格式化聚合统计结果"""
        counts = graph_result.get("counts", {})
        if not counts:
            return ""

        lines = ["<统计结果>"]
        for entity_type, count in sorted(counts.items()):
            lines.append(f"{entity_type}: {count} 个")

        return "\n".join(lines)

    @classmethod
    def _format_entity_context(
        cls,
        entities: list[Entity],
        max_tokens: int,
        tokenizer: Tokenizer,
    ) -> str:
        """格式化实体上下文（旧格式兼容）"""
        if not entities:
            return ""
        lines = []
        used = 0
        for e in entities:
            attrs = " ".join(f"{k}={v}" for k, v in e.attributes.items()) if e.attributes else ""
            line = f"({e.name}{TUPLE_DELIMITER}{e.type}{TUPLE_DELIMITER}{attrs})"
            line_tokens = tokenizer(line)
            if used + line_tokens > max_tokens and lines:
                break
            lines.append(line)
            used += line_tokens
        if not lines:
            return ""
        return "<实体>\n" + "\n".join(lines)

    @classmethod
    def _format_relation_context(
        cls,
        relations: list[Relation],
        max_tokens: int,
        tokenizer: Tokenizer,
    ) -> str:
        """格式化关系上下文"""
        if not relations:
            return ""
        lines = []
        used = 0
        for r in relations:
            line = f"({r.from_entity}{TUPLE_DELIMITER}{r.type}{TUPLE_DELIMITER}{r.to_entity})"
            line_tokens = tokenizer(line)
            if used + line_tokens > max_tokens and lines:
                break
            lines.append(line)
            used += line_tokens
        if not lines:
            return ""
        return "<关系>\n" + "\n".join(lines)

    @classmethod
    async def _build_source_context(
        cls,
        entities: list[Entity],
        relations: list[Relation],
        query_text: str,
        max_tokens: int,
        tokenizer: Tokenizer,
    ) -> str:
        """收集并 rerank 原始文本"""
        all_sources: list[str] = []
        seen: set[str] = set()
        for e in entities:
            for ref in e.references:
                if ref.source not in seen:
                    seen.add(ref.source)
                    all_sources.append(ref.source)
        for r in relations:
            for ref in r.references:
                if ref.source not in seen:
                    seen.add(ref.source)
                    all_sources.append(ref.source)

        if not all_sources:
            return ""

        # Rerank
        reranked = all_sources
        try:
            reranker = VectorStoresService.get_reranker()
            if reranker and query_text:
                order = await reranker.arerank(query_text, all_sources, top_k=len(all_sources))
                if order:
                    reranked = []
                    for item in order:
                        try:
                            idx = int(item.get("index"))
                            if 0 <= idx < len(all_sources):
                                reranked.append(all_sources[idx])
                        except Exception:
                            continue
        except Exception as e:
            logger.warning("QueryServiceV2 | rerank失败，使用原始顺序 | error={}", e)

        # Token 截断
        lines = []
        used = 0
        for i, source in enumerate(reranked):
            line = f"[{i + 1}] {source}"
            line_tokens = tokenizer(line)
            if used + line_tokens > max_tokens and lines:
                break
            lines.append(line)
            used += line_tokens

        if not lines:
            return ""
        return "<原始文本>\n" + "\n".join(lines)

    # ========== Phase 4 ==========

    @classmethod
    async def _generate_answer(cls, context: str, query_text: str) -> str:
        """LLM 生成自然语言回答"""
        model = ModelFactory.get_model()
        prompt = ANSWER_PROMPT.format(context=context, query=query_text)
        messages = [
            SystemMessage(content="你是专业的知识图谱问答助手，基于提供的结构化数据回答用户问题。"),
            HumanMessage(content=prompt),
        ]
        response = await model.ainvoke(messages)
        return response.content if hasattr(response, "content") else str(response)
