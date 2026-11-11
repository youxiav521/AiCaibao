from typing import List, Optional, Set
import re
import numpy as np
from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage

from src.ade.knowledge_graph.schema.kg_schema import Entity, QueryData, Relation
from src.ade.knowledge_graph.service.graph_storage_service import GraphStorageService
from src.ade.knowledge_graph.service.embed_service import EmbedService
from src.ade.knowledge_graph.query.prompt import (
    GRAPH_FAIL_RESPONSE,
    QUERY_FAIL_RESPONSE,
    QUERY_SYSTEM_PROMPT,
    TUPLE_DELIMITER,
)
from src.utils.model_factory import ModelFactory
from src.utils.tokenizer import Tokenizer
from src.utils.stopwatch import StopWatch
from src.applications.rag.services.vectorstore import VectorStoresService
from src.applications.rag.utils.text_normalize import add_spacing_for_tokenizer


# 关系类型触发词映射：关系类型 -> 相关触发词列表
RELATION_TYPE_TRIGGERS = {
    "主借款人": ["借款人","申请人","主借款人","贷款人", "贷款主体"],
    "保证人": ["保证人", "担保", "担保人", "保证"],
    "质/抵押人": ["抵押", "质押", "抵押人", "质押人","出质", "抵押物"],
    "配偶关系": ["配偶", "夫妻", "婚姻", "已婚", "丈夫", "妻子"],
    "法定代表": ["法定代表", "法定代表人", "法人", "法人代表"],
    "高管任职": ["高管", "管理人","管理层","任职", "董事", "经理"],
    "实际控制": ["实际控制", "实际控制人", "控制人", "控股"],
    "供应商关系": ["供应商", "供应", "供货","合作"]
}


def _has_project_reference(query: str) -> tuple[Optional[str], bool]:
    """检查查询是否引用了当前项目（此项目/本项目/该项目/此任务/本任务/此尽调/本尽调等），返回 (匹配文本, 是否匹配)"""
    project_patterns = [r"(此|本|该|这个)?(借贷|贷款|项目|任务|尽调)",]
    for pattern in project_patterns:
        m = re.search(pattern, query)
        if m:
            return m.group(), True
    return None, False
    
async def extract_entity_keywords(query: str, entity_names: List[str]) -> List[str]:
    """
    使用 LLM 从查询的"镂空片段"中提取实体关键词。
    只提取实质性实体名，忽略"公司""情况""状态"等泛化无意义词。
    """
    if not query or not query.strip():
        return []

    model = ModelFactory.get_model()
    prompt = f"""你是一个实体关键词提取器。给定一个问题的"镂空片段"（已移除项目引用和关系触发词），从中提取出包含实质性实体名称的关键词。

规则：
- 只提取包含具体实体名称的关键词（如人名、公司名、地名等专有名词）
- 不提取"公司""情况""状态""信息""方式""方法""结果""原因""影响""分析"等泛化无意义词
- 不提取"的""是""了""吗""和""与""或""在""有"等虚词/介词
- 如果没有实质性实体，返回空数组 []
- 按原文出现顺序输出

输入片段：{query}

请以 JSON 数组格式输出，只输出数组，不要其他内容。例如：["惠州市礼光新能源开发有限公司", "焦雪峰"]"""

    messages = [HumanMessage(content=prompt)]
    response = await model.ainvoke(messages)
    content = response.content if hasattr(response, "content") else str(response)
    content = content.strip()

    # 解析 JSON，兼容 markdown 代码块包裹
    try:
        import json
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
        keywords = json.loads(content)
        if not isinstance(keywords, list):
            keywords = []
        result = [str(k).strip() for k in keywords if k and str(k).strip()]
    except Exception:
        logger.warning("QueryService | LLM关键词提取解析失败 | content={}", content[:200])
        result = []

    logger.debug(
        "QueryService | LLM实体关键词提取完成 | query={}, keywords={}",
        query,
        result,
    )

    return result


class QueryService:

    DEFAULT_MAX_TOKENS = 8000  # 默认最大上下文token数，同比例放大为96000
    ENTITY_RATIO = 1  # 实体部分token比例
    RELATION_RATIO = 1  # 关系部分token比例
    SOURCE_RATIO = 2  # 原始文本部分token比例

    @classmethod
    def _detect_relation_types(cls, query_text: str) -> tuple[Set[str], List[str]]:
        """
        根据查询文本中的触发词检测可能的关系类型
        返回 (匹配到的关系类型集合, 匹配到的触发词列表)
        """
        detected_types = set()
        detected_triggers = []
        query_lower = query_text.lower()
        
        for relation_type, triggers in RELATION_TYPE_TRIGGERS.items():
            # 按长度降序匹配，确保优先命中最长触发词（如"主借款人"优先于"借款人"）
            for trigger in sorted(triggers, key=len, reverse=True):
                if trigger.lower() in query_lower:
                    detected_types.add(relation_type)
                    detected_triggers.append(trigger)
                    break
                    
        return detected_types, detected_triggers

    @classmethod
    async def query(
        cls,
        folder_id: str,
        query_text: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        return_context: bool = False,
    ):
        stopwatch = StopWatch(f"QueryService-{folder_id}")
        
        with stopwatch.task("query") as main_task:
            with main_task.task("get_graph") as t:
                graph = await GraphStorageService.get(folder_id)
                if graph is None:
                    return QueryData(answer=GRAPH_FAIL_RESPONSE)

            matched_entities = []
            project_entity = next((e for e in graph.entities if e.type == "项目"), None)
            project_fragment, has_project_ref = _has_project_reference(query_text)

            # 检测查询中的关系类型触发词
            with main_task.task("detect_relation_types"):
                detected_relation_types, detected_triggers = cls._detect_relation_types(query_text)
                if detected_relation_types:
                    logger.info(
                        "QueryService | 检测到关系类型触发词 | types={}",
                        ", ".join(detected_relation_types),
                    )

            # 项目实体：查询含项目引用，或涉及核心借贷关系（主借款人/保证人/抵质押）时加入
            core_loan_types = {"主借款人", "保证人", "质/抵押人"}
            if project_entity and (has_project_ref or detected_relation_types & core_loan_types):
                matched_entities.append(project_entity)
                logger.info(
                    "QueryService | 优先加入项目实体 | entity={}",
                    project_entity.name,
                )

            # 从 query_text 中移除已匹配的项目引用和关系触发词，避免干扰实体关键词提取
            fragments = [project_fragment] if project_fragment else []
            fragments += detected_triggers
            cleaned_query = query_text
            for fragment in fragments:
                cleaned_query = cleaned_query.replace(fragment, "")

            # 提取实体关键词并搜索
            with main_task.task("extract_entity_keywords") as t:
                entity_names = [entity.name for entity in graph.entities]
                entity_keywords = await extract_entity_keywords(cleaned_query, entity_names)
            
            # 搜索其他相关实体
            if entity_keywords:
                with main_task.task("keyword_entity_search") as t:
                    keyword_matched = await cls._keyword_based_entity_search(
                        keywords=entity_keywords,
                        entities=graph.entities,
                    )
                    t.detail(f"keywords={len(entity_keywords)}, entities={len(keyword_matched)}")
                    # 去重添加
                    seen_ids = {e.id for e in matched_entities}
                    for entity in keyword_matched:
                        if entity.id not in seen_ids:
                            matched_entities.append(entity)
            
            # 步骤2：检索关系
            with main_task.task("embed_query_text") as t:
                query_vecs = await EmbedService.embed_texts([query_text])
                query_vec = np.array(query_vecs[0])
                
            # 向量搜索关系，取所有匹配的关系
            with main_task.task("vector_search_relations"):
                matched_relations_vec = cls._vector_search_relations(query_vec, graph.relations, detected_relation_types)
            
            # 合并关系
            seen_relations = set()
            matched_relations = []
            
            # 先添加向量搜索到的关系
            for r in matched_relations_vec:
                if r.id not in seen_relations:
                    seen_relations.add(r.id)
                    matched_relations.append(r)
            
             # 如果有匹配实体，再添加与它们关联的关系
            if matched_entities:
                with main_task.task("expand_relations") as t:
                    # 对实体按degree倒序排序
                    matched_entities_sorted = sorted(matched_entities, key=lambda e: e.degree, reverse=True)
                    # 获取与匹配实体关联的所有关系，按实体degree顺序加载
                    expanded_relations = cls._expand_relations(matched_entities_sorted, graph.relations, detected_relation_types)
                    # 添加其他关系
                    for r in expanded_relations:
                        if r.id not in seen_relations:
                            seen_relations.add(r.id)
                            matched_relations.append(r)
                    t.detail(f"relations={len(expanded_relations)}")
            
            # 如果既没有实体也没有关系，才返回失败
            if not matched_entities and not matched_relations:
                return QueryData(answer=QUERY_FAIL_RESPONSE)

            # 步骤3：构建带token分配的上下文
            with main_task.task("build_context") as t:
                tokenizer = Tokenizer(model_name=ModelFactory.resolved_model_name_for_key(None))
                context, context_data = await cls._build_context_with_token_allocation(
                    entities=matched_entities,
                    relations=matched_relations,
                    max_total_tokens=max_tokens,
                    tokenizer=tokenizer,
                    query_text=query_text,
                )
                t.detail(f"entities={len(matched_entities)}, relations={len(matched_relations)}")

            logger.info(
                "QueryService | 上下文构建完成 | entities={} relations={}",
                len(matched_entities), len(matched_relations),
            )

            # 步骤4：构建最终的对话消息
            with main_task.task("llm_invoke") as t:
                model = ModelFactory.get_model()
                system_prompt = QUERY_SYSTEM_PROMPT.format(context_data=context)
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=query_text),
                ]
                response = await model.ainvoke(messages)
                result = response.content if hasattr(response, "content") else str(response)

            main_task.detail(f"folder_id={folder_id}")
            logger.info("QueryService | 查询完成 | folder_id={}", folder_id)
        
        stopwatch.print()
        
        query_result = QueryData(
            answer=result,
            entities=matched_entities,
            relations=matched_relations,
        )
        
        if return_context:
            return query_result, context
        return query_result

    @classmethod
    async def _build_context_with_token_allocation(
        cls,
        entities: List[Entity],
        relations: List[Relation],
        max_total_tokens: int,
        tokenizer: Tokenizer,
        query_text: str = "",
    ):
        """
        构建上下文，并按1:1:2的比例分配token
        返回 (context_str, context_data_dict)
        """
        # 实体保持向量搜索返回的topk顺序，不再重新排序
        entities_sorted = entities
        # 关系已经按优先级排序（向量搜索top_k在前，然后按degree排序）
        relations_sorted = relations

        # 收集所有原始文本source
        all_sources = []
        seen_sources = set()
        for e in entities_sorted:
            for ref in e.references:
                if ref.source not in seen_sources:
                    seen_sources.add(ref.source)
                    all_sources.append(ref.source)
        for r in relations_sorted:
            for ref in r.references:
                if ref.source not in seen_sources:
                    seen_sources.add(ref.source)
                    all_sources.append(ref.source)

        # 对原始文本source进行rerank
        reranked_sources = all_sources
        try:
            reranker = VectorStoresService.get_reranker()
            if reranker and all_sources and query_text:
                order = await reranker.arerank(query_text, all_sources, top_k=len(all_sources))
                if order:
                    reranked_sources = []
                    for item in order:
                        try:
                            doc_index = int(item.get("index"))
                            if 0 <= doc_index < len(all_sources):
                                reranked_sources.append(all_sources[doc_index])
                        except Exception:
                            continue
                    logger.info(
                        "QueryService | 原始文本rerank完成 | original_count={}, reranked_count={}",
                        len(all_sources),
                        len(reranked_sources),
                    )
        except Exception as e:
            logger.warning(
                "QueryService | 原始文本rerank失败，使用原始顺序 | error={}",
                e,
            )

        # 按1:1:2比例分配token
        total_ratio = cls.ENTITY_RATIO + cls.RELATION_RATIO + cls.SOURCE_RATIO
        max_entity_tokens = int(max_total_tokens * cls.ENTITY_RATIO / total_ratio)
        max_relation_tokens = int(max_total_tokens * cls.RELATION_RATIO / total_ratio)
        max_source_tokens = max_total_tokens - max_entity_tokens - max_relation_tokens

        # 构建实体部分（不含source）
        entity_context, entity_used_tokens = cls._build_entity_context(
            entities_sorted,
            max_entity_tokens,
            tokenizer,
        )

        # 构建关系部分（不含source）
        relation_context, relation_used_tokens = cls._build_relation_context(
            relations_sorted,
            max_relation_tokens,
            tokenizer,
        )

        # 构建原始文本部分（使用rerank后的顺序）
        source_context, source_used_tokens = cls._build_source_context(
            reranked_sources,
            max_source_tokens,
            tokenizer,
        )

        # 组装最终上下文
        context_parts = []
        if entity_context:
            context_parts.append(f"<实体>\n{entity_context}")
        if relation_context:
            context_parts.append(f"<关系>\n{relation_context}")
        if source_context:
            context_parts.append(f"<原始文本>\n{source_context}")

        final_context = "\n\n".join(context_parts)

        # 记录实际使用情况
        total_used = entity_used_tokens + relation_used_tokens + source_used_tokens
        logger.info(
            "QueryService | 实际Token使用 | total={} / {}",
            total_used,
            max_total_tokens,
        )

        return final_context, {
            "entities": entities_sorted,
            "relations": relations_sorted,
            "sources": reranked_sources,
        }

    @classmethod
    def _build_entity_context(
        cls,
        entities: List[Entity],
        max_tokens: int,
        tokenizer: Tokenizer,
    ):
        """构建实体上下文，不含source"""
        entity_lines = []
        used_tokens = 0
        for e in entities:
            attrs = " ".join(f"{k}={v}" for k, v in e.attributes.items()) if e.attributes else ""
            line = f"({e.name}{TUPLE_DELIMITER}{e.type}{TUPLE_DELIMITER}{attrs})"
            line_tokens = tokenizer(line)
            if used_tokens + line_tokens > max_tokens and entity_lines:
                break
            entity_lines.append(line)
            used_tokens += line_tokens
        return "\n".join(entity_lines), used_tokens

    @classmethod
    def _build_relation_context(
        cls,
        relations: List[Relation],
        max_tokens: int,
        tokenizer: Tokenizer,
    ):
        """构建关系上下文，不含source"""
        relation_lines = []
        used_tokens = 0
        for r in relations:
            line = f"({r.from_entity}{TUPLE_DELIMITER}{r.type}{TUPLE_DELIMITER}{r.to_entity})"
            line_tokens = tokenizer(line)
            if used_tokens + line_tokens > max_tokens and relation_lines:
                break
            relation_lines.append(line)
            used_tokens += line_tokens
        return "\n".join(relation_lines), used_tokens

    @classmethod
    def _build_source_context(
        cls,
        sources: List[str],
        max_tokens: int,
        tokenizer: Tokenizer,
    ):
        """构建原始文本上下文"""
        source_lines = []
        used_tokens = 0
        for i, source in enumerate(sources):
            line = f"[{i+1}] {source}"
            line_tokens = tokenizer(line)
            if used_tokens + line_tokens > max_tokens and source_lines:
                break
            source_lines.append(line)
            used_tokens += line_tokens
        return "\n".join(source_lines), used_tokens

    # @classmethod
    # def _vector_search_entities(cls, query_vec: np.ndarray, entities: List[Entity], top_k: int) -> List[Entity]:
    #     """对实体进行向量搜索"""
    #     scored = []
    #     for entity in entities:
    #         if entity.embed is None:
    #             continue
    #         emb = np.array(entity.embed)
    #         score = float(np.dot(query_vec, emb))
    #         scored.append((score, entity))
    #     scored.sort(key=lambda x: x[0], reverse=True)
        
    #     # 应用截止逻辑：如果下一个比前一个得分低20%，就停止
    #     result = []
    #     for i in range(min(top_k, len(scored))):
    #         if i > 0:
    #             prev_score = scored[i-1][0]
    #             curr_score = scored[i][0]
    #             if curr_score < prev_score * 0.8:
    #                 break
    #         result.append(scored[i][1])
        
    #     return result

    @classmethod
    async def _keyword_based_entity_search(
        cls,
        keywords: List[str],
        entities: List[Entity],
    ) -> List[Entity]:
        """
        基于关键词的实体搜索：
        1. 对每个关键词进行 Embedding
        2. 计算所有关键词与所有实体的相似度
        3. 应用 80% 截止逻辑
        """
        if not keywords or not entities:
            return []
        
        logger.info(
            "QueryService | 开始关键词实体搜索 | keywords={}",
            keywords,
        )
        
        # 获取所有关键词的 embedding
        keyword_vecs = await EmbedService.embed_texts(keywords)
        
        # 计算所有关键词与所有实体的相似度
        all_scores = []
        for keyword, keyword_vec in zip(keywords, keyword_vecs):
            keyword_emb = np.array(keyword_vec)
            for entity in entities:
                if entity.embed is None:
                    continue
                entity_emb = np.array(entity.embed)
                score = float(np.dot(entity_emb, keyword_emb))
                all_scores.append((score, entity))
        
        # 按相似度降序排序
        all_scores.sort(key=lambda x: x[0], reverse=True)
        
        # 去重：保留每个实体的最高得分
        unique_scores = []
        seen_ids = set()
        for score, entity in all_scores:
            if entity.id not in seen_ids:
                seen_ids.add(entity.id)
                unique_scores.append((score, entity))
        
        # 应用截止逻辑：如果比第一个得分低20%，就停止
        result = []
        first_score = unique_scores[0][0] if unique_scores else 0
        for i in range(len(unique_scores)):
            curr_score = unique_scores[i][0]
            if curr_score < first_score * 0.8:
                break
            result.append(unique_scores[i][1])
        
        logger.info(
            "QueryService | 关键词实体搜索完成 | entities_count={}",
            len(result),
        )
        
        return result

    @classmethod
    def _vector_search_relations(cls, query_vec: np.ndarray, relations: List[Relation], relation_types: Optional[Set[str]] = None) -> List[Relation]:
        """对关系进行向量搜索，可按关系类型过滤，返回所有匹配的关系"""
        scored = []
        if relation_types:
            relations = [r for r in relations if r.type in relation_types]
        for relation in relations:
            if relation.embed is None:
                continue
            emb = np.array(relation.embed)
            score = float(np.dot(query_vec, emb))
            scored.append((score, relation))
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # 应用截止逻辑：如果比第一个得分低20%，就停止
        result = []
        first_score = scored[0][0] if scored else 0
        for i in range(len(scored)):
            curr_score = scored[i][0]
            if curr_score < first_score * 0.8:
                break
            result.append(scored[i][1])
        
        return result

    @classmethod
    def _expand_relations(
        cls, 
        matched_entities: List[Entity], 
        all_relations: List[Relation],
        relation_types: Optional[Set[str]] = None
    ) -> List[Relation]:
        seen_relation_ids = set()
        filtered_relations = []
        
        # 按实体顺序依次加载它们的关系
        for entity in matched_entities:
            for r in all_relations:
                # 如果已经添加过这个关系，跳过
                if r.id in seen_relation_ids:
                    continue
                    
                # 1. 检查是否关联到当前实体
                if r.from_entity != entity.name and r.to_entity != entity.name:
                    continue
                    
                # 2. 如果指定了关系类型，进行过滤
                if relation_types and r.type not in relation_types:
                    continue
                    
                # 添加关系
                seen_relation_ids.add(r.id)
                filtered_relations.append(r)
            
        return filtered_relations
