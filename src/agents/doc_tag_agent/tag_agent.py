"""基于LangGraph的文档标签匹配Agent"""

from typing import List, Dict, Any, Optional, Set
import json
import asyncio
import re
from loguru import logger

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from src.utils.model_factory import ModelFactory
from src.agents.content_generate_agent.input_models import DocumentItems
from src.agents.content_generate_agent.hierarchical_summary.tree_builder import build_tree, Node
from src.agents.content_generate_agent.hierarchical_summary.summary_generator import SummaryGenerator
from src.applications.rag.schemas import BatchQueryRequest, QuestionItem
from src.applications.rag.services.query_service import run_batch_query


# Agent状态定义（单标签流转用）
class TagAgentState(MessagesState):
    """单标签匹配Agent状态"""
    tag: Dict[str, str]                 # 当前正在判断的标签
    matched_tags: List[str]             # 已匹配的标签ID（通常只包含当前标签或为空）
    is_complete: bool                   # 是否完成当前标签的判断


# 工具函数

def _get_config_value(config: RunnableConfig, key: str, default: Any = None) -> Any:
    """从RunnableConfig中获取配置值"""
    if not config:
        return default
    return config.get("configurable", {}).get(key, default)


def _remove_think_blocks(text: str) -> str:
    """移除 LLM 思考模式的 <think>...</think> 标签"""
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return re.sub(r"^\s*\n", "", text, flags=re.MULTILINE).strip()


# 工具函数
@tool
def read_document_page(page_no: int, config: RunnableConfig) -> str:
    """
    读取指定页码的文档内容
    
    Args:
        page_no: 页码（从1开始）
        config: 运行时配置（自动注入）
        
    Returns:
        str: 该页的文本内容摘要
    """
    doc_content = _get_config_value(config, "doc_content", [])
    if not doc_content:
        return "错误: 上下文未初始化或文档内容为空"
    
    # minerU格式使用'page'字段
    page_content = [elem for elem in doc_content if elem.get('page') == page_no]
    
    if not page_content:
        return f"页码 {page_no} 无内容"
    
    # 记录已阅读页码
    read_pages: Set[int] = _get_config_value(config, "read_pages")
    if read_pages is not None:
        read_pages.add(page_no)
    
    # 合并该页的所有内容元素
    content_parts = []
    for elem in page_content:
        elem_type = elem.get('type', 'text')
        # 优先使用value字段，其次使用md字段
        text = elem.get('value') or elem.get('md', '')
        if text:
            if elem_type == 'text' and elem.get('headingLevel'):
                content_parts.append(f"【标题{elem.get('headingLevel')}级】{text}")
            elif elem_type == 'table':
                caption = elem.get('table_caption', '')
                content_parts.append(f"【表格】{caption if caption else text[:100]}")
            elif elem_type == 'image':
                caption = elem.get('image_caption', '')
                content_parts.append(f"【图片】{caption if caption else '(图片)'}")
            elif elem_type == 'list':
                content_parts.append(f"【列表】{text[:100]}")
            else:
                content_parts.append(text)
    
    content = "\n".join(content_parts)
    return f"第{page_no}页内容:\n{content[:800]}..." if len(content) > 800 else f"第{page_no}页内容:\n{content}"


@tool
def summarize_document(max_pages: int = 0, config: RunnableConfig = None) -> str:
    """对文档进行摘要，用于在打标签前快速了解全文要点。

    默认会对全文进行摘要；只有在明确传入正整数 `max_pages` 时，才会截取前若干页。

    Args:
        max_pages: 可选，限制参与摘要的最大页数（从第一页开始截取）。小于等于0表示使用全文。
        config: 运行时配置（自动注入）

    Returns:
        str: 生成的摘要文本；异常时返回错误信息或简化摘要片段。
    """
    doc_content = _get_config_value(config, "doc_content", [])
    if not doc_content:
        return "文档内容为空，无法生成摘要。"

    # 限制页数（默认不过滤，使用全文）
    pages = sorted({d.get('page') for d in doc_content if d.get('page') is not None})
    if max_pages is not None and max_pages > 0 and len(pages) > max_pages:
        selected_pages = set(pages[:max_pages])
        filtered = [d for d in doc_content if d.get('page') in selected_pages]
    else:
        filtered = doc_content

    try:
        # 使用已有的 DocumentItems + 分层摘要能力
        doc_items = DocumentItems.model_validate(filtered)
    except Exception as e:
        logger.error(f"Summarization: DocumentItems 解析失败: {e}")
        # 兜底：直接拼接前若干片段作为简化摘要
        texts: List[str] = []
        for elem in filtered[:200]:
            text = (elem.get('value') or '') or (elem.get('md') or '')
            if text:
                texts.append(text)
        if not texts:
            return "摘要失败：无法从文档中提取文本内容。"
        joined = "\n\n".join(texts)[:4000]
        return f"【简化摘要原始片段】\n{joined}"

    root = build_tree(doc_items.root)
    # gen = SummaryGenerator(root=root, model_id="model_for_agent")
    gen = SummaryGenerator(root=root,model_id="model_for_doc_extract")
    asyncio.run(gen.run_batch()) # 12.15 xcm run_batch方法已改为异步方法

    summaries: List[str] = []

    def _collect(node: Node) -> None:
        if node.childs:
            for child in node.childs:
                _collect(child)
        else:
            if node.summary:
                summaries.append(node.summary)

    _collect(root)
    if not summaries:
        return "摘要生成完成，但未得到有效摘要内容。"

    combined = "\n\n".join(summaries[:10])
    return combined


@tool
def read_document_pages(start_page: int, end_page: int, config: RunnableConfig) -> str:
    """一次性读取一段连续页码的文档内容。

    适用于长文档场景：相比逐页调用 `read_document_page`，
    可以用更少的工具调用获取大段上下文，从而减少决策循环次数。

    Args:
        start_page: 起始页（从1开始）
        end_page: 结束页（包含在内，必须大于等于 start_page）
        config: 运行时配置（自动注入）

    Returns:
        str: 多页拼接后的内容摘要，按页分块输出；若无内容或参数非法则返回错误信息。
    """
    doc_content = _get_config_value(config, "doc_content", [])
    if not doc_content:
        return "文档内容为空，无法读取指定页范围。"

    if start_page < 1 or end_page < 1 or end_page < start_page:
        return "错误: 页码范围非法，请确保 start_page >= 1 且 end_page >= start_page。"

    target_pages = set(range(start_page, end_page + 1))
    page_map: Dict[int, List[Dict[str, Any]]] = {}
    for elem in doc_content:
        p = elem.get('page')
        if p in target_pages:
            page_map.setdefault(p, []).append(elem)

    if not page_map:
        return f"在页码范围 {start_page}-{end_page} 内未找到任何内容。"

    # 记录已阅读页码
    read_pages: Set[int] = _get_config_value(config, "read_pages")
    if read_pages is not None:
        read_pages.update(page_map.keys())

    # 逐页组装内容
    all_parts: List[str] = []
    for page_no in sorted(page_map.keys()):
        elems = page_map[page_no]
        content_parts: List[str] = []
        for elem in elems:
            elem_type = elem.get('type', 'text')
            text = elem.get('value') or elem.get('md', '')
            if not text:
                continue
            if elem_type == 'text' and elem.get('headingLevel'):
                content_parts.append(f"【标题{elem.get('headingLevel')}级】{text}")
            elif elem_type == 'table':
                caption = elem.get('table_caption', '')
                content_parts.append(f"【表格】{caption if caption else text[:100]}")
            elif elem_type == 'image':
                caption = elem.get('image_caption', '')
                content_parts.append(f"【图片】{caption if caption else '(图片)'}")
            elif elem_type == 'list':
                content_parts.append(f"【列表】{text[:100]}")
            else:
                content_parts.append(text)

        if not content_parts:
            continue
        page_text = "\n".join(content_parts)
        all_parts.append(
            f"第{page_no}页内容:\n" + (page_text[:800] + "..." if len(page_text) > 800 else page_text)
        )

    if not all_parts:
        return f"在页码范围 {start_page}-{end_page} 内未找到可用于展示的内容。"

    combined = "\n\n".join(all_parts)
    return combined



@tool
async def rag_tag_guideline(
    tag_id: str,
    tag_name: str,
    top_k: int = 5,
    config: RunnableConfig = None
) -> str:
    """从现有 RAG 知识库中检索与某个标签相关的定义、规则和示例。

    该工具的目标是为标签匹配提供“标签含义与命中条件”的外部知识，
    而不是直接根据文档内容做判断。

    Args:
        tag_id: 标签ID
        tag_name: 标签名称
        top_k: 返回的知识片段数量上限
        config: 运行时配置（自动注入）

    Returns:
        str: 拼接后的若干知识片段文本，供模型参考。
    """
    # 动态生成向量检索友好的查询（LLM生成陈述句/短语，而非固定问句模板）
    query_generation_prompt = f"""请为标签"{tag_name}"生成3个用于向量数据库检索的查询语句。

要求：
1. 使用陈述句或简短短语，不要使用问句
2. 包含具体、明确的关键词
3. 覆盖：标签定义、判断标准、典型特征

示例（标签：借款人身份证）：
- "借款人身份证号码 证件信息 贷款材料"
- "借款人身份证明文件 个人信息采集"  
- "借款人身份验证 证件复印件要求"

请直接输出3个查询，每行一个："""

    try:
        # 使用现有模型生成查询
        query_model = ModelFactory.get_model(key="model_for_doc_extract")
        query_response = await query_model.ainvoke([HumanMessage(content=query_generation_prompt)])
        generated_text = _remove_think_blocks(query_response.content)
        
        # 解析生成的查询（每行一个）
        generated_queries = [q.strip().strip('-').strip() for q in generated_text.split('\n') if q.strip() and not q.strip().startswith('#')]
        generated_queries = [q for q in generated_queries if len(q) > 5][:3]  # 取前3个有效查询
        
        if not generated_queries:
            # 兜底：使用简单的关键词组合
            generated_queries = [f"{tag_name} 定义 含义", f"{tag_name} 判断标准", f"{tag_name} 典型特征"]
            
        logger.info(f"为标签 {tag_name} 生成的向量检索查询: {generated_queries}")
        
    except Exception as e:
        logger.warning(f"动态生成查询失败，使用兜底方案: {e}")
        generated_queries = [f"{tag_name} 定义 含义", f"{tag_name} 判断标准", f"{tag_name} 典型特征"]
    
    questions = [
        QuestionItem(
            question_id=f"tag-{tag_id}-q{i}",
            question_text=query
        ) for i, query in enumerate(generated_queries)
    ]

    # 获取检索范围（file_ids 和 folder_id）
    file_ids = _get_config_value(config, "file_ids")
    folder_id = _get_config_value(config, "folder_id")
    
    if not file_ids and not folder_id:
         logger.warning(f"rag_tag_guideline: 未提供 file_ids 或 folder_id，将进行全库检索。tag={tag_name}")
    
    body = BatchQueryRequest(
        questions=questions,
        top_k=top_k,
        file_ids=file_ids,
        folder_id=folder_id,
        return_child_chunks=True,
        rerank_enabled=None,
        hybrid_enabled=None,
    )

    try:
        results = await run_batch_query(body)
    except Exception as e:
        logger.error(f"RAG 查询异常: {e}")
        return f"RAG 查询异常: {e}"

    if not results:
        return (
            f"未在知识库中检索到与标签 {tag_id}-{tag_name} 相关的定义或示例。"
            "请只依据文档原文进行谨慎判断。"
        )

    pieces: List[str] = []
    # 对结果进行去重和格式化（因为使用了多问题查询，可能会有重复 Chunk）
    seen_chunks = set()
    
    for idx, item in enumerate(results):
        if len(pieces) >= top_k:
            break
        
        # 使用 element_ids + file_id 作为唯一标识进行去重
        # element_ids 是 chunk 中包含的元素ID列表，结合 file_id 可以唯一标识一个 chunk
        element_ids_str = ",".join(map(str, item.element_ids)) if item.element_ids else ""
        chunk_key = f"{item.file_id}:{element_ids_str}" if element_ids_str else f"{item.file_id}:{(item.text or '')[:100]}"
        
        if chunk_key in seen_chunks:
            continue
        seen_chunks.add(chunk_key)
        
        snippet = item.text or ""
        fid = item.file_id
        score = item.score
        pieces.append(
            f"[RAG片段{len(pieces) + 1} 文件={fid} score={score}]\n{snippet}"
        )

    return "\n\n".join(pieces)


@tool
def match_tag(tag_id: str, reason: str, confidence: float = 0.8, config: RunnableConfig = None) -> str:
    """
    标记某个标签为匹配
    
    Args:
        tag_id: 标签ID
        reason: 匹配理由（必须包含具体页码引用）
        confidence: 匹配置信度 (0.0-1.0)
            - 0.9-1.0: 明确、直接的文档证据
            - 0.7-0.9: 较强的间接证据
            - 0.5-0.7: 存在证据但不充分
            - <0.5: 不建议匹配
        config: 运行时配置（自动注入）
        
    Returns:
        str: 确认信息
    """
    import re
    
    allowed_tag_id = _get_config_value(config, "tag_id", "")
    
    if allowed_tag_id and tag_id != allowed_tag_id:
        return (
            f"错误: 本轮只允许匹配当前标签 {allowed_tag_id}，"
            "请不要尝试匹配其他标签ID。"
        )

    read_pages: Set[int] = _get_config_value(config, "read_pages")
    if not read_pages:
        return (
            "错误: 不能仅根据文件名、标题或摘要打标签。请先调用 read_document_pages 或 "
            "read_document_page 阅读正文内容，并在 reason 中引用具体页码和片段，再重新尝试匹配标签。"
        )

    # 验证 reason 中必须包含页码引用
    if not re.search(r'第\d+页', reason):
        return "错误: reason 必须包含具体页码引用（如'第5页'），请重新提供包含页码的匹配理由。"
    
    # 置信度过低时给出警告
    if confidence < 0.5:
        return f"警告: 置信度 {confidence} 过低（<0.5），建议不匹配此标签。如确信应匹配，请提高置信度并重新调用。"

    result_holder = _get_config_value(config, "result_holder")
    if result_holder is not None:
        # 检查是否已匹配过（通过 tag_id 判断）
        existing_ids = [item["tag_id"] for item in result_holder if isinstance(item, dict)]
        if tag_id not in existing_ids:
            result_holder.append({
                "tag_id": tag_id,
                "reason": reason,
                "confidence": confidence
            })
            logger.info(f"匹配标签: {tag_id} (置信度: {confidence}) - {reason}")
            return f"成功匹配标签 {tag_id} (置信度: {confidence}): {reason}"
        else:
            return f"标签 {tag_id} 已经匹配过了"
            
    return f"成功匹配标签 {tag_id} (置信度: {confidence})"


@tool
def complete_matching() -> str:
    """
    标记匹配过程完成
    
    Returns:
        str: 完成确认信息
    """
    return "标签匹配已完成"


class DocTagAgent:
    """文档标签匹配Agent - 基于LangGraph"""
    
    def __init__(self):
        """初始化Agent"""
        # 初始化LLM模型
        # self.model = ModelFactory.get_model(key="model_for_agent")
        self.model = ModelFactory.get_model(key="model_for_doc_extract")
        # 定义工具
        self.tools = [
            summarize_document,
            read_document_pages,
            read_document_page,
            rag_tag_guideline,
            match_tag,
            complete_matching,
        ]
        self.model_with_tools = self.model.bind_tools(self.tools)
        self.tool_node = ToolNode(self.tools)
        
        # 创建工作流
        self.workflow = self._create_workflow()
        
        # 系统提示词 - 应用 Role Prompting + Chain-of-Thought + Step-Back Prompting
        self.system_prompt = """# 角色定义
你是一名资深的金融文档分类专家，专注于：
- 合规文档的语义理解与结构分析
- 标签定义与文档内容的精确匹配
- 基于证据的严谨判断（绝不凭猜测）

# 决策框架 (Chain-of-Thought)
请严格按以下步骤进行判断：

**Step 1 - 理解标签**
调用 `rag_tag_guideline(tag_id, tag_name)` 获取：
- 标签的正式定义
- 判别的必要条件
- 典型示例和反例

**Step 2 - 快速评估**
阅读文档摘要，初步判断匹配可能性（高/中/低）

**Step 3 - 深度验证**
使用 `read_document_pages` 阅读相关章节：
- 优先阅读目录、概述、关键章节
- 寻找与标签定义直接相关的内容

**Step 4 - 证据整理**
收集支撑证据，格式要求：
- 页码：第X页
- 原文摘录：不超过50字
- 关联说明：如何匹配标签定义

**Step 5 - 最终判断**
基于证据充分性做出决策：
- 证据充分 → 调用 `match_tag`，说明理由和置信度
- 证据不足 → 直接调用 `complete_matching`

# 可用工具
- `rag_tag_guideline(tag_id, tag_name)`: 检索标签定义、规则和典型示例
- `read_document_pages(start_page, end_page)`: 批量读取连续页码内容
- `read_document_page(page_no)`: 精读单页内容
- `match_tag(tag_id, reason, confidence)`: 匹配标签，需提供理由和置信度
- `complete_matching()`: 结束本轮判断

# 输出格式约束
在 `match_tag` 的 `reason` 参数中必须包含：
 "第X页" 具体页码引用
 原文摘录（用引号括起）
 与标签定义的对应关系

# 严格禁止
 仅凭文件名、目录或摘要匹配（必须有正文证据）
 推测性匹配（如"应该包含"、"可能有"）
 reason 中无页码引用
 跨标签匹配（只能匹配当前标签）

# 证件类型精确匹配规则
当标签涉及特定证件类型时，必须验证文档中的**证件名称**与标签**完全一致**：
- "身份证"标签 → 文档必须明确出现"居民身份证"或"身份证"字样
- "港澳通行证"标签 → 文档必须明确出现"港澳居民来往内地通行证"或"港澳通行证"
- "护照"标签 → 文档必须明确出现"护照"或"Passport"

 反例：文档标题为"港澳居民来往内地通行证"，不能匹配"身份证"标签，因为它们是不同的证件类型，即使都包含身份信息。

# 决策示例

 **正确示例**:
标签: 股东信息
reason: "第10页的'股东名册'表格中列出了3位自然人股东（张三持股30%、李四持股25%、王五持股20%），符合'股东信息'标签要求披露的持股比例信息。"
confidence: 0.95

 **错误示例**:
reason: "这是一份公司章程，通常会包含股东信息。"
(错误原因: 无具体页码，无原文摘录，仅为推测)"""

    def _create_workflow(self) -> StateGraph:
        """创建工作流图"""
        workflow = StateGraph(TagAgentState)
        
        # 添加节点
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self.tool_node)
        
        # 添加边
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "tools": "tools",
                "end": END
            }
        )
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    def _agent_node(self, state: TagAgentState) -> TagAgentState:
        """Agent决策节点"""
        messages = state["messages"]
        response = self.model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def _should_continue(self, state: TagAgentState) -> str:
        """判断是否继续执行工具"""
        last_message = state["messages"][-1]
        
        # 如果有工具调用，继续执行工具
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            # 检查是否有 complete_matching 调用
            for tool_call in last_message.tool_calls:
                if tool_call['name'] == 'complete_matching':
                    return "end"
            return "tools"
        
        # 否则结束
        return "end"

    def _summarize_doc_content(self, doc_content: List[Dict[str, Any]]) -> str:
        """对整个文档内容进行一次性摘要，供所有标签共享使用。"""
        if not doc_content:
            return "文档内容为空，无法生成摘要。"

        try:
            doc_items = DocumentItems.model_validate(doc_content)
        except Exception as e:
            logger.error(f"DocTagAgent 全文摘要解析失败: {e}")
            texts: List[str] = []
            for elem in doc_content[:200]:
                text = (elem.get("value") or "") or (elem.get("md") or "")
                if text:
                    texts.append(text)
            if not texts:
                return "摘要失败：无法从文档中提取文本内容。"
            joined = "\n\n".join(texts)[:4000]
            return f"【简化摘要原始片段】\n{joined}"

        root = build_tree(doc_items.root)
        # gen = SummaryGenerator(root=root, model_id="model_for_agent")
        gen = SummaryGenerator(root=root, model_id="model_for_doc_extract")
        asyncio.run(gen.run_batch()) # 12.15 xcm run_batch方法已改为异步方法

        summaries: List[str] = []

        def _collect(node: Node) -> None:
            if node.childs:
                for child in node.childs:
                    _collect(child)
            else:
                if node.summary:
                    summaries.append(node.summary)

        _collect(root)
        if not summaries:
            return "摘要生成完成，但未得到有效摘要内容。"

        combined = "\n\n".join(summaries[:10])
        return combined

    async def match_tags(
        self,
        tags: List[Dict[str, str]],
        doc_content: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """ 
        执行标签匹配（先全文摘要，再按标签并行流转）。
        """
        try:
            # 从文档内容中提取 file_id (通常取第一个非空的 file_id)
            # 假设 doc_content 是 minerU 格式列表，每个元素可能包含 file_id
            file_ids = set()
            folder_ids = set()
            for item in doc_content:
                if fid := item.get("file_id"):
                    file_ids.add(fid)
                if fold := item.get("folder_id"):
                    folder_ids.add(fold)
            
            # 如果有多个 file_id，这里暂且只取一个，或者全部作为检索范围
            # 为了兼容性，转为 list
            target_file_ids = list(file_ids) if file_ids else None
            target_folder_id = list(folder_ids)[0] if folder_ids else None
            
            logger.info(f"从文档内容提取到 file_ids: {target_file_ids}, folder_id: {target_folder_id}")

            # 计算文档整体统计信息
            pages = {d.get("page") for d in doc_content if d.get("page") is not None}
            heading_cnt = 0
            table_cnt = 0
            for d in doc_content:
                try:
                    if d.get("headingLevel") not in (None, "", 0, "0"):
                        heading_cnt += 1
                except Exception:
                    pass
                if str(d.get("type") or "").lower() == "table":
                    table_cnt += 1
            doc_stats = {
                "total_items": len(doc_content),
                "total_pages": len(pages),
                "total_headings": heading_cnt,
                "total_tables": table_cnt,
            }
            doc_summary_text = (
                f"文档共{doc_stats['total_items']}个片段，{doc_stats['total_pages']}页，"
                f"其中标题约{doc_stats['total_headings']}个，表格约{doc_stats['total_tables']}个。"
            )

            # 全文摘要（一次性，供所有标签共享）
            # 注意：summarize_document 是同步的，这里直接调用 _summarize_doc_content
            # 如果需要异步化摘要，需要改造 summary_generator
            global_summary = await asyncio.to_thread(self._summarize_doc_content, doc_content)

            # 准备并发任务
            tasks = []
            
            # 用于收集结果的容器列表，每个任务对应一个 result_holder 列表
            # 由于是并发的，每个任务使用自己独立的 list 来收集结果
            task_results = [] 

            for tag in tags:
                tag_id = tag.get("tagId") or tag.get("tag_id")
                tag_name = tag.get("tagName") or tag.get("tag_name") or ""
                if not tag_id:
                    continue

                user_message = f"""# 当前任务
判断以下标签是否与文档匹配：

| 属性 | 值 |
|-----|-----|
| 标签ID | `{tag_id}` |
| 标签名称 | {tag_name} |

# 文档概况
{doc_summary_text}

# 全文摘要
{global_summary}

# Step-Back 思考（请先回答以下问题再行动）
1. 这个标签的核心定义是什么？（调用 rag_tag_guideline 获取）
2. 什么样的文档内容能证明满足该标签？
3. 文档摘要中是否有相关线索？

# 执行步骤
1. 调用 `rag_tag_guideline("{tag_id}", "{tag_name}")` 获取标签定义
2. 根据标签要求，使用 `read_document_pages` 阅读关键内容
3. 收集充分证据后调用 `match_tag(tag_id, reason, confidence)`
4. 最后调用 `complete_matching` 结束本标签判断"""

                initial_state = {
                    "messages": [
                        SystemMessage(content=self.system_prompt),
                        HumanMessage(content=user_message),
                    ],
                    "tag": tag,
                    "matched_tags": [],
                    "is_complete": False,
                }
                
                # 为每个任务创建独立的配置和结果容器
                # result_holder 现在存储 {"tag_id": str, "reason": str, "confidence": float} 的字典列表
                result_holder: List[Dict[str, Any]] = []
                task_results.append(result_holder)
                
                config = {
                    "recursion_limit": 80,
                    "configurable": {
                        "doc_content": doc_content,
                        "tag_id": tag_id,
                        "read_pages": set(), # 每个标签独立的阅读记录
                        "result_holder": result_holder,
                        "file_ids": target_file_ids,  # 传递给工具使用
                        "folder_id": target_folder_id
                    }
                }
                
                tasks.append(self.workflow.ainvoke(initial_state, config=config))

            # 并发执行所有标签的判断
            if tasks:
                # 使用 return_exceptions=True 防止单个任务失败影响其他任务
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 检查是否有异常发生
                for i, res in enumerate(results):
                    if isinstance(res, Exception):
                        tag_info = tags[i] if i < len(tags) else "unknown"
                        logger.error(f"标签 {tag_info} 处理任务异常: {res}")
            
            # 收集结果
            matched_tag_ids = []
            explanations = []  # 收集每个标签的匹配说明
            
            for res_list in task_results:
                for item in res_list:
                    if isinstance(item, dict):
                        matched_tag_ids.append(item["tag_id"])
                        explanations.append(f"[{item['tag_id']}] (置信度:{item['confidence']}) {item['reason']}")
                    else:
                        # 兼容旧格式
                        matched_tag_ids.append(item)
            
            # 去重（理论上不需要，但为了保险）
            matched_tag_ids = list(dict.fromkeys(matched_tag_ids))  # 保持顺序的去重

            logger.info(f"匹配完成，共匹配 {len(matched_tag_ids)} 个标签: {matched_tag_ids}")

            return {
                "matched_tags": matched_tag_ids,
                "explanations": explanations,  # 新增：返回匹配说明列表
                "success": True,
            }

        except Exception as e:
            logger.error(f"标签匹配失败: {e}")
            return {
                "matched_tags": [],
                "success": False,
                "error": str(e),
            }
