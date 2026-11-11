from __future__ import annotations

from typing import Any, List, Optional, Dict, Callable, Awaitable
from ..input_models import Item
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from ..models import QuestionTask

class SelectedPiece(BaseModel):
    """单个被选中的片段。结构尽量贴合检索器的实际输出。"""
    doc_id: str = Field(...)
    element_ids: List[int] = Field(default_factory=list, description="Origin element_id list for this piece")
    file_id: str = Field(..., description="Origin file_id")
    relevance_score: Optional[int] = None
    selection_reason: Optional[str] = None
    # 标记片段来源阶段：chunk/summary/document，以便构建thinking_steps
    source_stage: Optional[str] = Field(default=None, description="Which retrieval stage produced this piece")
    # 直接承载的原始文本内容（用于且仅用于非索引类 data_sources 生成的片段）
    raw_text: Optional[str] = Field(
        default=None,
        description="原始文本内容。**仅用于**不存在 file_id/element_ids 映射的 data_sources 生成的片段。**否则**应保持为空。",
    )
    # 仅当为 True 时表示该片段来自 GenericDataSource，无真实文档来源，citation 解析时按“数据源”处理（不填 element_ids/file_id）
    raw_text_only: bool = Field(
        default=False,
        description="是否为纯数据源片段（无 element_ids/file_id）。True 时 citation 解析不还原为文档位置。",
    )


class SelectedContent(BaseModel):
    """每个问题对应的检索结果集合。"""
    question: str = Field(...)
    question_id: str = Field(..., description="问题的唯一标识符")
    selected_pieces: List[SelectedPiece] = Field(default_factory=list)


class BlockPieceSelection(BaseModel):
    """单个文档片段的选择结果"""
    piece_index: int = Field(..., description="文档片段索引（从1开始）")
    relevance_score: int = Field(..., ge=1, le=10, description="相关性评分（1-10分）")
    selection_reason: Optional[str] = Field(None, description="选择理由")
    # key_info: Optional[str] = Field(None, description="关键信息摘要")

class BlockSelectionResponse(BaseModel):
    """LLM返回的文档片段选择结果"""
    selected_pieces: List[BlockPieceSelection] = Field(..., description="选中的文档片段列表")


class BlockPieceSelectionSimple(BaseModel):
    """单个文档片段的选择结果（简单版，不包含选择理由）"""
    piece_index: int = Field(..., description="文档片段索引（从1开始）")
    relevance_score: int = Field(..., ge=1, le=10, description="相关性评分（1-10分）")

class BlockSelectionResponseSimple(BaseModel):
    """LLM返回的文档片段选择结果（简单版，不包含选择理由）"""
    selected_pieces: List[BlockPieceSelectionSimple] = Field(..., description="选中的文档片段列表")


class Source(BaseModel):
    """来源信息模型"""
    element_id: Optional[int] = Field(None, description="元素ID")
    file_id: Optional[str] = Field(None, description="文件ID")


class Piece(BaseModel):
    """文档片段模型"""
    piece_index: int = Field(..., description="片段索引")
    token_count: int = Field(..., description="token数量")
    text: str = Field(..., description="片段文本内容")
    texts: List[str] = Field(default_factory=list, description="原始文本列表")
    sources: List[Source] = Field(default_factory=list, description="来源信息列表")


class Block(BaseModel):
    """基础块模型"""
    block_index: int = Field(..., description="块索引")
    token_count: int = Field(..., description="token数量")
    text_full: List[str] = Field(default_factory=list, description="完整文本列表")
    sources_full: List[Source] = Field(default_factory=list, description="完整来源信息列表，与text_full一一对应")


class BlockWithPieces(BaseModel):
    """包含pieces的块模型"""
    block_index: int = Field(..., description="块索引")
    token_count: int = Field(..., description="token数量")
    pieces: List[Piece] = Field(default_factory=list, description="文档片段列表")
    pieces_count: int = Field(0, description="片段数量")


class RetrieverState(MessagesState):
    """检索工作流的状态结构（占位版本）。

    与现有 content_generate_agent 的风格一致：
    - 使用 MessagesState 以兼容消息累积
    - 提供基础输入与流程中间变量占位
    """

    # 输入
    question_tasks: List[QuestionTask]  # QuestionTask实例列表
    unfinished_question_tasks: List[QuestionTask]  # 上下文还不足够的问题任务列表
    origin_content: List[Item]
    model_id: str  # 用于batch操作的模型（summary_index_retrival, document_reading_retrival等）
    model_id_for_judge: str  # 用于judge_if_continue节点的模型
    answer_lang: str  # 回答语言 locale（如 zh_CN / en_US），用于选择中间 prompt 模板

    # 中间产物
    selected_content: List[SelectedContent]  # 每一个元素为对每一个问题选择出的片段
    # selected_documents: List[Any] = []

    # 控制参数
    document_reading_retrival_done: bool
    summary_index_retrival_done: bool
    chunk_index_retrival_done: bool
    context_token_limit: int  # 上下文token限制
    # max_iterations: int = 3
    
    # 新增：生产者-消费者架构相关字段
    finished_question_ids: List[str]  # 已完成检索的问题ID列表（生产者产出）
    finished_question_answers: Dict[str, str]  # 已完成问题的答案缓存（消费者产出）
    use_batch_for_llm: bool  # 是否在 LLM 调用节点启用批量调用
    
    # 新增：记录每个问题的思考步骤信息（检索query、资源、分析等）
    thinking_steps_map: Dict[str, Dict[str, Any]]
    summary_tree_provider: Optional[Callable[[List[Item]], Awaitable[Any]]]

    # 性能统计：各节点执行耗时（秒）
    node_timings: Dict[str, float]


def create_retriever_state_with_defaults(state: dict) -> RetrieverState:
    """创建RetrieverState实例，自动补全默认值"""
    defaults = {
        "question_tasks": [],
        "unfinished_question_tasks": [],
        "origin_content": [],
        "model_id": "model_for_content_generation_backup",
        "model_id_for_judge": "model_for_content_generation_backup",  # 默认与model_id相同
        "answer_lang": "zh_CN",
        "selected_content": [],
        "document_reading_retrival_done": False,
        "summary_index_retrival_done": False,
        "chunk_index_retrival_done": False,
        "context_token_limit": 25000,  # 默认25k token限制
        "finished_question_ids": [],
        "finished_question_answers": {},
        "use_batch_for_llm": False,
        "thinking_steps_map": {},  # 用于填充 FinalAnswer.thinking_steps
        "summary_tree_provider": None,
        "messages": [],  # MessagesState的默认字段
        "node_timings": {},  # 用于存储各节点执行耗时
    }
    
    # 合并默认值和传入的state
    full_state = {**defaults, **state}
    
    # 特殊处理：如果unfinished_question_tasks为空，则初始化为question_tasks的副本
    if not full_state.get("unfinished_question_tasks") and full_state.get("question_tasks"):
        full_state["unfinished_question_tasks"] = full_state["question_tasks"].copy()
    
    return RetrieverState(**full_state)
