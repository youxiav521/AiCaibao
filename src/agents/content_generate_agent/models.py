"""内容生成相关的数据模型定义"""
from typing import List, Dict, Any, Optional, Literal, Union

from loguru import logger
from pydantic import BaseModel, Field, ConfigDict, model_validator
from pydantic.alias_generators import to_camel

from .normalization import normalize_noneish


class Document(BaseModel):
    """文档模型"""
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)
    
    file_name: str = Field(description="文件名")
    content: List[Dict[str, Any]] = Field(description="内容条目数组")


class ChunkIndexRetrievalTool(BaseModel):
    """chunk索引检索工具配置"""
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)
    
    type: Literal["chunk_index_retrieval"] = Field("chunk_index_retrieval", description="工具类型")
    query: Optional[str] = Field(
        None,
        description=(
            "检索查询文本；若为空，则在实际检索时会通过大模型"
            "综合问题 description 与 prompt 自动生成合适的默认 query"
        ),
    )
    file_ids: Optional[List[str]] = Field(
        default=None,
        description=(
            "可选：限制RAG检索的文件范围（对应RAG请求中的 file_ids）。"
            "为空时保持默认行为：从 contentList/origin_content 中提取所有唯一 file_id 作为检索范围。"
        ),
    )
    top_k: int = Field(30, description="返回top_k个结果")
    return_child_chunk: bool = Field(True, description="是否返回子chunk")
    rerank_enabled: bool = Field(False, description="是否启用重排序")
    hybrid_enabled: bool = Field(True, description="是否启用混合检索")
    search_table: bool = Field(False, description="是否仅检索表格")


class CodingAgentTool(BaseModel):
    """编码代理工具配置"""
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)
    
    type: Literal["coding_agent"] = Field("coding_agent", description="工具类型")
    enable: bool = Field(True, description="是否启用 CodingAgent 流程（planner->coder->reporter）。为 False 时仅走 reporter 流程。")


class GenericDataSource(BaseModel):
    """通用数据源配置（非 chunk_index_retrieval 类型）。
    
    用于承载诸如 financial_report / extract 等直接作为上下文片段使用的原始数据。
    """
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    type: str = Field(description="数据源类型（如 financial_report / extract 等）")
    data: Any = Field(description="该数据源对应的原始内容，将直接作为上下文片段使用")


class OtherDataSource(BaseModel):
    """自定义字符串数据源。

    schema：
    {
      "type": "other_datasource",
      "data": "<任意字符串，作为真正内容>"
    }
    """

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    type: Literal["other_datasource"] = Field("other_datasource", description="固定数据源类型：other_datasource")
    data: str = Field(..., description="任意字符串，作为真正内容（将被直接注入上下文）")


class QuestionTask(BaseModel):
    """问题任务模型"""
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)
    
    id: str = Field(description="问题ID")
    description: str = Field(description="问题文本")
    type: str = Field(description="问题类型：text/table/text-json")
    prompt: Optional[str] = Field(default="", description="可选的提示信息")
    html: Optional[str] = Field(default="", description="可选的HTML内容")
    # tools 仅用于控制 AnswerAgent 的行为（例如是否启用 CodingAgent），
    # 不再承载检索策略；检索相关配置迁移至 data_sources 字段。
    tools: Optional[List[CodingAgentTool]] = Field(
        default=None,
        description="工具配置列表（当前仅支持 CodingAgentTool，用于控制是否启用 CodingAgent 流程）",
    )
    # 新增：按数据源配置检索策略（优先级高于基于 tools 的检索配置）
    data_sources: Optional[List[Union[ChunkIndexRetrievalTool, OtherDataSource, GenericDataSource]]] = Field(
        default=None,
        description="数据源配置列表。chunk_index_retrieval 类型走 RAG，其它类型的 data 将直接作为 SelectedPiece.raw_text 使用"
    )
    # 新增：每个问题可指定额外需要加载的文件ID列表（knowledge）
    knowledge: Optional[List[str]] = Field(
        default=None,
        description="该问题额外需要加载的文件ID列表（knowledge），用于扩展可检索的内容范围",
    )
    # 新增：整个检索阶段统一使用的检索 query（如果为空则在检索相关节点回退为 description）
    retrieval_query: Optional[str] = Field(
        default=None,
        description="整个检索阶段统一使用的检索query；为空时默认回退为 description", # TODO：验证，似乎现在应该不是回退为description而是会生成？
    )
    # 新增：问题依赖关系（DAG）
    base_on_context: Optional[List[str]] = Field(
        default=None,
        description="当前问题依赖的问题ID列表（如 ['q1','q2']）；为空/None 视为无依赖",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_input_noneish(cls, data: Any) -> Any:
        """
        何时触发：
        - 当 Pydantic 要解析/校验 `QuestionTask` 时，在字段解析与类型转换之前触发。
        - 常见入口：FastAPI 请求体解析（questionTasks）、`QuestionTask.model_validate()`。

        做什么：
        - 对整个输入 dict 做一次递归“空值归一化”，把 Java 侧的：
          - 空白字符串（""/"   "）
          - 空数组 []
          - 空对象 {}
          统一转成 None，并剔除归一化为 None 的子项。
        - 目标是让后续业务逻辑只需要处理 `None` 这一种“无值”表达，避免 None/""/[]/{} 混用。
        """
        if not isinstance(data, dict):
            return data

        qid = data.get("id")
        base_path = f"QuestionTask[{qid}]" if isinstance(qid, str) and qid.strip() else "QuestionTask"

        # 递归：把所有 \"\"/空白字符串/[]/{} 统一视为 None，并剔除 None 元素/键
        normalized = normalize_noneish(data, path=base_path)
        # normalize_noneish 可能把整个 dict 变成 None（例如全为空值）；这里保持原样交给 pydantic 报错更直观
        return normalized if normalized is not None else data

    @model_validator(mode="after")
    def _normalize_base_on_context(self) -> "QuestionTask":
        """
        进一步清洗 base_on_context：
        - None 视为无依赖
        - 过滤空白字符串
        - 去重但保持稳定顺序
        - 清洗后为空则置为 None（与 normalize_noneish 的语义保持一致）
        """
        if not self.base_on_context:
            self.base_on_context = None
            return self

        seen = set()
        cleaned: List[str] = []
        for dep in self.base_on_context:
            if not isinstance(dep, str):
                logger.warning("base_on_context 中的依赖项 {} 不是字符串，跳过", dep)
                continue
            dep2 = dep.strip()
            if not dep2:
                logger.warning("base_on_context 中的依赖项 {} 为空字符串，跳过", dep)
                continue
            if dep2 in seen:
                logger.warning("base_on_context 中的依赖项 {} 已存在，跳过", dep)
                continue
            seen.add(dep2)
            cleaned.append(dep2)

        self.base_on_context = cleaned or None
        return self

    def get_resolved_tools(self) -> Optional[List[CodingAgentTool]]:
        """获取处理后的 tools 列表。

        当前 tools 仅支持 CodingAgentTool，且不再承载检索相关配置，
        因此这里只做简单返回（为未来扩展预留钩子）。
        """
        if self.tools is None:
            return None
        # 直接返回（如需后续扩展，可在此处做默认值填充等处理）
        return list(self.tools)

    def get_resolved_data_sources(self) -> Optional[List[Union[ChunkIndexRetrievalTool, OtherDataSource, GenericDataSource]]]:
        """获取处理后的 data_sources 列表。
        
        - 对于 chunk_index_retrieval 类型，保留原配置（不在此处生成默认 query），
          实际检索时若 query 为空，会通过大模型基于问题 description 与 prompt 生成默认 query。
        - 其它类型（GenericDataSource）原样返回。
        """
        if self.data_sources is None:
            return None

        resolved: List[Union[ChunkIndexRetrievalTool, OtherDataSource, GenericDataSource]] = []
        for ds in self.data_sources:
            if isinstance(ds, ChunkIndexRetrievalTool):
                # 不在此处填充默认 query，保留原始配置，便于在检索阶段按需调用大模型生成默认 query
                resolved.append(ds)
            else:
                resolved.append(ds)
        return resolved

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，用于向后兼容内部方法"""
        result = {
            "id": self.id,
            "text": self.description,  # 兼容旧字段名
            "description": self.description,  # 兼容内部方法使用的字段名
            "type": self.type,
            "prompt": self.prompt or "",
            "html": self.html or "",
        }
        # 如果tools存在，添加到字典中
        if self.tools is not None:
            result["tools"] = [tool.model_dump() for tool in self.tools]
        # 如果data_sources存在，也一并输出，便于调试
        if self.data_sources is not None:
            result["data_sources"] = [ds.model_dump() for ds in self.data_sources]
        # 如果knowledge存在，也输出，便于调试
        if self.knowledge is not None:
            result["knowledge"] = list(self.knowledge)
        # 如果retrieval_query存在，也输出，便于调试
        if self.retrieval_query is not None:
            result["retrieval_query"] = self.retrieval_query
        # 如果base_on_context存在，也输出，便于调试
        if self.base_on_context is not None:
            result["base_on_context"] = list(self.base_on_context)
        return result


class ContentGenerateResponse(BaseModel):
    """内容生成响应模型"""
    # 注：alias_generator=to_camel自动为所有字段生成小驼峰命名的别名
    # populate_by_name=True允许在反序列化（接收数据）时，同时接受原始字段名和别名。
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    success: bool = Field(description="生成是否成功")
    message: str = Field(description="生成结果消息")
    completed_questions_counts: int = Field(description="完成回答了的总问题数量")
    # question_answers: List[Dict[str, Any]] = Field(description="所有问题的回答结果")


class Citation(BaseModel):
    """引用模型"""
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)
    
    id: int = Field(..., description="Citation ID aligned with appearance order in answer")
    element_ids: List[int] = Field(default_factory=list, description="List of source element_ids that support this citation")
    file_id: str = Field("", description="The file_id of the source document for these elements")
    quote: str = Field(..., description="The VERBATIM quote from the specified source that justifies the answer.")


class RetrievalResourceItem(BaseModel):
    """检索资源项（用于resources items中）"""
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)
    
    page: str = Field(..., description="页码（字符串格式）")
    file_id: str = Field(..., description="文件ID")


class RetrievalQueryItem(BaseModel):
    """检索查询项"""
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)
    
    type: Literal["query"] = Field("query", description="类型：query")
    text: str = Field(..., description="查询文本")


class RetrievalResourcesItem(BaseModel):
    """检索资源项（包含多个资源）"""
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)
    
    type: Literal["resources"] = Field("resources", description="类型：resources")
    items: List[RetrievalResourceItem] = Field(default_factory=list, description="资源列表")


class RetrievalAnalysisItem(BaseModel):
    """检索分析项"""
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)
    
    type: Literal["analysis"] = Field("analysis", description="类型：analysis")
    text: str = Field(..., description="分析文本")


RetrievalItem = Union[RetrievalQueryItem, RetrievalResourcesItem, RetrievalAnalysisItem]


class ChunkRetrievalInfo(BaseModel):
    """Chunk检索信息"""
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)
    
    type: Literal["chunkRetrieval"] = Field("chunkRetrieval", description="检索类型")
    items: List[RetrievalItem] = Field(default_factory=list, description="检索项列表（query固定第一个，然后是resources，最后是analysis（如果有））")


class SummaryRetrievalInfo(BaseModel):
    """Summary检索信息"""
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)
    
    type: Literal["summaryRetrieval"] = Field("summaryRetrieval", description="检索类型")
    items: List[RetrievalItem] = Field(default_factory=list, description="检索项列表（query固定第一个，然后是resources，最后是analysis（如果有））")


class DocumentRetrievalInfo(BaseModel):
    """Document检索信息"""
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)
    
    type: Literal["documentRetrieval"] = Field("documentRetrieval", description="检索类型")
    items: List[RetrievalItem] = Field(default_factory=list, description="检索项列表（query固定第一个，然后是resources，最后是analysis（如果有））")

class AnswerAgentStepItem(BaseModel):
    """Answer Agent步骤项"""
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)
    
    type: Literal["planner", "coder", "reporter"] = Field(..., description="步骤类型：planner/coder/reporter")
    text: Optional[str] = Field(default=None, description="步骤文本内容（planner和reporter使用）")
    items: Optional[List[Dict[str, Any]]] = Field(default=None, description="coder 步骤中的子步骤列表（直接使用trace_steps的原始数据）")

class ThinkingSteps(BaseModel):
    """思考步骤模型"""
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)
    
    chunk_retrieval: Optional[ChunkRetrievalInfo] = Field(default=None, description="Chunk检索信息")
    summary_retrieval: Optional[SummaryRetrievalInfo] = Field(default=None, description="Summary检索信息")
    document_retrieval: Optional[DocumentRetrievalInfo] = Field(default=None, description="Document检索信息")
    answer_agent_steps: Optional[List[AnswerAgentStepItem]] = Field(default=None, description="Answer Agent 步骤列表")


class  FinalAnswer(BaseModel):
    """Answer with resolved citations back to element_ids/file_id."""
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)
    
    is_answerable: bool = Field(..., description="Whether the answer is answerable from the given sources.")
    answer_text: str = Field(...)
    citations: List[Citation] = Field(default_factory=list)
    thinking_steps: ThinkingSteps = Field(default_factory=ThinkingSteps, description="思考步骤")


class ContentGenerateResponseEvent(BaseModel):
    """内容生成响应事件 - Agent输出模型"""
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    question_id: str = Field(description="问题id")
    question_text: str = Field(description="要回答的问题")
    question_type: str = Field(description="问题类型")
    answer: FinalAnswer = Field(description="答案")

