from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SearchType(StrEnum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"

    @classmethod
    def field_description(cls) -> str:
        options = "、".join(f"'{member.value}'" for member in cls)
        return f"按 type 过滤，可选值: {options}，为空时不过滤"

    @classmethod
    def from_value(cls, value: str | "SearchType" | None) -> "SearchType" | None:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        if not normalized:
            return None
        return cls(normalized)

    def to_metadata_type_filter(self) -> str | list[str]:
        if self is SearchType.TABLE:
            return ["table", "excel_table", "table_row"]
        return self.value


class DeleteRequestHttp(BaseModel):
    """删除索引的请求体（HTTP 请求用）。"""
    fileId: str


class DeleteResponse(BaseModel):
    """删除索引的响应体。"""
    file_id: str
    deleted: bool


class UploadIndexResponse(BaseModel):
    """上传并入库的响应体。"""
    success: bool


class UploadIndexRequestHttp(BaseModel):
    """索引入库请求体（HTTP 请求用）。"""
    model_config = ConfigDict(extra="forbid")

    docParseContent: List[Dict[str, Any]] = Field(..., description="converted_output 生成的 items 对象数组")
    fileName: str = Field(..., min_length=1, description="原始文件名")
    docFolderId: Optional[str] = Field(default=None, description="文档文件夹 ID")

    @field_validator("docParseContent", mode="before")
    @classmethod
    def _validate_doc_parse_content(cls, value: Any) -> Any:
        if not isinstance(value, list):
            raise ValueError("docParseContent 必须是 items 对象数组")
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise ValueError(f"docParseContent[{index}] 必须是对象")
        return value


class ChunkFields(BaseModel):
    """所有切片类型的公共存储字段基类。"""

    model_config = ConfigDict(extra="forbid")

    # 身份标识
    parent_id: str = ""
    child_id: str = ""
    grandchild_id: str = ""

    # 文本内容
    text: str = ""

    # 来源信息
    file_id: str = ""
    folder_id: str = ""
    file_name: str = ""

    # 内容分类
    type: str = ""
    doc_kind: str = ""

    # 前端输出
    source_element_ids: List[Any] = Field(default_factory=list)
    heading_node_ids: List[Any] = Field(default_factory=list)
    element_id: Any = None
    image_storage_path: str = ""
    table_caption_context_element_ids: List[Any] = Field(default_factory=list)

    # PCR 去重键
    split_strategy: str = ""
    table_row_start: Any = None
    table_row_end: Any = None
    heading_path: str = ""


class ParentChunk(ChunkFields):
    parent_id: str
    text: str
    summary: str = ""


class ChildChunk(ChunkFields):
    child_id: str
    parent_id: str
    text: str


class GrandchildChunk(ChunkFields):
    grandchild_id: str
    child_id: str
    parent_id: str
    text: str


class QuestionItem(BaseModel):
    """批量查询中的单个问题项。"""
    question_id: str
    question_text: str


class QuestionItemHttp(BaseModel):
    """批量查询中的单个问题项（驼峰命名，供 HTTP 请求使用）。"""
    questionId: str
    questionText: str


class BatchQueryRequest(BaseModel):
    """批量查询请求体。"""
    questions: List[QuestionItem]
    top_k: int = Field(default=8, ge=1)
    file_ids: Optional[List[str]] = Field(default=None, description="按 file_id 过滤")
    folder_id: Optional[str] = Field(default=None, description="按 folder_id 过滤（资源库 ID）")
    return_child_chunks: bool = Field(default=True, description="是否检索孙切片但返回子切片")
    rerank_enabled: Optional[bool] = Field(default=False, description="是否启用重排，None 时使用配置默认值")
    hybrid_enabled: Optional[bool] = Field(default=None, description="是否启用混合检索，None 时使用配置默认值")
    search_type: Optional[SearchType] = Field(
        default=None,
        description=SearchType.field_description(),
    )
    summary_mode: Literal["off", "on", "only"] = Field(
        default="off",
        description="summary index 检索模式：off=仅 chunk，on=chunk+summary，only=仅 summary",
    )


class BatchQueryRequestHttp(BaseModel):
    """批量查询请求体。"""
    questions: List[QuestionItemHttp]
    topK: int = Field(default=8, ge=1)
    fileIds: Optional[List[str]] = Field(default=None, description="按 file_id 过滤")
    folderId: Optional[str] = Field(default=None, description="按 folder_id 过滤（资源库 ID）")
    returnChildChunks: bool = Field(default=True, description="是否检索孙切片但返回子切片")
    rerankEnabled: Optional[bool] = Field(default=False, description="是否启用重排，None 时使用配置默认值")
    hybridEnabled: Optional[bool] = Field(default=None, description="是否启用混合检索，None 时使用配置默认值")
    searchType: Optional[SearchType] = Field(
        default=None,
        description=SearchType.field_description(),
    )
    summaryMode: Literal["off", "on", "only"] = Field(
        default="on",
        description="summary index 检索模式：off=仅 chunk，on=chunk+summary，only=仅 summary",
    )


class BatchQueryItem(BaseModel):
    """批量查询的单条结果项。"""
    question_id: str
    question_text: Optional[str] = None
    element_ids: List[int]
    file_id: str
    file_name: Optional[str] = None
    score: Optional[int] = None
    text: Optional[str] = None
    image_storage_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchQueryItemHttp(BaseModel):
    """批量查询的单条结果项（驼峰命名，供 HTTP 响应使用）。"""
    questionId: str
    questionText: Optional[str] = None
    elementIds: List[int]
    fileId: str
    fileName: Optional[str] = None
    score: Optional[int] = None
    text: Optional[str] = None
    imageStoragePath: Optional[str] = None
    tableCaptionContextElementIds: List[int] = Field(default_factory=list)


def batch_query_item_to_http(item: BatchQueryItem) -> BatchQueryItemHttp:
    """将内部 BatchQueryItem 转换为 HTTP 响应用的 BatchQueryItemHttp。"""
    raw_context_ids = item.metadata.get("table_caption_context_element_ids", [])
    if not isinstance(raw_context_ids, list):
        raw_context_ids = [raw_context_ids] if raw_context_ids is not None else []

    context_ids: List[int] = []
    seen: set[int] = set()
    for raw_id in raw_context_ids:
        if raw_id is None or isinstance(raw_id, bool):
            continue
        candidate: int | None = None
        if isinstance(raw_id, int):
            candidate = raw_id
        elif isinstance(raw_id, float) and raw_id.is_integer():
            candidate = int(raw_id)
        else:
            raw_text = str(raw_id).strip()
            if raw_text.isdigit():
                candidate = int(raw_text)
        if candidate is None or candidate in seen:
            continue
        seen.add(candidate)
        context_ids.append(candidate)

    return BatchQueryItemHttp(
        questionId=item.question_id,
        questionText=item.question_text,
        elementIds=item.element_ids,
        fileId=item.file_id,
        fileName=item.file_name,
        score=item.score,
        text=item.text,
        imageStoragePath=item.image_storage_path or item.metadata.get("image_storage_path"),
        tableCaptionContextElementIds=context_ids,
    )


def batchQueryRequestHttp_to_batchQueryRequest(item: BatchQueryRequestHttp) -> BatchQueryRequest:
    """将 HTTP 请求体 BatchQueryRequestHttp 转换为内部 BatchQueryRequest。"""
    return BatchQueryRequest(
        questions=[QuestionItem(question_id=q.questionId, question_text=q.questionText) for q in item.questions],
        top_k=item.topK,
        file_ids=item.fileIds,
        folder_id=item.folderId,
        return_child_chunks=item.returnChildChunks,
        rerank_enabled=item.rerankEnabled,
        hybrid_enabled=item.hybridEnabled,
        search_type=item.searchType,
        summary_mode=item.summaryMode,
    )


class StructureBatchRecallRequest(BaseModel):
    """文档结构批量召回请求体。

    将多个问题对同一批文件的结构召回合并为一次大模型调用，
    使文档结构的 KV cache 在多个问题间共享。
    """
    questions: List[QuestionItemHttp] = Field(..., min_length=1, description="召回问题列表")
    fileIds: List[str] = Field(..., min_length=1, description="文件ID列表")
    maxTokens: int = Field(default=-1, description="每个问题最大返回token数，-1=不限制")
    maxStructureNum: int = Field(default=-1, description="每个问题最多返回结构数，-1=不限制")


class FolderFileItem(BaseModel):
    """文件夹下单个文件的信息。"""
    fileId: str = Field(description="文件 ID")
    fileName: Optional[str] = Field(default=None, description="文件名")


class FolderFileCountResponse(BaseModel):
    """根据 folder_id 查询文件数量和文件信息的响应体。"""
    folderId: str = Field(description="文件夹 ID")
    fileCount: int = Field(description="文件数量")
    fileIds: List[str] = Field(default_factory=list, description="文件 ID 列表")
    files: List[FolderFileItem] = Field(default_factory=list, description="文件信息列表")


class RagBackfillRequestHttp(BaseModel):
    """旧 RAG 集合回填到新存储的异步任务请求体。"""
    oldParentCollection: str = Field(..., min_length=1, description="旧父切片 ES 集合名称")
    oldChildCollection: Optional[str] = Field(default=None, min_length=1, description="旧子切片 ES 集合名称")
    oldGrandchildCollection: Optional[str] = Field(default=None, min_length=1, description="旧孙切片 ES 集合名称")


class RagBackfillSubmitResponse(BaseModel):
    """RAG 回填任务提交结果。"""
    jobId: str
    taskType: str


class RagBackfillJobResponse(BaseModel):
    """RAG 回填任务状态。"""
    jobId: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
    attempts: int = 0
