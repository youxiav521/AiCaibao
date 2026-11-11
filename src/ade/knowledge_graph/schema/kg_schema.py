import hashlib

from pydantic import BaseModel, Field, model_validator
from typing import Optional, Any, Literal


def entity_id(folder_id: str, name: str, type_: str) -> str:
    """由知识库、名称、类型生成实体ID，与实体去重键一致。"""
    return "ent_" + hashlib.blake2b(
        "\x00".join((folder_id, name, type_)).encode(), digest_size=8
    ).hexdigest()


def relation_id(from_id: str, to_id: str, type_: str) -> str:
    """由两端实体ID和关系类型生成关系ID，与关系去重键一致。"""
    return "rel_" + hashlib.blake2b(
        "\x00".join((from_id, to_id, type_)).encode(), digest_size=8
    ).hexdigest()


class ElementMeta(BaseModel):
    """单个元素的元数据，用于溯源匹配"""
    element_id: int = Field(..., description="元素ID")
    type: str = Field(default="text", description="元素类型: text/image/table")
    md: str = Field(default="", description="元素的markdown文本")
    page: int = Field(default=0, description="所在页码")
    rows: Optional[list] = Field(default=None, description="表格行数据，仅table类型")


class FileMeta(BaseModel):
    """文件在 file_index 中的元数据，用于增量 diff"""
    file_id: str = Field(..., description="文件ID")
    piece_hashes: list[str] = Field(default_factory=list, description="该文件所有 piece 的 content_hash")


class FileInput(BaseModel):
    """创建任务请求中的文件输入"""
    id: str = Field(..., description="文件ID")
    name: str = Field(..., description="文件名")
    elements: list[ElementMeta] = Field(default_factory=list, description="文件所有元素元数据列表")


class JobCreateRequest(BaseModel):
    """创建知识图谱生成任务请求"""
    folderName: str = Field(..., description="知识库名称")
    folderId: str = Field(..., description="知识库唯一标识 ID")
    files: list[FileInput] = Field(..., description="所有文件解析结果")


class JobCreateData(BaseModel):
    """创建任务响应数据"""
    jobId: str = Field(..., description="任务ID")


class JobCreateResponse(BaseModel):
    """创建任务响应"""
    code: str = "0"
    message: str = "success"
    data: JobCreateData


class JobItem(BaseModel):
    """任务列表中的单条记录"""
    job_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    failureReason: Optional[str] = Field(default=None, description="失败原因")
    folderId: str = Field(default="", description="知识库ID")
    incremental: bool = Field(default=False, description="是否增量任务")


class JobListData(BaseModel):
    """任务列表响应数据"""
    content: list[JobItem] = Field(default_factory=list, description="任务列表")


class JobListResponse(BaseModel):
    """获取任务列表响应"""
    code: str = "0"
    message: str = "success"
    total: int = 0
    data: JobListData


class FileMetadata(BaseModel):
    """任务详情中的文件元信息"""
    id: str = Field(..., description="文件ID")
    name: str = Field(..., description="文件名")


class JobDetailMetadata(BaseModel):
    """任务详情中的元数据"""
    folderId: str = Field(..., description="知识库ID")
    folderName: str = Field(..., description="知识库名称")
    files: list[FileMetadata] = Field(default_factory=list, description="文件列表")


class GraphSummary(BaseModel):
    """图谱摘要信息"""
    entities: int = Field(default=0, description="实体数量")
    relations: int = Field(default=0, description="关系数量")


class JobDetailData(BaseModel):
    """任务详情响应数据"""
    status: str = Field(default="", description="任务状态: pending/running/success/failed")
    failureReason: Optional[str] = Field(default=None, description="失败原因")
    graphId: Optional[str] = Field(default=None, description="图谱ID")
    metadata: Optional[JobDetailMetadata] = None
    summary: Optional[GraphSummary] = None
    


class JobDetailResponse(BaseModel):
    """获取任务详情响应"""
    code: str = "0"
    message: str = "success"
    data: JobDetailData


class Reference(BaseModel):
    """实体/关系的来源引用"""
    source: str = Field(..., description="来源文本")
    element_id: int = Field(default=0, description="LLM指定的元素ID，无效时为0")
    file_id: str = Field(..., description="文件ID")
    page: Optional[int] = Field(default=None, description="来源页码")
    source_matched: bool = Field(default=False, description="来源文本是否匹配指定元素")
    confidence: Optional[Literal["High", "Medium", "Low"]] = Field(
        default=None, description="该来源支持实体或关系的置信度"
    )
    confidence_reason: Optional[str] = Field(default=None, description="Medium或Low置信度的原因")

    @model_validator(mode="before")
    @classmethod
    def infer_legacy_source_matched(cls, data: Any) -> Any:
        """兼容未存储 source_matched 的旧图谱。"""
        if isinstance(data, dict) and "source_matched" not in data:
            data = {**data, "source_matched": bool(data.get("element_id", 0))}
        return data


class Entity(BaseModel):
    """知识图谱实体"""
    id: str = Field(default="", description="实体ID，由 folder_id+name+type 哈希生成")
    name: str = Field(..., description="实体名称")
    type: str = Field(..., description="中文类型名称")
    type_en: str = Field(..., description="英文类型名")
    attributes: dict[str, Any] = Field(default_factory=dict, description="实体属性")
    references: list[Reference] = Field(..., description="来源引用")
    degree: int = Field(default=0, description="连接关系数量")
    embed: Optional[list[float]] = Field(
        default=None, exclude=True, description="实体向量"
    )


class Relation(BaseModel):
    """知识图谱关系"""
    id: str = Field(default="", description="关系ID，由两端实体ID+type 哈希生成")
    type: str = Field(..., description="关系类型: 主借款人, 保证人, 实际控制等")
    type_en: str = Field(..., description="英文类型名")
    from_entity: str = Field(...,description="起始实体名称")
    from_entity_id: str = Field(default="", description="起始实体ID")
    to_entity: str = Field(..., description="目标实体名称")
    to_entity_id: str = Field(default="", description="目标实体ID")
    attributes: dict[str, list[str]] = Field(
        default_factory=dict, description="关系属性"
    )
    references: list[Reference] = Field(..., description="来源引用")
    degree: int = Field(default=0, description="连接实体数量和")
    embed: Optional[list[float]] = Field(
        default=None, exclude=True, description="关系向量"
    )


class KnowledgeGraph(BaseModel):
    """知识图谱完整数据"""
    id: str = Field(..., description="图谱ID")
    snapshot_version: str = Field(
        default="", description="图谱保存快照版本"
    )
    folderId: str = Field(..., description="知识库ID")
    folderName: str = Field(..., description="知识库名称")
    files: list[FileInput] = Field(default_factory=list, description="所有文件解析结果")
    entities: list[Entity] = Field(default_factory=list, description="实体列表")
    relations: list[Relation] = Field(default_factory=list, description="关系列表")
    piece_index: dict[str, list[str]] = Field(
        default_factory=dict,
        description="piece_hash -> [file_id, ...]，用于 piece 级去重；删除文件时同步清理",
    )
    file_index: dict[str, FileMeta] = Field(
        default_factory=dict,
        description="file_id -> FileMeta，用于文件级 diff",
    )
    

class GraphData(BaseModel):
    """图谱数据"""
    totalEntities: int = Field(default=0, description="实体总数")
    totalRelations: int = Field(default=0, description="关系总数")
    entities: list[Entity] = Field(default_factory=list, description="实体列表")
    relations: list[Relation] = Field(default_factory=list, description="关系列表")


class GraphGetResponse(BaseModel):
    """获取图谱响应"""
    code: str = "0"
    message: str = "success"
    data: GraphData


class Piece(BaseModel):
    """文档切片"""
    id: str = Field(..., description="切片ID")
    content: str = Field(..., description="切片内容")
    content_hash: str = Field(default="", description="切片内容 md5，用于 piece 级去重")
    file_id: str = Field(default="", description="来源文件ID")
    file_name: str = Field(default="", description="来源文件名")
    elements: list[ElementMeta] = Field(default_factory=list, description="该切片覆盖的元素元数据列表")


class QueryRequest(BaseModel):
    """知识图谱查询请求"""
    folderId: str = Field(..., description="知识库ID")
    query: str = Field(..., description="查询问题")


class QueryData(BaseModel):
    """知识图谱查询响应数据"""
    answer: str = Field(..., description="回答内容")
    entities: list[Entity] = Field(default_factory=list, description="召回实体列表")
    relations: list[Relation] = Field(default_factory=list, description="扩展关系列表")


class QueryResponse(BaseModel):
    """知识图谱查询响应"""
    code: str = "0"
    message: str = "success"
    data: QueryData
