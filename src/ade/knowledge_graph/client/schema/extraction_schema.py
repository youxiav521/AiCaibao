"""LLM 结构化输出的 Pydantic 模型与类型映射表。

从 definitions.py 派生，供 KGClient 与 ExtractorService 共享。
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.ade.knowledge_graph.client.schema.builder import SchemaBuilder


ENTITY_TYPE_MAP = SchemaBuilder.entity_type_map()
RELATION_TYPE_MAP = SchemaBuilder.relation_type_map()


# ---------- LLM 结构化输出 schema（function-calling 直出结构） ----------


class ExtractedAttribute(BaseModel):
    """实体或关系属性键值对"""
    name: str = Field(description="属性名，如 曾用名、身份证号、注册资本、股票代码、权证编号、事件类型")
    value: str = Field(description="属性值")


class ExtractedEntity(BaseModel):
    """从文本中抽取的单个实体"""
    name: str = Field(
        description="实体名称；英文保留原格式大小写；使用第三人称，禁止使用代词（如本公司、该公司、我司）。必须是输入正文或有效文档元数据中的一段连续文本。"
    )
    type: str = Field(
        description=f"实体类型，必须是以下之一：{'、'.join(ENTITY_TYPE_MAP)}"
    )
    source: str = Field(
        description="溯源信息：正文实体逐字复制 <Input Text> 原文；仅来自元数据的实体逐字复制对应元数据字段。"
    )
    element_id: int = Field(
        description="正文 source 所在 element 的 ID；实体仅来自元数据时固定为 0。"
    )
    confidence: Literal["High", "Medium", "Low"] = Field(
        description="置信度。High：原文直接明确支持，无需推断且无冲突；Medium：较可靠但存在有限不确定性；Low：有倾向性证据但不确定性显著。"
    )
    confidence_reason: Optional[str] = Field(
        default=None,
        description="High 时留空；Medium/Low 时必填，格式为『原因类别：具体说明』，原因类别仅限：证据不足、OCR噪声、类型歧义、指代歧义。",
    )
    attributes: list[ExtractedAttribute] = Field(
        default_factory=list, description="实体自身属性列表；无属性时为空列表，不要编造。"
    )


class ExtractedRelation(BaseModel):
    """从文本中抽取的单个二元关系"""
    from_entity: str = Field(
        description="源实体名称，与实体抽取严格一致；可由正文或有效文档元数据补全。"
    )
    type: str = Field(
        description=f"关系类型，必须取自：{'、'.join(RELATION_TYPE_MAP)}。每条关系只能填写一个类型；同一实体对存在多种关系时分别输出多条关系。"
    )
    to_entity: str = Field(
        description="目标实体名称，与实体抽取严格一致；可由正文或有效文档元数据补全。项目关系的目标实体固定为本次项目名称。"
    )
    source: str = Field(
        description="溯源信息：逐字来自输入文本 <Input Text> 的原文句子或完整 Markdown 数据行，需包含支持该关系类型与方向的原文信号。"
    )
    element_id: int = Field(
        description="source 所在 element 的 ID，必须从对应 <element> 标签原样返回。"
    )
    confidence: Literal["High", "Medium", "Low"] = Field(description="置信度，语义同实体。")
    confidence_reason: Optional[str] = Field(
        default=None,
        description="High 时留空；Medium/Low 时必填，格式为『原因类别：具体说明』。",
    )
    attributes: list[ExtractedAttribute] = Field(
        default_factory=list,
        description="关系属性列表；仅使用关系类型定义中的属性，且必须有原文依据。无属性时为空列表。",
    )


class KGExtractionResult(BaseModel):
    """知识图谱抽取结果：实体列表 + 关系列表"""
    entities: list[ExtractedEntity] = Field(
        default_factory=list, description="抽取到的实体列表"
    )
    relations: list[ExtractedRelation] = Field(
        default_factory=list, description="抽取到的关系列表"
    )
