"""知识图谱查询意图 Pydantic 模型。"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class QueryIntent(BaseModel):
    """LLM 意图分析的结构化输出"""

    query_type: Literal["attribute_query", "relation_query", "multi_hop", "aggregation", "rag"] = Field(
        default="rag",
        description=(
            "attribute_query: 询问实体属性值（注册资本、身份证号等）; "
            "relation_query: 询问实体间关系（担保人、股东等）; "
            "multi_hop: 需要多步关系穿透（股权穿透、关联图谱等）; "
            "aggregation: 统计计数（几家企业、几个自然人等）; "
            "rag: 兜底类型"
        ),
    )
    target_entity_types: list[str] = Field(
        default_factory=list,
        description="目标实体类型，从可用类型中选择",
    )
    target_relation_types: list[str] = Field(
        default_factory=list,
        description="目标关系类型，从可用类型中选择",
    )
    requested_attributes: list[str] = Field(
        default_factory=list,
        description="用户明确询问的属性名（attribute_query 时必填）",
    )
    max_hops: int = Field(
        default=0,
        description="多跳遍历最大层数（multi_hop 时 >= 2）",
    )
    aggregation: Optional[str] = Field(
        default=None,
        description="聚合方式：count_by_type",
    )
