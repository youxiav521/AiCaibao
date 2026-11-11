from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


EntityKey = tuple[str, str]
RelationTriple = tuple[str, str, str]
AttributeKey = tuple[str, str, str]
AttributeAtom = tuple[str, str, str, str]
RelationAttributeKey = tuple[str, str, str, str]
RelationAttributeAtom = tuple[str, str, str, str, str]
GraphFormat = Literal["auto", "txt", "json"]
ConfidenceLevel = Literal["High", "Medium", "Low"]

KEY_RELATION_TYPES: tuple[str, ...] = ("主借款人", "保证人", "担保人", "出质人", "抵押人")
EXPECTED_TO_ENTITY = "测试知识库"

ENTERPRISE_ATTR_ALIASES: dict[str, str] = {
    "code": "企业证件号",
    "证件号": "企业证件号",
    "统一社会信用代码": "企业证件号",
}


class EvaluationEntity(BaseModel):
    name: str = Field(..., description="实体名称")
    type: str = Field(..., description="中文实体类型")
    type_en: str = Field(..., description="英文实体类型")
    attributes: dict[str, str] = Field(default_factory=dict, description="实体属性")
    confidence: ConfidenceLevel | None = Field(default=None, description="置信度")


class EvaluationRelation(BaseModel):
    from_entity: str = Field(..., description="起始实体名称")
    type: str = Field(..., description="中文关系类型")
    type_en: str = Field(default="", description="英文关系类型")
    to_entity: str = Field(..., description="目标实体名称")
    attributes: dict[str, list[str]] = Field(default_factory=dict, description="关系属性")
    confidence: ConfidenceLevel | None = Field(default=None, description="置信度")


class EvaluationGraph(BaseModel):
    entities: list[EvaluationEntity] = Field(default_factory=list)
    relations: list[EvaluationRelation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class MetricResult(BaseModel):
    tp: int
    fp: int
    fn: int
    correct: set[Any] = Field(default_factory=set)
    missing: set[Any] = Field(default_factory=set)
    extra: set[Any] = Field(default_factory=set)

    @property
    def accuracy(self) -> float:
        denom = self.tp + self.fp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def zero_denominator_notes(self) -> list[str]:
        notes: list[str] = []
        if self.tp + self.fp + self.fn == 0:
            notes.append("accuracy 分母为 0，记为 0.0000")
        if self.tp + self.fp == 0:
            notes.append("precision 分母为 0，记为 0.0000")
        if self.tp + self.fn == 0:
            notes.append("recall 分母为 0，记为 0.0000")
        if self.precision + self.recall == 0:
            notes.append("f1 分母为 0，记为 0.0000")
        return notes


class AttributeEvalResult(BaseModel):
    metrics: MetricResult
    value_wrong: list[tuple[AttributeKey, set[str], set[str]]] = Field(default_factory=list)


class RelationAttributeEvalResult(BaseModel):
    metrics: MetricResult
    value_wrong: list[tuple[RelationAttributeKey, set[str], set[str]]] = Field(
        default_factory=list
    )


class ConfidenceEvaluationResult(BaseModel):
    entity: MetricResult
    relation: MetricResult
    key_relation: MetricResult


class ConfidenceDetectionResult(BaseModel):
    true_errors: int = 0
    flagged: int = 0
    flagged_errors: int = 0

    @property
    def precision(self) -> float:
        return self.flagged_errors / self.flagged if self.flagged else 0.0

    @property
    def recall(self) -> float:
        return self.flagged_errors / self.true_errors if self.true_errors else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0


class ConfidenceDetectionSummary(BaseModel):
    entity: ConfidenceDetectionResult
    relation: ConfidenceDetectionResult
    overall: ConfidenceDetectionResult


def _empty_confidence_evaluation() -> ConfidenceEvaluationResult:
    return ConfidenceEvaluationResult(
        entity=MetricResult(tp=0, fp=0, fn=0),
        relation=MetricResult(tp=0, fp=0, fn=0),
        key_relation=MetricResult(tp=0, fp=0, fn=0),
    )


def _empty_confidence_detection() -> ConfidenceDetectionSummary:
    return ConfidenceDetectionSummary(
        entity=ConfidenceDetectionResult(),
        relation=ConfidenceDetectionResult(),
        overall=ConfidenceDetectionResult(),
    )


class EvaluationOptions(BaseModel):
    include_attributes: bool = Field(default=False, description="是否评估属性")
    list_limit: int = Field(default=0, ge=0, description="普通条目最大列举数量，0 表示全部")


class EvaluationResult(BaseModel):
    gold_path: Path
    target_path: Path
    gold: EvaluationGraph
    target: EvaluationGraph
    entity_all: MetricResult
    relation: MetricResult
    attribute: AttributeEvalResult | None = None
    relation_attribute: RelationAttributeEvalResult | None = None
    key_relation_total: MetricResult
    confidence_exclude_low: ConfidenceEvaluationResult = Field(
        default_factory=_empty_confidence_evaluation
    )
    confidence_high_only: ConfidenceEvaluationResult = Field(
        default_factory=_empty_confidence_evaluation
    )
    confidence_detection: ConfidenceDetectionSummary = Field(
        default_factory=_empty_confidence_detection
    )
    options: EvaluationOptions = Field(default_factory=EvaluationOptions)
