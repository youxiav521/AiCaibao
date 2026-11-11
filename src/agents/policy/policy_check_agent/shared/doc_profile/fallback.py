from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.agents.content_generate_agent.utils import ainvoke_llm_with_structured_output

from .classifier import classify_doc_profile
from .schemas import DocProfile, DocSignal, DocType


class _DocProfileFallbackDecision(BaseModel):
    primary_type: str = Field(default="UNKNOWN", description="Primary doc type enum value")
    candidate_types: List[str] = Field(default_factory=list, description="Candidate doc type enum values")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


def should_trigger_llm_fallback(
    profile: DocProfile,
    *,
    confidence_threshold: float = 0.72,
    margin_threshold: float = 0.18,
) -> bool:
    if profile.primary_type == DocType.UNKNOWN:
        return True
    if profile.confidence < confidence_threshold:
        return True
    hints = profile.raw_hints or {}
    top_score = float(hints.get("topScore") or 0.0)
    top2_score = float(hints.get("top2Score") or 0.0)
    if top_score <= 0:
        return True
    margin_ratio = (top_score - top2_score) / top_score
    return margin_ratio < margin_threshold


async def classify_doc_profile_with_llm_fallback(
    *,
    file_name: str,
    doc_content: List[Dict[str, Any]],
    model_id: str = "model_bailian_policy_check",
    enable_llm_fallback: bool = True,
    confidence_threshold: float = 0.72,
    margin_threshold: float = 0.18,
    max_text_chars: int = 5000,
) -> DocProfile:
    base_profile = classify_doc_profile(file_name=file_name, doc_content=doc_content)
    if not enable_llm_fallback:
        return base_profile
    if not should_trigger_llm_fallback(
        base_profile,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
    ):
        return base_profile

    prompt = _build_fallback_prompt(
        file_name=file_name,
        doc_content=doc_content,
        base_profile=base_profile,
        max_text_chars=max_text_chars,
    )
    try:
        decision = await ainvoke_llm_with_structured_output(
            prompt,
            model_id=model_id,
            response_type=_DocProfileFallbackDecision,
        )
    except Exception:
        return base_profile

    return _merge_fallback_decision(base_profile, decision)


def _build_fallback_prompt(
    *,
    file_name: str,
    doc_content: List[Dict[str, Any]],
    base_profile: DocProfile,
    max_text_chars: int,
) -> str:
    text_sample = _collect_text_sample(doc_content, max_text_chars=max_text_chars)
    allowed_types = [doc_type.value for doc_type in DocType]
    base_scores = base_profile.scores or {}
    return (
        "你是文档类型分类器。请基于文件名和内容，判断文档类型。\n"
        f"允许的类型枚举: {allowed_types}\n"
        f"文件名: {file_name}\n"
        f"规则分类器建议 primary={base_profile.primary_type.value}, candidates={[t.value for t in base_profile.doc_types]}, "
        f"confidence={base_profile.confidence}, scores={base_scores}\n"
        "文档内容片段如下：\n"
        f"{text_sample}\n"
        "要求：\n"
        "1) primary_type 必须是允许枚举之一。\n"
        "2) candidate_types 至少包含 primary_type。\n"
        "3) confidence 取 0~1。\n"
        "4) reason 简要说明依据，优先引用关键词/字段锚点。\n"
    )


def _collect_text_sample(doc_content: List[Dict[str, Any]], *, max_text_chars: int) -> str:
    parts: List[str] = []
    total = 0
    for item in doc_content or []:
        text = _item_text(item)
        if not text:
            continue
        remaining = max_text_chars - total
        if remaining <= 0:
            break
        snippet = text[:remaining]
        parts.append(snippet)
        total += len(snippet)
    return "\n".join(parts)[:max_text_chars]


def _item_text(item: Any) -> str:
    if isinstance(item, dict):
        for key in ("md", "value", "text", "image_caption", "ocr", "content"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
    for key in ("md", "value", "text", "image_caption", "ocr", "content"):
        value = getattr(item, key, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _merge_fallback_decision(
    base_profile: DocProfile,
    decision: _DocProfileFallbackDecision,
) -> DocProfile:
    primary = _to_doc_type(decision.primary_type) or base_profile.primary_type
    merged_candidates: List[DocType] = [primary]
    for value in decision.candidate_types:
        doc_type = _to_doc_type(value)
        if doc_type and doc_type not in merged_candidates:
            merged_candidates.append(doc_type)
    if not merged_candidates:
        merged_candidates = list(base_profile.doc_types) or [base_profile.primary_type]

    merged_confidence = max(base_profile.confidence, min(0.99, float(decision.confidence or 0.0)))
    merged_signals = list(base_profile.signals or [])
    merged_signals.append(
        DocSignal(
            source="rule",
            token="llm_fallback",
            weight=round(merged_confidence - base_profile.confidence, 4),
            matchedSpan=(decision.reason or "")[:120] or None,
        )
    )
    merged_hints = dict(base_profile.raw_hints or {})
    merged_hints["llmFallback"] = {
        "enabled": True,
        "primaryType": primary.value,
        "candidateTypes": [doc_type.value for doc_type in merged_candidates],
        "confidence": merged_confidence,
        "reason": decision.reason,
    }
    return DocProfile(
        primaryType=primary,
        docTypes=merged_candidates[:3],
        confidence=round(merged_confidence, 4),
        isMedia=primary in {DocType.AUDIO, DocType.VIDEO},
        scores=base_profile.scores,
        signals=merged_signals[:24],
        rawHints=merged_hints,
    )


def _to_doc_type(value: Optional[str]) -> Optional[DocType]:
    if not value:
        return None
    normalized = value.strip().upper()
    for doc_type in DocType:
        if doc_type.value == normalized:
            return doc_type
    return None
