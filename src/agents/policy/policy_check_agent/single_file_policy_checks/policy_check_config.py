from __future__ import annotations

from dataclasses import dataclass

UNRELATED_CONTROL_REASON = "当前单文档与该规则无明显相关性，已自动跳过。"
RELEVANCE_ONLY_REASON = "该规则与当前单文档相关，但因仅启用相关性判断模式，已跳过正式审查。"
DEGRADED_REASON = "校验流程未能稳定完成，已使用自动降级兜底结果。"


@dataclass(slots=True)
class PolicyCheckConfig:
    model_id: str = "model_bailian_policy_check"
    max_concurrency: int = 12
    relevance_threshold: float = 4.5
    include_skipped_controls: bool = True
    enable_relevance_llm_profiles: bool = False
    enable_relevance_llm_adjudication: bool = False
    gray_zone_lower: float = 0.35
    gray_zone_upper: float = 0.55
    execution_retrieval_floor: float = 0.22
    fail_relevance_floor: float = 0.58
    fail_retrieval_floor: float = 0.2
    review_execute_score_floor: float = 0.58
    review_execute_retrieval_floor: float = 0.28
    max_review_execute_controls_per_requirement: int = 12
    single_evidence_max_files: int = 2
    multi_evidence_max_files: int = 4
    repository_evidence_max_files: int = 6
    candidate_file_score_margin: float = 0.06
    llm_veto_confidence: float = 0.62
    llm_promote_confidence: float = 0.78
    llm_review_execute_confirm_confidence: float = 0.62

    def __post_init__(self) -> None:
        self.single_evidence_max_files = max(1, int(self.single_evidence_max_files))
        self.multi_evidence_max_files = max(1, int(self.multi_evidence_max_files))
        self.repository_evidence_max_files = max(1, int(self.repository_evidence_max_files))
        self.candidate_file_score_margin = max(0.0, float(self.candidate_file_score_margin))
        self.llm_veto_confidence = max(0.5, min(0.95, float(self.llm_veto_confidence)))
        self.llm_promote_confidence = max(0.5, min(0.98, float(self.llm_promote_confidence)))
        self.llm_review_execute_confirm_confidence = max(
            0.55,
            min(0.95, float(self.llm_review_execute_confirm_confidence)),
        )
