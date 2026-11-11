from .schemas import (
    CandidateExtractionResult,
    CandidateJudgment,
    ConsistencyCandidate,
    ConsistencyEvidence,
    ConsistencyIssue,
    ConsistencySourceLocation,
    CrossReference,
    DocumentSegment,
    FactMention,
    LlmCandidateOutline,
    SingleDocConsistencyCheckRequest,
    SingleDocConsistencyCheckResult,
    TermDefinition,
)

__all__ = [
    "single_file_consistency_checks_router",
    "SingleDocConsistencyCheckRequest",
    "SingleDocConsistencyCheckResult",
    "ConsistencyIssue",
    "ConsistencyEvidence",
    "ConsistencySourceLocation",
    "DocumentSegment",
    "TermDefinition",
    "FactMention",
    "CrossReference",
    "ConsistencyCandidate",
    "LlmCandidateOutline",
    "CandidateExtractionResult",
    "CandidateJudgment",
    "SingleDocConsistencyService",
]


def __getattr__(name: str):
    if name == "single_file_consistency_checks_router":
        from .router import single_file_consistency_checks_router

        return single_file_consistency_checks_router
    if name == "SingleDocConsistencyService":
        from .service import SingleDocConsistencyService

        return SingleDocConsistencyService
    raise AttributeError(name)
