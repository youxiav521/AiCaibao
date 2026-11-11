from .schemas import (
    CompletenessCheckItem,
    CompletenessEvidence,
    CompletenessSourceLocation,
    DocumentTypeClassification,
    SectionExistenceJudgment,
    SectionGroupClassification,
    SingleFileCompletenessCheckRequest,
    SingleFileCompletenessCheckResult,
)

__all__ = [
    "SingleFileCompletenessCheckRequest",
    "SingleFileCompletenessCheckResult",
    "CompletenessCheckItem",
    "CompletenessEvidence",
    "CompletenessSourceLocation",
    "DocumentTypeClassification",
    "SectionExistenceJudgment",
    "SectionGroupClassification",
    "single_file_completeness_checks_router",
    "SingleFileCompletenessCheckService",
]


def __getattr__(name: str):
    if name == "single_file_completeness_checks_router":
        from .router import single_file_completeness_checks_router

        return single_file_completeness_checks_router
    if name == "SingleFileCompletenessCheckService":
        from .service import SingleFileCompletenessCheckService

        return SingleFileCompletenessCheckService
    raise AttributeError(name)
