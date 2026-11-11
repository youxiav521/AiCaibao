from .schemas import (
    ControlRetrievalStats,
    SingleFilePolicyAuditedPayload,
    SingleFilePolicyCallbackControlResult,
    SingleFilePolicyCallbackEvidence,
    SingleFilePolicyCallbackResultItem,
    SingleFilePolicyCallbackRule,
    SingleFilePolicyCheckRequest,
    SingleFilePolicyCheckResult,
    SingleFilePolicyCompletedPayload,
    SingleFilePolicyControlResult,
    SingleFilePolicyRequirementResult,
)

__all__ = [
    "single_file_policy_router",
    "SingleFilePolicyCheckRequest",
    "SingleFilePolicyCheckResult",
    "SingleFilePolicyRequirementResult",
    "SingleFilePolicyControlResult",
    "ControlRetrievalStats",
    "SingleFilePolicyAuditedPayload",
    "SingleFilePolicyCompletedPayload",
    "SingleFilePolicyCallbackRule",
    "SingleFilePolicyCallbackEvidence",
    "SingleFilePolicyCallbackControlResult",
    "SingleFilePolicyCallbackResultItem",
    "SingleFilePolicyCheckService",
]


def __getattr__(name: str):
    if name == "single_file_policy_router":
        from .router import single_file_policy_router

        return single_file_policy_router
    if name == "SingleFilePolicyCheckService":
        from .service import SingleFilePolicyCheckService

        return SingleFilePolicyCheckService
    raise AttributeError(name)
