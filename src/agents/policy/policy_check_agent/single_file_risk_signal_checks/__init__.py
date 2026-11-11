from .schemas import (
    RiskSignalEvidence,
    RiskSignalIssue,
    RiskSignalSourceLocation,
    SingleFileRiskSignalCheckRequest,
    SingleFileRiskSignalCheckResult,
)

__all__ = [
    "SingleFileRiskSignalCheckRequest",
    "SingleFileRiskSignalCheckResult",
    "RiskSignalIssue",
    "RiskSignalEvidence",
    "RiskSignalSourceLocation",
    "single_file_risk_signal_checks_router",
    "SingleFileRiskSignalCheckService",
]


def __getattr__(name: str):
    if name == "single_file_risk_signal_checks_router":
        from .router import single_file_risk_signal_checks_router

        return single_file_risk_signal_checks_router
    if name == "SingleFileRiskSignalCheckService":
        from .service import SingleFileRiskSignalCheckService

        return SingleFileRiskSignalCheckService
    raise AttributeError(name)
