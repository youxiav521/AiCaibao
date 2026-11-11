from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen=True)
class StatementStrategyResult:
    label: Optional[str]
    confidence: float = 0.0
    reason: str = ""


class StatementExtractionStrategy(Protocol):
    """Region/accounting-standard specific statement recognition hooks."""

    name: str

    def detect_label_from_text(self, text: str, *, locale: Optional[str] = None) -> StatementStrategyResult:
        ...

    def detect_label_from_content(self, table_md: str, *, locale: Optional[str] = None) -> StatementStrategyResult:
        ...

    def should_merge_balance_sheet_halves(self, previous_table: dict, next_table: dict) -> bool:
        ...


class ChinaStatementStrategy:
    """CAS/default strategy.

    This delegates to the existing domestic implementation so the current China
    path remains behavior-compatible while international logic can evolve
    separately.
    """

    name = "china"

    def detect_label_from_text(self, text: str, *, locale: Optional[str] = None) -> StatementStrategyResult:
        from src.applications.parsing.service.three_statements_frontend_service import (
            _normalize_statement_label,
        )

        label = _normalize_statement_label(text)
        return StatementStrategyResult(label=label, confidence=1.0 if label else 0.0)

    def detect_label_from_content(self, table_md: str, *, locale: Optional[str] = None) -> StatementStrategyResult:
        from src.applications.parsing.utils.table_classification_utils import (
            detect_statement_type_by_content,
        )

        label = detect_statement_type_by_content(table_md or "", locale=locale)
        return StatementStrategyResult(label=label, confidence=0.6 if label else 0.0)

    def should_merge_balance_sheet_halves(self, previous_table: dict, next_table: dict) -> bool:
        from src.applications.parsing.service.three_statements_frontend_service import (
            _can_merge_adjacent_balance_sheet_halves,
        )

        return _can_merge_adjacent_balance_sheet_halves(previous_table, next_table)


class InternationalStatementStrategy:
    """International financial statement strategy skeleton.

    The international path intentionally does not change domestic behavior. It
    starts by reusing stable common mechanics, then adds IFRS/English-specific
    merge behavior and prompts in dedicated modules.
    """

    name = "international"

    def detect_label_from_text(self, text: str, *, locale: Optional[str] = None) -> StatementStrategyResult:
        from src.applications.parsing.service.three_statements_frontend_service import (
            _normalize_english_statement_label,
            _normalize_pl_oci_combined_label,
        )

        label = _normalize_pl_oci_combined_label(text or "") or _normalize_english_statement_label(text or "")
        return StatementStrategyResult(label=label, confidence=1.0 if label else 0.0)

    def detect_label_from_content(self, table_md: str, *, locale: Optional[str] = None) -> StatementStrategyResult:
        from src.applications.parsing.service.three_statements_frontend_service import (
            _normalize_pl_oci_combined_label,
        )
        from src.applications.parsing.utils.table_classification_utils import (
            detect_statement_type_by_content,
        )

        label = _normalize_pl_oci_combined_label(table_md or "")
        if label:
            return StatementStrategyResult(label=label, confidence=0.8)
        label = detect_statement_type_by_content(table_md or "", locale=locale or "en")
        return StatementStrategyResult(label=label, confidence=0.5 if label else 0.0)

    def should_merge_balance_sheet_halves(self, previous_table: dict, next_table: dict) -> bool:
        from src.applications.parsing.service.three_statements.international_rules import (
            should_merge_international_balance_sheet_halves,
        )

        return should_merge_international_balance_sheet_halves(previous_table, next_table)


def choose_statement_strategy(strategy: str | None = None, *, locale: Optional[str] = None) -> StatementExtractionStrategy:
    key = (strategy or "").strip().lower()
    if key in {"international", "intl", "ifrs", "us-gaap", "us_gaap"}:
        return InternationalStatementStrategy()
    if key in {"china", "cn", "cas"}:
        return ChinaStatementStrategy()
    if locale and locale.strip().lower().startswith("en"):
        return InternationalStatementStrategy()
    return ChinaStatementStrategy()
