from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class RelevanceEvalSample:
    sample_id: str
    expected_label: str
    predicted_decision: str
    final_result: str
    llm_used: bool = False
    llm_corrected: bool = False


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def evaluate_relevance_metrics(samples: Iterable[RelevanceEvalSample]) -> Dict[str, float]:
    data: List[RelevanceEvalSample] = list(samples)
    if not data:
        return {
            "hard_skip_false_kill_rate": 0.0,
            "soft_skip_should_review_ratio": 0.0,
            "review_recall": 0.0,
            "fast_fail_false_judgment_rate": 0.0,
            "gray_zone_accuracy_gain": 0.0,
        }

    hard_skip_total = sum(1 for item in data if item.predicted_decision == "hard_skip")
    hard_skip_false_kill = sum(
        1
        for item in data
        if item.predicted_decision == "hard_skip" and item.expected_label in {"clearly relevant", "weakly relevant", "ambiguous"}
    )

    soft_skip_total = sum(1 for item in data if item.predicted_decision == "soft_skip")
    soft_skip_should_review = sum(
        1
        for item in data
        if item.predicted_decision == "soft_skip" and item.expected_label in {"weakly relevant", "ambiguous"}
    )

    review_expected = sum(1 for item in data if item.expected_label in {"clearly relevant", "weakly relevant", "ambiguous"})
    review_hit = sum(
        1
        for item in data
        if item.expected_label in {"clearly relevant", "weakly relevant", "ambiguous"}
        and item.predicted_decision in {"review", "fast_fail_candidate"}
    )

    fast_fail_total = sum(1 for item in data if item.predicted_decision == "fast_fail_candidate")
    fast_fail_false = sum(
        1
        for item in data
        if item.predicted_decision == "fast_fail_candidate" and item.final_result == "fail" and item.expected_label != "clearly irrelevant"
    )

    gray_zone_llm = [item for item in data if item.llm_used]
    before_correct = sum(1 for item in gray_zone_llm if not item.llm_corrected)
    after_correct = sum(1 for item in gray_zone_llm if item.llm_corrected)
    gray_zone_accuracy_gain = _ratio(after_correct - before_correct, max(1, len(gray_zone_llm)))

    return {
        "hard_skip_false_kill_rate": _ratio(hard_skip_false_kill, hard_skip_total),
        "soft_skip_should_review_ratio": _ratio(soft_skip_should_review, soft_skip_total),
        "review_recall": _ratio(review_hit, review_expected),
        "fast_fail_false_judgment_rate": _ratio(fast_fail_false, fast_fail_total),
        "gray_zone_accuracy_gain": gray_zone_accuracy_gain,
    }


def build_minimal_offline_eval_set() -> List[RelevanceEvalSample]:
    return [
        RelevanceEvalSample(
            sample_id="sample_relevant_001",
            expected_label="clearly relevant",
            predicted_decision="review",
            final_result="review",
        ),
        RelevanceEvalSample(
            sample_id="sample_weak_001",
            expected_label="weakly relevant",
            predicted_decision="review",
            final_result="review",
            llm_used=True,
            llm_corrected=True,
        ),
        RelevanceEvalSample(
            sample_id="sample_irrelevant_001",
            expected_label="clearly irrelevant",
            predicted_decision="soft_skip",
            final_result="skipped",
        ),
        RelevanceEvalSample(
            sample_id="sample_ambiguous_001",
            expected_label="ambiguous",
            predicted_decision="review",
            final_result="uncertain",
            llm_used=True,
            llm_corrected=False,
        ),
    ]
