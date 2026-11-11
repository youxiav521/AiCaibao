from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from typing import Optional

from loguru import logger

from src.applications.parsing.service.subject_matching.en.text_rules import deep_normalize_english
from src.applications.parsing.schemas.subject_match_schema import Cell, Report, Subject, Table


def _deep(text: str) -> str:
    return deep_normalize_english(text or "")


def whole_table_same_id_dedup(
    table: Table,
    subject_id_to_name: dict[str, str],
    pre_matched_cell_ids: set[str],
) -> None:
    sid_to_cells: dict[str, list[Cell]] = defaultdict(list)
    for row in table.rows:
        for cell in row:
            if cell.subjectId:
                sid_to_cells[cell.subjectId].append(cell)

    for sid, cells in sid_to_cells.items():
        if len(cells) <= 1:
            continue

        tname = subject_id_to_name.get(sid, sid)
        deep_tname = _deep(tname)
        scored = sorted(
            cells,
            key=lambda c: (
                _deep(c.text.strip()) == deep_tname,
                c.text.strip().lower() == tname.lower(),
                SequenceMatcher(None, c.text.strip().lower(), tname.lower()).ratio(),
                c.cellId in pre_matched_cell_ids,
            ),
            reverse=True,
        )
        keep = scored[0]
        for cell in scored[1:]:
            origin = "Java" if cell.cellId in pre_matched_cell_ids else "Python"
            cell.subjectId = None
            logger.warning(
                "[EN Same-ID] template={!r}: KEPT={!r} vs CLEARED={!r} ({})",
                tname,
                keep.text.strip(),
                cell.text.strip(),
                origin,
            )


def enforce_one_to_one(
    table: Table,
    subject_id_to_name: dict[str, str],
    pre_matched_cell_ids: set[str],
) -> None:
    tname_to_cells: dict[str, list[tuple[str, Cell]]] = defaultdict(list)
    for row in table.rows:
        for cell in row:
            sid = cell.subjectId
            if not sid:
                continue
            tname = subject_id_to_name.get(sid, sid)
            deep_tname = _deep(tname)
            if deep_tname:
                tname_to_cells[deep_tname].append((tname, cell))

    for deep_tname, pairs in tname_to_cells.items():
        if len(pairs) <= 1:
            continue

        scored: list[tuple[Cell, tuple[bool, bool, float, bool]]] = []
        for tname, cell in pairs:
            src = cell.text.strip()
            score = (
                _deep(src) == deep_tname,
                src.lower() == tname.lower(),
                SequenceMatcher(None, src.lower(), tname.lower()).ratio(),
                cell.cellId in pre_matched_cell_ids,
            )
            scored.append((cell, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        keep_cell = scored[0][0]
        for cell, _score in scored[1:]:
            origin = "Java" if cell.cellId in pre_matched_cell_ids else "Python"
            cell.subjectId = None
            logger.warning(
                "[EN Enforce-1:1] template_family={!r}: KEPT={!r} vs CLEARED={!r} ({})",
                deep_tname,
                keep_cell.text.strip(),
                cell.text.strip(),
                origin,
            )


def resolve_aggregate_to_leaf(
    text_to_first_cells: dict[str, list[Cell]],
    report: Report,
) -> None:
    subject_by_id: dict[str, Subject] = {s.subjectId: s for s in report.subjectList}
    leaf_by_deep_name: dict[str, Subject] = {}
    for subject in report.subjectList:
        if subject.summaryRelation:
            continue
        deep_name = _deep(subject.subjectName.strip())
        if deep_name and deep_name not in leaf_by_deep_name:
            leaf_by_deep_name[deep_name] = subject
        for synonym in subject.synonymSubject:
            deep_synonym = _deep(synonym.strip())
            if deep_synonym and deep_synonym not in leaf_by_deep_name:
                leaf_by_deep_name[deep_synonym] = subject

    for text, cells in text_to_first_cells.items():
        sid = cells[0].subjectId if cells else None
        if not sid:
            continue
        matched = subject_by_id.get(sid)
        if not matched or not matched.summaryRelation:
            continue

        deep_text = _deep(text)
        best_leaf: Optional[Subject] = None
        for child_id in matched.summaryRelation:
            child = subject_by_id.get(child_id)
            if child is None:
                candidate = leaf_by_deep_name.get(_deep(child_id.strip()))
                if candidate and _deep(child_id.strip()) == deep_text:
                    best_leaf = candidate
                    break
                continue
            if _deep(child.subjectName.strip()) == deep_text:
                best_leaf = child
                break
            for synonym in child.synonymSubject:
                if _deep(synonym.strip()) == deep_text:
                    best_leaf = child
                    break
            if best_leaf:
                break

        if not best_leaf:
            best_leaf = leaf_by_deep_name.get(deep_text)

        if best_leaf:
            for cell in cells:
                cell.subjectId = best_leaf.subjectId
            logger.info(
                "[EN Aggregate] resolved to leaf: {!r}: {} -> {}",
                text,
                matched.subjectName,
                best_leaf.subjectName,
            )


def try_spread_to_leaves(
    texts: list[str],
    text_to_first_cells: dict[str, list[Cell]],
    aggregate: Subject,
    subject_by_id: dict[str, Subject],
) -> bool:
    available_leaves = [
        subject_by_id[child_id]
        for child_id in aggregate.summaryRelation
        if child_id in subject_by_id and not subject_by_id[child_id].summaryRelation
    ]
    if not available_leaves:
        return False

    assignments: dict[str, Subject] = {}
    used_leaf_ids: set[str] = set()
    for text in texts:
        deep_text = _deep(text)
        for leaf in available_leaves:
            if leaf.subjectId in used_leaf_ids:
                continue
            if _deep(leaf.subjectName.strip()) == deep_text or any(
                _deep(synonym.strip()) == deep_text for synonym in leaf.synonymSubject
            ):
                assignments[text] = leaf
                used_leaf_ids.add(leaf.subjectId)
                break

    if len(assignments) < len(texts):
        return False

    for text, leaf in assignments.items():
        for cell in text_to_first_cells[text]:
            cell.subjectId = leaf.subjectId
    return True


def whole_table_source_family_dedup(
    table: Table,
    subject_id_to_name: dict[str, str],
    pre_matched_cell_ids: set[str],
) -> None:
    return None


def whole_table_name_family_dedup(
    table: Table,
    subject_id_to_name: dict[str, str],
    pre_matched_cell_ids: set[str],
) -> None:
    return None


def apply_net_amount_exclusion(
    text_to_first_cells: dict[str, list[Cell]],
    subject_id_to_name: dict[str, str],
) -> None:
    return None


def apply_name_family_dedup(
    text_to_first_cells: dict[str, list[Cell]],
    subject_id_to_name: dict[str, str],
) -> None:
    return None


def reject_income_statement_matches(
    text_to_first_cells: dict[str, list[Cell]],
    subject_id_to_name: dict[str, str],
) -> None:
    return None


def noop_text_cells_postprocess(
    text_to_first_cells: dict[str, list[Cell]],
    subject_id_to_name: dict[str, str],
) -> None:
    return None


def noop_table_postprocess(
    table: Table,
    subject_id_to_name: dict[str, str],
    pre_matched_cell_ids: set[str],
) -> None:
    return None
