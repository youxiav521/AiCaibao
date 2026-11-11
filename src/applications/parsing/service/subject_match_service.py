from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from src.utils.chinese_converter import traditional_to_simplified

from ..schemas.subject_match_schema import (
    Cell,
    Report,
    Subject,
    SubjectMatchRequest,
    SubjectMatchResponse,
    Table,
)
from ..utils.llm_semantic_matcher import LLMSemanticMatcher
from ..utils.semantic_matcher import (
    match_report_name,
    semantic_match,
)
from .subject_matching.zh import postprocess as _zh_postprocess
from .subject_matching.en import postprocess as _en_postprocess
from .subject_matching import llm_helpers as _llm_helpers
from .subject_matching.column_roles import detect_column_roles
from .subject_matching.strategy import FinancialStatementLanguage, SubjectMatchStrategy
from .subject_matching.strategy_factory import detect_subject_match_language, get_subject_match_strategy
from .subject_matching.text_utils import (
    FAMILY_SUFFIXES_DEDUP,
    classify_family_pair,
    deep_normalize,
    has_trailing_heading_punctuation,
    normalize_subject_text,
    template_text_key,
)

_AGGREGATE_SUFFIXES: tuple[str, ...] = ("净额", "净增加额", "净变动额", "净减少额")
_NET_TYPE_SUFFIXES: tuple[str, ...] = ("净额", "净值", "净损失", "净收益")

# 补充资料（间接法）调节项关键词——这些科目属于间接法附表，不应与直接法主表科目混匹
_INDIRECT_METHOD_KEYWORDS: tuple[str, ...] = (
    "存货的减少", "存货的增加",
    "经营性应收项目的减少", "经营性应收项目的增加",
    "经营性应付项目的增加", "经营性应付项目的减少",
    "待摊费用减少", "预提费用增加",
    "固定资产折旧", "无形资产摊销", "长期待摊费用摊销",
    "处置固定资产", "固定资产报废损失",
    "财务费用", "投资损失", "投资收益",
    "递延税款", "递延所得税",
    "公允价值变动损失", "公允价值变动收益",
    "计提的资产减值准备", "资产减值准备",
)

_SUMMARY_SUFFIXES: tuple[str, ...] = ("合计", "小计", "总计")
_CASHFLOW_SECTION_KEYWORDS: tuple[str, ...] = (
    "经营活动产生的现金流量",
    "投资活动产生的现金流量",
    "筹资活动产生的现金流量",
)
_EPS_KEYWORDS: tuple[str, ...] = ("每股收益", "每股净资产", "每股")
_PROFIT_KEYWORDS: tuple[str, ...] = (
    "净利润", "利润总额", "营业利润", "归属于母公司股东的净利润",
    "持续经营净利润", "非持续经营净利润",
)
_SECTION_HEADING_TERMS: tuple[str, ...] = (
    "流动资产", "非流动资产", "流动负债", "非流动负债",
    "长期投资", "固定资产", "在建工程", "使用权资产",
    "无形资产", "长期待摊费用", "递延资产", "其他长期资产",
)
_SKIP_EXACT: frozenset[str] = frozenset({
    "合计", "小计", "总计", "项目", "科目", "科目名称",
    "期末余额", "期初余额", "本期金额", "上期金额",
    "本年累计", "上年累计", "附注",
    "本年累计数", "上年累计数", "本年同期数", "上年同期数",
    "本期数", "上期数", "期末数", "期初数",
    "本年金额", "上年金额", "本期发生额", "上期发生额",
    "收入", "成本", "金额", "余额", "备注", "说明",
    "行次", "序号", "编号",
    "调整数", "重述数", "调整后", "重分类",
    "资产", "负债", "所有者权益", "股东权益",
    "流动资产", "非流动资产", "流动负债", "非流动负债",
})
_SUPP_SECTION_HEADERS: tuple[str, ...] = (
    "补充资料", "将净利润调节为经营活动现金流量",
    "不涉及现金收支的重大投资和筹资活动",
    "现金及现金等价物净变动情况",
)
_MIN_SIMILARITY_FOR_LLM: float = 0.15
_VALUE_CELL_RE = re.compile(r'^[\d,.\-—\s%()（）±+×÷/＋－＊／]+$')
_ROWSPAN_SEGMENT_MIN_SCORE = 0.82
_ROWSPAN_SPLIT_MIN_AVERAGE_SCORE = 0.88
_ROWSPAN_SEGMENT_MIN_MARGIN = 0.12


class SubjectMatchService:
    """Financial statement subject matching service."""

    def __init__(self, use_llm: bool = True, llm_max_concurrency: int = 10, llm_model_id: str = "model_for_table_classify"):
        self.use_llm = use_llm
        self.llm_max_concurrency = llm_max_concurrency
        self._llm_model_id = llm_model_id
        self._llm_matcher: Optional[LLMSemanticMatcher] = None
        self._current_strategy: SubjectMatchStrategy = get_subject_match_strategy(FinancialStatementLanguage.ZH)
        logger.info(
            "SubjectMatchService initialized: use_llm={}, max_concurrency={}",
            use_llm, llm_max_concurrency,
        )

    def _get_llm_matcher(self) -> Optional[LLMSemanticMatcher]:
        if self.use_llm and self._llm_matcher is None:
            try:
                self._llm_matcher = LLMSemanticMatcher(model_id=self._llm_model_id)
            except Exception as e:
                logger.error("Failed to initialize LLM matcher: {}", e)
                return None
        return self._llm_matcher

    def _postprocess_module(self):
        if self._current_strategy.language == FinancialStatementLanguage.EN:
            return _en_postprocess
        return _zh_postprocess

    @staticmethod
    def _detect_column_roles(table: Table) -> dict[int, str]:
        return detect_column_roles(table)

    @staticmethod
    def _is_cell_in_subject_column(cell: Cell, col_idx: int, column_roles: dict[int, str]) -> bool:
        """Check if a cell belongs to a subject column."""
        if not column_roles:
            return True
        role = column_roles.get(col_idx, "other")
        return role == "subject"

    def _normalize_subject_text(self, text: str) -> str:
        return self._current_strategy.text_rules.normalize_subject_text(text)

    def _deep_normalize(self, text: str) -> str:
        return self._current_strategy.text_rules.deep_normalize(text)

    def _template_text_key(self, text: str) -> str:
        return self._current_strategy.text_rules.template_text_key(text)

    def _deterministic_match_key(self, text: str) -> str | None:
        return self._current_strategy.text_rules.deterministic_match_key(text)

    def _build_template_subject_keys(self, report: Report) -> set[str]:
        """Build exact-match keys from template subjects and their synonyms."""
        keys: set[str] = set()
        for subject in report.subjectList:
            for candidate in (subject.subjectName, *subject.synonymSubject):
                key = self._template_text_key(candidate)
                if key:
                    keys.add(key)
        return keys

    def _find_definitive_template_subject_id(
        self,
        report: Report,
        source_text: str,
    ) -> Optional[str]:
        """Return a unique exact template match without using fuzzy semantics."""
        raw_text = source_text.strip()
        normalized_text = self._normalize_subject_text(raw_text)
        if not raw_text or not normalized_text:
            return None

        match_groups = (
            [
                subject
                for subject in report.subjectList
                if subject.subjectName.strip() == raw_text
            ],
            [
                subject
                for subject in report.subjectList
                if self._normalize_subject_text(subject.subjectName.strip()) == normalized_text
            ],
            [
                subject
                for subject in report.subjectList
                if any(synonym.strip() == raw_text for synonym in subject.synonymSubject)
            ],
            [
                subject
                for subject in report.subjectList
                if any(
                    self._normalize_subject_text(synonym.strip()) == normalized_text
                    for synonym in subject.synonymSubject
                )
            ],
        )

        for matches in match_groups:
            subject_ids = {subject.subjectId for subject in matches}
            if len(subject_ids) == 1:
                return next(iter(subject_ids))
            if len(subject_ids) > 1:
                logger.info(
                    "[definitive-match-ambiguous] text={!r}, subject_ids={}",
                    source_text,
                    sorted(subject_ids),
                )
                return None
        return None

    def _apply_definitive_template_matches(self, table: Table, report: Report) -> None:
        """Apply unambiguous names and synonyms from the runtime template first."""
        subject_names = {
            subject.subjectId: subject.subjectName
            for subject in report.subjectList
        }
        definitive_cell_ids: set[str] = set()
        definitive_subject_ids: set[str] = set()
        for row in table.rows:
            for cell in row:
                if not cell.text:
                    continue
                raw_text = cell.text.strip()
                normalized_text = self._normalize_subject_text(raw_text)
                if (
                    not raw_text
                    or self._has_trailing_heading_punctuation(raw_text)
                    or not normalized_text
                    or self._is_hard_non_subject_text(normalized_text)
                ):
                    continue
                matched_id = self._find_definitive_template_subject_id(report, raw_text)
                if not matched_id:
                    continue
                definitive_cell_ids.add(cell.cellId)
                definitive_subject_ids.add(matched_id)
                if matched_id == cell.subjectId:
                    continue
                previous_id = cell.subjectId
                cell.subjectId = matched_id
                if previous_id:
                    logger.warning(
                        "[definitive-match-corrected] text={!r}, old_subject={!r} (sid={}), "
                        "new_subject={!r} (sid={})",
                        cell.text,
                        subject_names.get(previous_id, previous_id),
                        previous_id,
                        subject_names.get(matched_id, matched_id),
                        matched_id,
                    )
                else:
                    logger.info(
                        "[definitive-match] text={!r} -> subject={!r} (sid={})",
                        cell.text,
                        subject_names.get(matched_id, matched_id),
                        matched_id,
                    )

        if not definitive_subject_ids:
            return

        for row in table.rows:
            for cell in row:
                if (
                    cell.cellId in definitive_cell_ids
                    or cell.subjectId not in definitive_subject_ids
                ):
                    continue
                released_id = cell.subjectId
                cell.subjectId = None
                logger.warning(
                    "[definitive-match-released] text={!r}, released_subject={!r} (sid={})",
                    cell.text,
                    subject_names.get(released_id, released_id),
                    released_id,
                )

    def _repair_malformed_subject_rowspans(self, table: Table, report: Report) -> None:
        """Split a merged subject cell only when the surrounding row shape proves it is malformed."""
        if self._current_strategy.language != FinancialStatementLanguage.ZH:
            return

        for row_index in range(len(table.rows) - 1):
            row = table.rows[row_index]
            following_row = table.rows[row_index + 1]
            if len(row) < 2 or len(following_row) != len(row) - 1:
                continue

            subject_cell = row[0]
            if subject_cell.rowSpan != 2 or not subject_cell.text.strip():
                continue
            if not all(
                _VALUE_CELL_RE.fullmatch(cell.text.strip())
                for cell in [*row[1:], *following_row]
                if cell.text.strip()
            ):
                continue

            split = self._find_rowspan_subject_split(subject_cell.text, report)
            if split is None:
                continue
            left_text, left_subject, right_text, right_subject = split

            original_cell_id = subject_cell.cellId
            subject_cell.text = left_text
            subject_cell.subjectId = left_subject.subjectId
            subject_cell.rowSpan = 1

            inserted_cell = subject_cell.model_copy(deep=True)
            inserted_cell.cellId = f"{original_cell_id}__rowspan_2"
            inserted_cell.text = right_text
            inserted_cell.subjectId = right_subject.subjectId
            following_row.insert(0, inserted_cell)

            logger.info(
                "[rowspan-subject-repaired] report={}, row={}, left={!r} (sid={}), "
                "right={!r} (sid={})",
                report.reportName,
                row_index,
                left_text,
                left_subject.subjectId,
                right_text,
                right_subject.subjectId,
            )

    def _find_rowspan_subject_split(
        self,
        source_text: str,
        report: Report,
    ) -> tuple[str, Subject, str, Subject] | None:
        raw_text = source_text.strip()
        if len(raw_text) < 12:
            return None

        variants: list[tuple[int, Subject, str]] = []
        for subject_index, subject in enumerate(report.subjectList):
            for candidate in (subject.subjectName, *subject.synonymSubject):
                normalized = self._deep_normalize(candidate.strip())
                if normalized:
                    variants.append((subject_index, subject, normalized))

        def best_match(segment: str) -> tuple[int, Subject, float, float] | None:
            normalized_segment = self._deep_normalize(segment)
            if len(normalized_segment) < 6:
                return None

            scores_by_subject: dict[str, tuple[int, Subject, float]] = {}
            for subject_index, subject, normalized_candidate in variants:
                score = SequenceMatcher(
                    None,
                    normalized_segment,
                    normalized_candidate,
                ).ratio()
                previous = scores_by_subject.get(subject.subjectId)
                if previous is None or score > previous[2]:
                    scores_by_subject[subject.subjectId] = (
                        subject_index,
                        subject,
                        score,
                    )
            ranked = sorted(
                scores_by_subject.values(),
                key=lambda item: item[2],
                reverse=True,
            )
            if not ranked:
                return None
            second_score = ranked[1][2] if len(ranked) > 1 else 0.0
            subject_index, subject, score = ranked[0]
            return subject_index, subject, score, score - second_score

        candidates: list[
            tuple[float, float, str, Subject, str, Subject]
        ] = []
        for split_index in range(4, len(raw_text) - 3):
            left_text = raw_text[:split_index].strip()
            right_text = raw_text[split_index:].strip()
            left_match = best_match(left_text)
            right_match = best_match(right_text)
            if left_match is None or right_match is None:
                continue

            left_index, left_subject, left_score, left_margin = left_match
            right_index, right_subject, right_score, right_margin = right_match
            if left_subject.subjectId == right_subject.subjectId or left_index >= right_index:
                continue
            if min(left_score, right_score) < _ROWSPAN_SEGMENT_MIN_SCORE:
                continue
            if min(left_margin, right_margin) < _ROWSPAN_SEGMENT_MIN_MARGIN:
                continue
            average_score = (left_score + right_score) / 2
            if average_score < _ROWSPAN_SPLIT_MIN_AVERAGE_SCORE:
                continue
            candidates.append(
                (
                    average_score,
                    min(left_score, right_score),
                    left_text,
                    left_subject,
                    right_text,
                    right_subject,
                )
            )

        if not candidates:
            return None
        _, _, left_text, left_subject, right_text, right_subject = max(
            candidates,
            key=lambda item: (item[0], item[1]),
        )
        return left_text, left_subject, right_text, right_subject

    def _is_soft_non_subject_text(self, text: str) -> bool:
        """Return True for texts that look like non-subject aggregates but may be valid template subjects."""
        return self._current_strategy.text_rules.is_soft_non_subject_text(text)

    def _is_hard_non_subject_text(self, text: str) -> bool:
        """Return True for headers, pure values, dates and other texts that should never match."""
        return self._current_strategy.text_rules.is_hard_non_subject_text(text)

    def _is_non_subject_text(self, text: str, template_subject_keys: Optional[set[str]] = None) -> bool:
        """Return True if the text should be skipped before subject matching.

        Hard non-subject texts are always skipped. Soft aggregate-like texts remain
        skipped unless they exactly exist in the caller-provided template subjects.
        """
        if self._is_hard_non_subject_text(text):
            return True
        if self._is_soft_non_subject_text(text):
            key = self._template_text_key(text)
            return key not in (template_subject_keys or set())
        return False

    @staticmethod
    def _table_matches_report(report: Report, table: Table) -> bool:
        """Match table to report with reportId precedence, then name+caliber.

        Why reportId first:
        - In many payloads, consolidated/parent reports share the same reportName
          (e.g. both are "资产负债表"). If table.reportName also has no caliber token,
          name-only matching can make one table match BOTH reports.
        - table.reportId/report.reportId is the strongest disambiguator and should
          be treated as authoritative when both are present.
        """
        # Strongest signal: explicit reportId.
        if table.reportId and report.reportId:
            return table.reportId == report.reportId

        # Fallback: semantic name + caliber compatibility.
        return match_report_name(report.reportName, report.reportCaliber, table.reportName)

    async def match_subjects(self, request: SubjectMatchRequest) -> SubjectMatchResponse:
        try:
            table_names = [t.reportName for t in request.tables]
            table_report_ids = [t.reportId for t in request.tables]
            report_names = [
                f"{r.reportName}(caliber={r.reportCaliber})" for r in request.reportList
            ]
            report_ids = [r.reportId for r in request.reportList]
            language = detect_subject_match_language(
                report_names=(r.reportName for r in request.reportList),
                table_names=(t.reportName for t in request.tables),
                subject_names=(s.subjectName for r in request.reportList for s in r.subjectList),
                cell_texts=(cell.text for table in request.tables for row in table.rows for cell in row),
            )
            self._current_strategy = get_subject_match_strategy(language)
            logger.info(
                "Subject matching started: fileId={}, reports={}, tables={}, language={}, "
                "reportNames={}, reportIds={}, tableNames={}, tableReportIds={}",
                request.fileId, len(request.reportList), len(request.tables), language.value,
                report_names, report_ids, table_names, table_report_ids,
            )

            # 打印 Java 传入的科目模板（简化版）
            for report in request.reportList:
                subject_names = [s.subjectName for s in report.subjectList]
                logger.info(
                    "[Java传入模板] 报表={}, 口径={}, 科目数={}, 科目列表={}",
                    report.reportName, report.reportCaliber, len(report.subjectList), subject_names
                )

            matched_tables = [table.model_copy(deep=True) for table in request.tables]

            # Snapshot Java pre-matched cell IDs per table BEFORE any processing.
            table_pre_matched: dict[int, set[str]] = {}
            for idx, table in enumerate(matched_tables):
                pre = set()
                for row in table.rows:
                    for cell in row:
                        if cell.subjectId:
                            pre.add(cell.cellId)
                table_pre_matched[idx] = pre

            # Build a combined subject-name lookup from ALL reports.
            all_subject_id_to_name: dict[str, str] = {}
            for report in request.reportList:
                for s in report.subjectList:
                    all_subject_id_to_name.setdefault(s.subjectId, s.subjectName)

            for report in request.reportList:
                matched_table_indices = [
                    idx for idx, table in enumerate(matched_tables)
                    if self._table_matches_report(report, table)
                ]

                if not matched_table_indices:
                    logger.warning(
                        "No table matched for report: {} (caliber={}), "
                        "available tables: {}",
                        report.reportName, report.reportCaliber, table_names,
                    )
                    continue

                for table_idx in matched_table_indices:
                    await self._match_subjects_for_table(matched_tables[table_idx], report)

            logger.info("Subject matching completed: fileId={}", request.fileId)
            return SubjectMatchResponse(fileId=request.fileId, tables=matched_tables)

        except Exception as e:
            logger.opt(exception=True).error("Subject matching failed: {}", e)
            raise

    @staticmethod
    def _detect_table_statement_type(report_name: str) -> Optional[str]:
        """Detect statement type from report name for cross-table guardrails."""
        if not report_name:
            return None
        rn = traditional_to_simplified(report_name)
        if "现金流量表" in rn:
            return "cashflow"
        if "利润表" in rn:
            return "income"
        if "资产负债表" in rn:
            return "balance"
        return None

    @staticmethod
    def _is_indirect_method_item(text: str) -> bool:
        """Return True if the text is a cash flow supplementary (indirect method) reconciliation item."""
        text_s = traditional_to_simplified(text)
        return any(kw in text_s for kw in _INDIRECT_METHOD_KEYWORDS)

    def _has_trailing_heading_punctuation(self, text: str) -> bool:
        return self._current_strategy.text_rules.has_trailing_heading_punctuation(text)

    def _looks_like_section_heading(self, text: str) -> bool:
        return self._current_strategy.text_rules.looks_like_section_heading(text)

    async def _match_subjects_for_table(self, table: Table, report: Report) -> None:
        """Match subjects for a single table against the given report template."""
        caliber = "consolidated" if report.reportCaliber == 2 else "parent"
        context = f"report={report.reportName}, caliber={caliber}"
        matcher = self._get_llm_matcher()
        stmt_type = self._detect_table_statement_type(report.reportName)

        text_to_first_cells: dict[str, list] = {}
        source_family_kept_text: dict[str, str] = {}
        template_subject_keys = self._build_template_subject_keys(report)
        self._repair_malformed_subject_rowspans(table, report)
        self._apply_definitive_template_matches(table, report)

        # Detect column roles (subject / amount / other) for parallel table support
        column_roles = self._detect_column_roles(table)
        subject_col_indices = {ci for ci, role in column_roles.items() if role == "subject"}
        has_column_roles = bool(subject_col_indices)

        if has_column_roles:
            logger.info(
                "[column-roles] Active for report={}: subject_columns={}",
                report.reportName, sorted(subject_col_indices),
            )
        else:
            logger.info(
                "[column-roles] Not detected for report={}, using legacy full-scan",
                report.reportName,
            )

        input_subject_cells: list[dict[str, Any]] = []
        for row in table.rows:
            if not row:
                continue
            for col_idx, cell in enumerate(row):
                if not cell.text or cell.subjectId:
                    continue
                raw_txt = cell.text.strip()
                if not raw_txt:
                    continue
                _diag = "负债" in raw_txt or "权益" in raw_txt or "总计" in raw_txt
                if _VALUE_CELL_RE.match(raw_txt):
                    if _diag:
                        logger.info("[filter-diag] VALUE_CELL_RE matched: {!r}", raw_txt)
                    continue
                # When column roles are detected, skip cells not in subject columns.
                # FIXME: 列角色检测可能不准确（特别是双排表格的第二组列），暂时禁用过滤功能
                # 保留检测逻辑和日志输出，方便后续调试和改进
                # if has_column_roles and not self._is_cell_in_subject_column(cell, col_idx, column_roles):
                #     continue
                # 原始文本以冒号结尾的是大类标题行（如"流动资产："），在normalize前过滤
                if self._has_trailing_heading_punctuation(raw_txt):
                    if _diag:
                        logger.info("[filter-diag] trailing_heading_punctuation: {!r}", raw_txt)
                    else:
                        logger.debug("Skipping category header cell (trailing colon): {!r}", raw_txt)
                    continue
                if self._looks_like_section_heading(raw_txt):
                    if _diag:
                        logger.info("[filter-diag] section_heading(raw): {!r}", raw_txt)
                    else:
                        logger.debug("Skipping section heading cell: {!r}", raw_txt)
                    continue
                txt = self._normalize_subject_text(raw_txt)
                if not txt:
                    if _diag:
                        logger.info("[filter-diag] normalize_subject_text returned empty: raw={!r}", raw_txt)
                    continue
                if self._looks_like_section_heading(txt):
                    if _diag:
                        logger.info("[filter-diag] section_heading(normalized): {!r} raw={!r}", txt, raw_txt)
                    else:
                        logger.debug("Skipping normalized section heading cell: {!r} (raw: {!r})", txt, raw_txt)
                    continue
                if self._is_non_subject_text(txt, template_subject_keys):
                    if _diag:
                        logger.info("[filter-diag] non_subject_text: {!r} raw={!r}", txt, raw_txt)
                    else:
                        logger.debug("Skipping non-subject cell: {!r} (raw: {!r})", txt, raw_txt)
                    continue
                if _diag:
                    logger.info("[filter-diag] PASSED all filters: {!r} raw={!r}", txt, raw_txt)

                input_subject_cells.append(
                    {
                        "cell_id": cell.cellId,
                        "raw_text": raw_txt,
                        "normalized_text": txt,
                        "existing_subject_id": cell.subjectId,
                    }
                )

                # ---------- 单表前置家族去重（关键） ----------
                deep_txt = self._deep_normalize(txt)
                family_key = deep_txt
                for suffix in self._current_strategy.text_rules.family_suffixes_dedup:
                    if family_key.endswith(suffix) and len(family_key) > len(suffix):
                        family_key = family_key[: -len(suffix)]
                        break

                kept_txt = source_family_kept_text.get(family_key)

                if kept_txt is None:
                    source_family_kept_text[family_key] = txt
                elif kept_txt != txt:
                    kept_is_net = any(kept_txt.endswith(s) for s in self._current_strategy.text_rules.net_type_suffixes)
                    cur_is_net = any(txt.endswith(s) for s in self._current_strategy.text_rules.net_type_suffixes)

                    # 优先保留非净额项（应收账款 > 应收账款净额）
                    # 再优先保留更短文本（其他长期资产 > 其他长期资产净额）
                    replace_kept = False
                    if kept_is_net and not cur_is_net:
                        replace_kept = True
                    elif kept_is_net == cur_is_net and len(txt) < len(kept_txt):
                        replace_kept = True

                    if replace_kept:
                        logger.info(
                            "[v3] Pre-collect family replace: family={!r}, old={!r}, new={!r}",
                            family_key, kept_txt, txt,
                        )
                        # 被替换的旧家族成员：强制清空其可能存在的Java历史匹配
                        if kept_txt in text_to_first_cells:
                            for old_cell in text_to_first_cells[kept_txt]:
                                old_cell.subjectId = None
                            text_to_first_cells.pop(kept_txt, None)
                        source_family_kept_text[family_key] = txt
                    else:
                        logger.info(
                            "[v3] Pre-collect family skip: text={!r}, kept={!r}, family={!r}",
                            txt, kept_txt, family_key,
                        )
                        # 被跳过的衍生科目：强制清空匹配，防止带着Java的错误匹配逃逸
                        cell.subjectId = None
                        continue

                # 同一标准化文本在单表内应收录所有相关cell，不能直接continue遗漏
                if txt in text_to_first_cells:
                    logger.info(
                        "[v3] Pre-collect duplicate text append: {!r} (cellId={})",
                        txt, cell.cellId,
                    )
                    text_to_first_cells[txt].append(cell)
                    continue

                text_to_first_cells[txt] = [cell]

        logger.info(
            "[input-subject-cells] total={} cells={}",
            len(input_subject_cells),
            input_subject_cells[:300],
        )

        if not text_to_first_cells:
            return

        logger.info(
            "[texts-to-match] total_unique={} texts={}",
            len(text_to_first_cells),
            sorted(text_to_first_cells.keys()),
        )

        # Prefer leaf subjects (no summaryRelation) over aggregates in matching order.
        # This prevents "应收账款" from matching the aggregate instead of the leaf.
        subjects_ordered = sorted(
            report.subjectList,
            key=lambda s: (bool(s.summaryRelation), s.subjectName),
        )

        all_candidates: list[dict[str, Any]] = []
        subject_id_to_name: dict[str, str] = {}

        for s in subjects_ordered:
            all_candidates.append({"id": s.subjectId, "name": s.subjectName})
            subject_id_to_name[s.subjectId] = s.subjectName

            for syn in s.synonymSubject:
                all_candidates.append({"id": s.subjectId, "name": syn, "source": "synonym"})

        # ── Diagnostic: show template structure ──
        agg_count = sum(1 for s in report.subjectList if s.summaryRelation)
        logger.info(
            "[v3] Starting subject matching: unique_texts={}, template_subjects={}, "
            "candidates={}, aggregates_with_summaryRelation={}",
            len(text_to_first_cells), len(report.subjectList), len(all_candidates),
            agg_count,
        )

        # Pre-compute aggregate subject IDs so the matching closure can filter them
        # out from LLM candidates when leaf alternatives exist.
        aggregate_ids: set[str] = {s.subjectId for s in subjects_ordered if s.summaryRelation}

        # ── Collect Java pre-matched info ──
        pre_matched_sids: set[str] = set()
        pre_matched_cell_ids: set[str] = set()
        pre_matched_texts: dict[str, str] = {}  # cellId → text
        java_pre_matched_source_texts: set[str] = set()
        for row in table.rows:
            for cell in row:
                if cell.subjectId:
                    pre_matched_sids.add(cell.subjectId)
                    pre_matched_cell_ids.add(cell.cellId)
                    pre_matched_texts[cell.cellId] = cell.text.strip()
                    normalized_pre_text = self._normalize_subject_text(cell.text.strip())
                    if normalized_pre_text:
                        java_pre_matched_source_texts.add(normalized_pre_text)

        logger.info(
            "[java-pre-summary] pre_cells={}, pre_sids={}",
            len(pre_matched_cell_ids), len(pre_matched_sids),
        )

        if pre_matched_sids:
            java_pairs = []
            for row in table.rows:
                for cell in row:
                    if cell.subjectId:
                        java_pairs.append(
                            {
                                "template_sid": cell.subjectId,
                                "template_name": subject_id_to_name.get(cell.subjectId, cell.subjectId),
                                "source_text": cell.text.strip(),
                                "cell_id": cell.cellId,
                            }
                        )

            logger.info("[java-pre] pairs={}", java_pairs[:80])
        else:
            total_cells = sum(len(row) for row in table.rows)
            logger.info(
                "[java-pre] 调用方未传入任何预匹配的 subjectId, "
                "Python 将全量匹配 | report={}, total_cells={}, "
                "hint=如果调用方(Java)应该预匹配, 请检查请求体中 rows[][].subjectId 是否被正确填充",
                report.reportName, total_cells,
            )

        # ── Build blocked set ──
        # NOTE: summaryRelation-based blocking is disabled because Java currently
        # sends summaryRelation=[] for all subjects (aggregates_with_summaryRelation=0).
        # If Java starts populating summaryRelation in the future, re-enable the
        # block below.
        subject_by_id_early: dict[str, Subject] = {
            s.subjectId: s for s in subjects_ordered
        }
        blocked_leaf_sids: set[str] = set()

        # ── Build blocked set: NAME-BASED fallback ──
        # When summaryRelation is empty, detect siblings by name patterns.
        # If Java pre-matched a cell to subject "X", block subject "X净额",
        # "X净值", "X准备" etc. (common accounting suffixes that form families).
        _FAMILY_SUFFIXES = ("净额", "净值", "原值", "摊销", "准备", "减值准备", "合计", "小计", "总计")
        pre_matched_subject_names: dict[str, str] = {}  # sid → deep_normalized name
        for sid in pre_matched_sids:
            subj = subject_by_id_early.get(sid)
            if subj and not subj.summaryRelation:
                pre_matched_subject_names[sid] = self._deep_normalize(subj.subjectName.strip())

        for s in subjects_ordered:
            if s.subjectId in pre_matched_sids or s.subjectId in blocked_leaf_sids:
                continue
            if s.summaryRelation:
                continue
            s_norm = self._deep_normalize(s.subjectName.strip())
            if not s_norm:
                continue
            for pm_sid, pm_norm in pre_matched_subject_names.items():
                if not pm_norm:
                    continue
                is_family = False
                # "应收账款" pre-matched → block "应收账款净额"
                if s_norm.startswith(pm_norm) and s_norm != pm_norm:
                    suffix = s_norm[len(pm_norm):]
                    if suffix in _FAMILY_SUFFIXES or (suffix and suffix[0] in ("及", "和", "与")):
                        is_family = True
                # "应收账款净额" pre-matched → block "应收账款"
                if pm_norm.startswith(s_norm) and pm_norm != s_norm:
                    suffix = pm_norm[len(s_norm):]
                    if suffix in _FAMILY_SUFFIXES or (suffix and suffix[0] in ("及", "和", "与")):
                        is_family = True
                if is_family:
                    blocked_leaf_sids.add(s.subjectId)
                    logger.info(
                        "[v3] Name-based block: {!r} blocked by pre-matched {!r}",
                        s.subjectName, subject_id_to_name.get(pm_sid, pm_sid),
                    )
                    break

        if blocked_leaf_sids:
            logger.info(
                "[v3] Total blocked leaf sids: {} (from {} pre-matched cells)",
                len(blocked_leaf_sids), len(pre_matched_sids),
            )

        # Batch LLM term standardization (one call for all texts, before per-item matching)
        llm_norm_map: dict[str, str] = {}
        if self.use_llm:
            try:
                llm_norm_map = await self._llm_normalize_batch(
                    list(text_to_first_cells.keys()), report.reportName,
                )
            except Exception as e:
                logger.warning("Batch LLM normalization failed (non-fatal): {}", str(e)[:200])

        semaphore = asyncio.Semaphore(self.llm_max_concurrency)
        
        # 声明全局锁定机制：记录 template_id -> (score, source_text, cells, trusted)
        template_assignment_lock = asyncio.Lock()
        template_best_match: dict[str, tuple[float, str, list, bool]] = {}
        trusted_rule_matches: dict[str, str] = {}

        async def process_single_text(text: str, first_cells: list) -> None:
            async with semaphore:
                try:
                    # 提取内部的竞争分配逻辑
                    async def _try_assign(
                        target_id: str,
                        score: float,
                        match_type: str,
                        matched_name: str,
                        *,
                        trusted: bool = False,
                    ) -> bool:
                        async with template_assignment_lock:
                            actual_template_name = subject_id_to_name.get(target_id, target_id)

                            has_preexisting_owner = any(
                                cell.subjectId == target_id and cell.cellId in pre_matched_cell_ids
                                for row in table.rows
                                for cell in row
                            )

                            if has_preexisting_owner:
                                logger.info(
                                    "[assign-blocked-preexisting] type={} text={!r}, "
                                    "template_subject={!r} (sid={})",
                                    match_type,
                                    text,
                                    actual_template_name,
                                    target_id,
                                )
                                return False

                            if target_id not in template_best_match:
                                template_best_match[target_id] = (
                                    score,
                                    text,
                                    first_cells,
                                    trusted,
                                )
                                for cell in first_cells:
                                    cell.subjectId = target_id
                                if trusted:
                                    trusted_rule_matches[text] = target_id
                                else:
                                    trusted_rule_matches.pop(text, None)

                                logger.info(
                                    "[assign] type={} text={!r} -> template_subject={!r} (sid={}) "
                                    "hit_name={!r} score={:.3f} cells={}",
                                    match_type,
                                    text,
                                    actual_template_name,
                                    target_id,
                                    matched_name,
                                    score,
                                    [c.cellId for c in first_cells],
                                )
                                return True
                            else:
                                prev_score, prev_text, prev_cells, _ = template_best_match[target_id]
                                is_net = any(text.endswith(s) for s in self._current_strategy.text_rules.family_suffixes_dedup)
                                prev_is_net = any(prev_text.endswith(s) for s in self._current_strategy.text_rules.family_suffixes_dedup)

                                should_override = score > prev_score or (
                                    score == prev_score and prev_is_net and not is_net
                                )

                                if should_override:
                                    for cell in prev_cells:
                                        cell.subjectId = None
                                    trusted_rule_matches.pop(prev_text, None)
                                    template_best_match[target_id] = (
                                        score,
                                        text,
                                        first_cells,
                                        trusted,
                                    )
                                    for cell in first_cells:
                                        cell.subjectId = target_id
                                    if trusted:
                                        trusted_rule_matches[text] = target_id
                                    else:
                                        trusted_rule_matches.pop(text, None)

                                    logger.warning(
                                        "[assign-override] type={} template_subject={!r} (sid={}) "
                                        "new={!r} score={:.3f} old={!r} score={:.3f} "
                                        "new_cells={} old_cells={}",
                                        match_type,
                                        actual_template_name,
                                        target_id,
                                        text,
                                        score,
                                        prev_text,
                                        prev_score,
                                        [c.cellId for c in first_cells],
                                        [c.cellId for c in prev_cells],
                                    )
                                    return True
                                else:
                                    logger.warning(
                                        "[assign-blocked] type={} template_subject={!r} (sid={}) "
                                        "keep={!r} score={:.3f}, blocked={!r} score={:.3f}",
                                        match_type,
                                        actual_template_name,
                                        target_id,
                                        prev_text,
                                        prev_score,
                                        text,
                                        score,
                                    )
                                    return False

                    # Rule matches are trusted only when the strongest matching
                    # tier resolves to exactly one template subject.  Normalized
                    # names and synonyms can legitimately collide in a template.
                    static_match_ambiguous = False
                    rule_groups: list[tuple[list[tuple[Subject, str]], float, str]] = [
                        (
                            [
                                (s, s.subjectName)
                                for s in subjects_ordered
                                if s.subjectName.strip() == text
                            ],
                            1.0,
                            "Rule (exact)",
                        ),
                        (
                            [
                                (s, s.subjectName)
                                for s in subjects_ordered
                                if self._normalize_subject_text(s.subjectName.strip()) == text
                            ],
                            0.95,
                            "Rule (normalized)",
                        ),
                        (
                            [
                                (s, synonym)
                                for s in subjects_ordered
                                for synonym in s.synonymSubject
                                if synonym.strip() == text
                            ],
                            0.9,
                            "Rule (synonym)",
                        ),
                        (
                            [
                                (s, synonym)
                                for s in subjects_ordered
                                for synonym in s.synonymSubject
                                if self._normalize_subject_text(synonym.strip()) == text
                            ],
                            0.85,
                            "Rule (synonym norm)",
                        ),
                    ]

                    for matches, score, match_type in rule_groups:
                        if not matches:
                            continue
                        matches_by_id = {
                            subject.subjectId: matched_name
                            for subject, matched_name in matches
                        }
                        if len(matches_by_id) == 1:
                            target_id, matched_name = next(iter(matches_by_id.items()))
                            await _try_assign(
                                target_id,
                                score,
                                match_type,
                                matched_name,
                                trusted=True,
                            )
                            return
                        logger.info(
                            "[rule-match-ambiguous] type={} text={!r}, subject_ids={}",
                            match_type,
                            text,
                            sorted(matches_by_id),
                        )
                        static_match_ambiguous = True
                        break

                    # Step 2: Deep-normalized exact match.  A collision at a
                    # stronger tier must be resolved by context, not by a weaker
                    # normalization rule.
                    deep_text = self._deep_normalize(text)
                    if not static_match_ambiguous:
                        deep_groups: list[tuple[list[tuple[Subject, str]], float, str]] = [
                            (
                                [
                                    (s, s.subjectName)
                                    for s in subjects_ordered
                                    if deep_text
                                    and self._deep_normalize(s.subjectName.strip()) == deep_text
                                ],
                                0.8,
                                "Rule (deep)",
                            ),
                            (
                                [
                                    (s, synonym)
                                    for s in subjects_ordered
                                    for synonym in s.synonymSubject
                                    if deep_text
                                    and self._deep_normalize(synonym.strip()) == deep_text
                                ],
                                0.75,
                                "Rule (deep syn)",
                            ),
                        ]
                        for matches, score, match_type in deep_groups:
                            if not matches:
                                continue
                            matches_by_id = {
                                subject.subjectId: matched_name
                                for subject, matched_name in matches
                            }
                            if len(matches_by_id) == 1:
                                target_id, matched_name = next(iter(matches_by_id.items()))
                                await _try_assign(
                                    target_id,
                                    score,
                                    match_type,
                                    matched_name,
                                    trusted=True,
                                )
                                return
                            logger.info(
                                "[rule-match-ambiguous] type={} text={!r}, subject_ids={}",
                                match_type,
                                text,
                                sorted(matches_by_id),
                            )
                            static_match_ambiguous = True
                            break

                    # Step 2.25: deterministic accounting equivalence.
                    # Exact names above still win when a future template adds a
                    # more specific subject.
                    deterministic_key = self._deterministic_match_key(text)
                    if deterministic_key and not static_match_ambiguous:
                        deterministic_matches: dict[str, str] = {}
                        for s in subjects_ordered:
                            for candidate in (s.subjectName, *s.synonymSubject):
                                if self._deterministic_match_key(candidate) == deterministic_key:
                                    deterministic_matches.setdefault(s.subjectId, candidate)
                                    break
                        if len(deterministic_matches) == 1:
                            target_id, matched_name = next(iter(deterministic_matches.items()))
                            await _try_assign(
                                target_id,
                                0.72,
                                "Rule (accounting equivalence)",
                                matched_name,
                                trusted=True,
                            )
                            return
                        if len(deterministic_matches) > 1:
                            logger.info(
                                "[deterministic-match-ambiguous] text={!r}, key={}, subject_ids={}",
                                text,
                                deterministic_key,
                                sorted(deterministic_matches),
                            )

                    # Step 2.5: LLM-normalized matching
                    if text in llm_norm_map and not static_match_ambiguous:
                        llm_std = llm_norm_map[text]
                        deep_llm_std = self._deep_normalize(llm_std)
                        llm_norm_groups: list[tuple[list[tuple[Subject, str]], float, str]] = [
                            (
                                [
                                    (s, s.subjectName)
                                    for s in subjects_ordered
                                    if deep_llm_std
                                    and self._deep_normalize(s.subjectName.strip()) == deep_llm_std
                                ],
                                0.7,
                                "Rule (LLM-norm)",
                            ),
                            (
                                [
                                    (s, synonym)
                                    for s in subjects_ordered
                                    for synonym in s.synonymSubject
                                    if deep_llm_std
                                    and self._deep_normalize(synonym.strip()) == deep_llm_std
                                ],
                                0.65,
                                "Rule (LLM-norm syn)",
                            ),
                        ]
                        for matches, score, match_type in llm_norm_groups:
                            if not matches:
                                continue
                            matches_by_id = {
                                subject.subjectId: matched_name
                                for subject, matched_name in matches
                            }
                            if len(matches_by_id) == 1:
                                target_id, matched_name = next(iter(matches_by_id.items()))
                                await _try_assign(
                                    target_id,
                                    score,
                                    match_type,
                                    matched_name,
                                )
                                return
                            logger.info(
                                "[llm-normalized-match-ambiguous] text={!r}, subject_ids={}",
                                text,
                                sorted(matches_by_id),
                            )
                            break

                    # Step 3: semantic_match
                    sem_matches: list[Subject] = []
                    for s in subjects_ordered:
                        if semantic_match(text, s.subjectName.strip()):
                            sem_matches.append(s)
                        elif any(semantic_match(text, syn.strip()) for syn in s.synonymSubject):
                            sem_matches.append(s)
                    if sem_matches:
                        leaf_sem = [s for s in sem_matches if not s.summaryRelation]
                        if leaf_sem:
                            sem_matches = leaf_sem
                    if len(sem_matches) == 1:
                        await _try_assign(sem_matches[0].subjectId, 0.6, "Semantic", sem_matches[0].subjectName)
                        return
                    elif len(sem_matches) > 1:
                        logger.debug("Semantic match ambiguous ({} hits) for {!r}, falling through to LLM", len(sem_matches), text)

                    if not (self.use_llm and matcher):
                        return

                    # (此处保留原来的诸多 Gate 过滤逻辑，如 _AGGREGATE_SUFFIXES, _NET_TYPE_SUFFIXES 等...)
                    matched_suffix = next((s for s in self._current_strategy.text_rules.aggregate_suffixes if text.endswith(s)), None)
                    if matched_suffix:
                        if not any(c["name"].endswith(matched_suffix) for c in all_candidates):
                            return

                    source_is_net_type = any(text.endswith(s) for s in self._current_strategy.text_rules.net_type_suffixes)
                    candidates_for_llm = all_candidates if source_is_net_type else [
                        c for c in all_candidates if not any(c["name"].endswith(s) for s in self._current_strategy.text_rules.net_type_suffixes)
                    ]

                    source_is_summary = any(text.endswith(s) for s in self._current_strategy.text_rules.summary_suffixes)
                    summary_candidates = [c for c in candidates_for_llm if any(c["name"].endswith(s) for s in self._current_strategy.text_rules.summary_suffixes)]
                    non_summary_candidates = [c for c in candidates_for_llm if not any(c["name"].endswith(s) for s in self._current_strategy.text_rules.summary_suffixes)]
                    if source_is_summary:
                        if summary_candidates:
                            candidates_for_llm = summary_candidates
                    else:
                        candidates_for_llm = non_summary_candidates or candidates_for_llm

                    source_is_cashflow_section = any(kw in text for kw in self._current_strategy.text_rules.cashflow_section_keywords)
                    if not source_is_cashflow_section:
                        non_section_candidates = [c for c in candidates_for_llm if not any(kw in c["name"] for kw in self._current_strategy.text_rules.cashflow_section_keywords)]
                        if non_section_candidates:
                            candidates_for_llm = non_section_candidates

                    source_is_eps = any(kw in text for kw in self._current_strategy.text_rules.eps_keywords)
                    source_is_profit = any(kw in text for kw in self._current_strategy.text_rules.profit_keywords)
                    if source_is_eps:
                        pre_eps = [c for c in candidates_for_llm if not any(kw in c["name"] for kw in self._current_strategy.text_rules.profit_keywords)]
                        if pre_eps:
                            candidates_for_llm = pre_eps
                    elif source_is_profit:
                        pre_profit = [c for c in candidates_for_llm if not any(kw in c["name"] for kw in self._current_strategy.text_rules.eps_keywords)]
                        if pre_profit:
                            candidates_for_llm = pre_profit

                    leaf_cands = [c for c in candidates_for_llm if c["id"] not in aggregate_ids]
                    if leaf_cands:
                        candidates_for_llm = leaf_cands

                    filtered_candidates, top_score = self._filter_candidates_by_similarity(text, candidates_for_llm, top_k=60)
                    if top_score < _MIN_SIMILARITY_FOR_LLM:
                        return

                    seen_names: set[str] = set()
                    deduped = []
                    for c in filtered_candidates:
                        if c["name"] not in seen_names:
                            seen_names.add(c["name"])
                            deduped.append(c)
                    filtered_candidates = deduped[:50]

                    best_id = await matcher.select_best_match(
                        text,
                        filtered_candidates,
                        context,
                        language=self._current_strategy.language,
                    )

                    if best_id:
                        matched_name = subject_id_to_name.get(best_id, "")
                        if matched_name:
                            post_sim = self._post_llm_similarity(text, matched_name)
                            if post_sim < 0.30:
                                best_id = None
                                
                    if best_id:
                        matched_name = subject_id_to_name.get(best_id, "unknown")
                        # LLM match分数设为 0.5，比规则匹配更低，容易被规则替换
                        await _try_assign(best_id, 0.5, "LLM", matched_name)

                except Exception as e:
                    logger.opt(exception=True).error("Error processing subject {!r}: {}", text, e)

        # ── Pre-match: source-text family dedup ──
        # In all-at-once mode, "应收账款" and "应收账款净额" would both be matched
        # independently, causing "多个加总科目". Proactively remove derivative
        # source texts BEFORE matching so they never get a subjectId assigned.
        source_texts = list(text_to_first_cells.keys())
        texts_to_skip: set[str] = set()
        for i, txt_a in enumerate(source_texts):
            if txt_a in texts_to_skip:
                continue
            deep_a = self._deep_normalize(txt_a)
            if not deep_a:
                continue
            for j in range(i + 1, len(source_texts)):
                txt_b = source_texts[j]
                if txt_b in texts_to_skip:
                    continue
                deep_b = self._deep_normalize(txt_b)
                if not deep_b:
                    continue

                family_type = self._classify_family_pair(deep_a, deep_b)
                if not family_type:
                    continue

                if family_type == "identical":
                    skip_target = txt_b if len(txt_a) <= len(txt_b) else txt_a
                else:
                    # suffix / compound → skip the longer (derivative) text
                    skip_target = txt_b if len(deep_a) <= len(deep_b) else txt_a

                texts_to_skip.add(skip_target)
                logger.info(
                    "[v3] Pre-match family skip ({}): {!r} removed (kept {!r})",
                    family_type,
                    skip_target, txt_a if skip_target == txt_b else txt_b,
                )

        texts_to_match = {
            txt: cells for txt, cells in text_to_first_cells.items()
            if txt not in texts_to_skip
        }

        await asyncio.gather(
            *(process_single_text(txt, cells) for txt, cells in texts_to_match.items())
        )

        # Preventive sibling/family block: clear any Python matches whose
        # subjectId is a sibling/child of a Java pre-matched cell.
        if blocked_leaf_sids:
            for txt, cells in text_to_first_cells.items():
                sid = cells[0].subjectId if cells else None
                if sid and sid in blocked_leaf_sids:
                    for cell in cells:
                        cell.subjectId = None
                    logger.warning(
                        "Preventive family block: cleared {!r} -> {!r} "
                        "(sibling/parent already pre-matched by Java)",
                        txt, subject_id_to_name.get(sid, sid),
                    )

        # Post-match: resolve aggregate matches to leaf subjects when possible
        self._resolve_aggregate_to_leaf(text_to_first_cells, report)

        # Conflict resolution: multiple source texts mapped to the same template subject
        subject_id_to_texts: dict[str, list[str]] = defaultdict(list)
        for txt, cells in text_to_first_cells.items():
            sid = cells[0].subjectId if cells else None
            if sid:
                subject_id_to_texts[sid].append(txt)

        subject_by_id_cr: dict[str, Subject] = {s.subjectId: s for s in report.subjectList}

        def _is_derivative_text(src_text: str) -> bool:
            t = (src_text or "").strip()
            return any(t.endswith(suf) for suf in self._current_strategy.text_rules.family_suffixes_dedup)

        def _conflict_score(txt: str, sid: str) -> tuple:
            template_name = subject_id_to_name.get(sid, "")
            normalized_template = self._normalize_subject_text(template_name.strip())
            deep_template = self._deep_normalize(template_name.strip())
            deep_txt = self._deep_normalize(txt.strip())
            sim = SequenceMatcher(None, txt, template_name).ratio()
            has_java = any(
                c.cellId in pre_matched_cell_ids
                for c in text_to_first_cells.get(txt, [])
            )

            # 优先级：
            # 1) deep 完全一致
            # 2) 原文/normalize 一致
            # 3) 非衍生项优先（净额/原值/摊销等降权）
            # 4) 相似度
            # 5) Java 预匹配兜底
            # 6) 文本越短越优先（通常主项）
            return (
                deep_txt == deep_template,
                txt == template_name or txt == normalized_template,
                not _is_derivative_text(txt),
                sim,
                has_java,
                -len(deep_txt or txt),
            )

        for sid, texts in subject_id_to_texts.items():
            if len(texts) <= 1:
                continue

            # 聚合科目优先尝试分摊到叶子
            matched_subj = subject_by_id_cr.get(sid)
            if matched_subj and matched_subj.summaryRelation:
                spread_ok = self._try_spread_to_leaves(
                    texts, text_to_first_cells, matched_subj, subject_by_id_cr
                )
                if spread_ok:
                    logger.info(
                        "Conflict spread to leaves: aggregate={!r}, texts={!r}",
                        matched_subj.subjectName, texts,
                    )
                    continue

            # 严格一对一：同一模板subjectId只保留一个来源文本
            texts_sorted = sorted(
                texts,
                key=lambda t: _conflict_score(t, sid),
                reverse=True,
            )
            keep_text = texts_sorted[0]
            keep_score = _conflict_score(keep_text, sid)
            template_name = subject_id_to_name.get(sid, "")

            for txt in texts_sorted[1:]:
                lose_score = _conflict_score(txt, sid)
                for cell in text_to_first_cells[txt]:
                    cell.subjectId = None
                
                # 简洁日志：显示竞争关系和决策理由
                reason = ""
                for idx, (k, l) in enumerate(zip(keep_score, lose_score)):
                    if k != l:
                        score_names = ["deep_exact", "raw_exact", "similarity", "non_deriv", "java", "len"]
                        reason = f"{score_names[idx]}: {k}>{l}"
                        break
                
                logger.warning(
                    "[Conflict] template={!r}: KEPT={!r} vs CLEARED={!r} | reason: {}",
                    template_name, keep_text, txt, reason or "stable_sort"
                )

        # Second pass: after conflict resolution may have freed up some aggregate slots,
        # give leaf resolution another chance (e.g. "应收账款净额" now uncontested on aggregate).
        self._resolve_aggregate_to_leaf(text_to_first_cells, report)

        # Re-apply preventive block after aggregate-to-leaf resolution and
        # conflict spreading, which may have introduced blocked leaf sids.
        if blocked_leaf_sids:
            for txt, cells in text_to_first_cells.items():
                sid = cells[0].subjectId if cells else None
                if sid and sid in blocked_leaf_sids:
                    for cell in cells:
                        cell.subjectId = None
                    logger.warning(
                        "Preventive family block (post-resolution): cleared {!r} -> {!r}",
                        txt, subject_id_to_name.get(sid, sid),
                    )

        # Post-match guardrail: cross-statement type rejection
        if stmt_type == "cashflow":
            self._reject_income_statement_matches(text_to_first_cells, subject_id_to_name)

        # Post-match guardrail: LLM holistic review
        if self.use_llm and matcher:
            await self._llm_review_matches(
                text_to_first_cells,
                subject_id_to_name,
                report.reportName,
                stmt_type,
                trusted_rule_matches,
            )

        # Final deduplication: remove duplicate cells with same (subjectId, text)
        # This handles cases where the same cell text appears multiple times (e.g., merged cells parsed as separate cells)
        seen_subject_text_pairs: dict[tuple[str, str], Cell] = {}
        for row in table.rows:
            for cell in row:
                if not cell.subjectId:
                    continue
                key = (cell.subjectId, cell.text.strip())
                if key not in seen_subject_text_pairs:
                    seen_subject_text_pairs[key] = cell
                else:
                    # Duplicate found: clear subjectId
                    cell.subjectId = None
                    logger.info(
                        "[Dedup] Cleared duplicate cell: text={!r}, subjectId={}, cellId={} (kept cellId={})",
                        cell.text.strip(),
                        key[0],
                        cell.cellId,
                        seen_subject_text_pairs[key].cellId,
                    )

    def _whole_table_same_id_dedup(
        self,
        table,
        subject_id_to_name: dict[str, str],
        pre_matched_cell_ids: set[str],
    ) -> None:
        return self._postprocess_module().whole_table_same_id_dedup(table, subject_id_to_name, pre_matched_cell_ids)

    def _whole_table_name_family_dedup(
        self,
        table,
        subject_id_to_name: dict[str, str],
        pre_matched_cell_ids: set[str],
    ) -> None:
        return self._postprocess_module().whole_table_name_family_dedup(table, subject_id_to_name, pre_matched_cell_ids)

    def _whole_table_source_family_dedup(
        self,
        table: Table,
        subject_id_to_name: dict[str, str],
        pre_matched_cell_ids: set[str],
    ) -> None:
        return self._postprocess_module().whole_table_source_family_dedup(table, subject_id_to_name, pre_matched_cell_ids)

    def _enforce_one_to_one(
        self,
        table: Table,
        subject_id_to_name: dict[str, str],
        pre_matched_cell_ids: set[str],
    ) -> None:
        return self._postprocess_module().enforce_one_to_one(table, subject_id_to_name, pre_matched_cell_ids)

    def _apply_net_amount_exclusion(
        self,
        text_to_first_cells: dict[str, list],
        subject_id_to_name: dict[str, str],
    ) -> None:
        return self._postprocess_module().apply_net_amount_exclusion(text_to_first_cells, subject_id_to_name)

    _FAMILY_SUFFIXES_DEDUP: tuple[str, ...] = FAMILY_SUFFIXES_DEDUP

    def _classify_family_pair(self, deep_a: str, deep_b: str) -> Optional[str]:
        return self._current_strategy.text_rules.classify_family_pair(deep_a, deep_b)

    def _apply_name_family_dedup(
        self,
        text_to_first_cells: dict[str, list],
        subject_id_to_name: dict[str, str],
    ) -> None:
        return self._postprocess_module().apply_name_family_dedup(text_to_first_cells, subject_id_to_name)

    def _reject_income_statement_matches(
        self,
        text_to_first_cells: dict[str, list],
        subject_id_to_name: dict[str, str],
    ) -> None:
        return self._postprocess_module().reject_income_statement_matches(text_to_first_cells, subject_id_to_name)

    def _resolve_aggregate_to_leaf(
        self,
        text_to_first_cells: dict[str, list],
        report: Report,
    ) -> None:
        return self._postprocess_module().resolve_aggregate_to_leaf(text_to_first_cells, report)

    def _try_spread_to_leaves(
        self,
        texts: list[str],
        text_to_first_cells: dict[str, list],
        aggregate: Subject,
        subject_by_id: dict[str, Subject],
    ) -> bool:
        return self._postprocess_module().try_spread_to_leaves(texts, text_to_first_cells, aggregate, subject_by_id)

    async def _llm_review_matches(
        self,
        text_to_first_cells: dict[str, list],
        subject_id_to_name: dict[str, str],
        report_name: str,
        stmt_type: Optional[str],
        trusted_rule_matches: dict[str, str],
    ) -> None:
        return await _llm_helpers.llm_review_matches(
            text_to_first_cells,
            subject_id_to_name,
            report_name,
            stmt_type,
            self._get_llm_matcher(),
            self._current_strategy.language,
            trusted_rule_matches,
        )

    async def _llm_normalize_batch(
        self,
        texts: list[str],
        report_name: str,
    ) -> dict[str, str]:
        return await _llm_helpers.llm_normalize_batch(
            texts,
            report_name,
            self._get_llm_matcher(),
            self._current_strategy.language,
        )

    @staticmethod
    def _post_llm_similarity(source_text: str, matched_name: str) -> float:
        return _llm_helpers.post_llm_similarity(source_text, matched_name)

    def _filter_candidates_by_similarity(
        self,
        text: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 50,
    ) -> Tuple[List[Dict[str, Any]], float]:
        return _llm_helpers.filter_candidates_by_similarity(text, candidates, top_k)
