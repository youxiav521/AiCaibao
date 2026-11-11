import asyncio
import inspect
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, Optional
from uuid import uuid4

import json_repair
from langchain_core.messages import HumanMessage, SystemMessage
from openai import APIError

try:
    from .prompts import build_framework_messages, build_section_messages
    from .schemas import (
        Control,
        ControlType,
        Framework,
        NormalizedChunk,
        PolicyDocument,
        Requirement,
        RiskLevel,
        SectionExtractionResult,
        SourceLocation,
    )
except ImportError:  # pragma: no cover
    from prompts import build_framework_messages, build_section_messages
    from schemas import (
        Control,
        ControlType,
        Framework,
        NormalizedChunk,
        PolicyDocument,
        Requirement,
        RiskLevel,
        SectionExtractionResult,
        SourceLocation,
    )

from src.utils.model_factory import ModelFactory


@dataclass(slots=True)
class ExtractorConfig:
    """Runtime config for chunking, concurrency, and retry behavior."""
    enable_llm: bool = True
    model_id: str = "model_qwen35_35b"
    max_chars_per_section: int = 12000
    section_overlap_chunks: int = 1
    skip_heading_only_sections: bool = True
    merge_small_sections: bool = False
    min_section_chars: int = 500
    max_section_concurrency: int = 16
    auto_section_concurrency: bool = True
    stable_max_section_concurrency: int = 16
    max_retries: int = 3
    retry_delay_seconds: float = 2.0


class PolicyExtractor:
    """LLM-first extractor. Local code prepares section context and normalizes output."""

    def __init__(
        self,
        config: Optional[ExtractorConfig] = None,
        model: Any = None,
        requirement_callback: Optional[Callable[[Requirement], Any]] = None,
    ):
        """Initialize extractor and model client."""
        self.config = config or ExtractorConfig()
        self.model = model
        self.requirement_callback = requirement_callback
        if self.config.enable_llm and self.model is None:
            self.model = ModelFactory.get_model(key=self.config.model_id)

    def extract(self, document: PolicyDocument) -> Framework:
        """Sync wrapper around the async extraction pipeline."""
        return asyncio.run(self._aextract(document))

    async def _aextract(self, document: PolicyDocument) -> Framework:
        """Run section extraction first, then derive framework-level summary."""
        grouped_chunks = self._group_chunks(document.chunks)
        print(
            f"[policy_retractor] grouped {len(document.chunks)} chunks into {len(grouped_chunks)} section windows; "
            f"model_id={self.config.model_id}, enable_llm={self.config.enable_llm}, "
            f"max_section_concurrency={self.config.max_section_concurrency}",
            flush=True,
        )
        if self.config.enable_llm and self.model is not None:
            section_results, section_error_count = await self._extract_sections(document, grouped_chunks)
        else:
            section_results, section_error_count = [], 0

        framework_error = False
        framework_draft: dict
        if self.config.enable_llm and self.model is not None:
            try:
                framework_draft = await self._extract_framework(document, grouped_chunks, section_results)
            except Exception:
                framework_error = True
                framework_draft = {}
        else:
            framework_draft = {}

        has_error = framework_error or section_error_count > 0 or not document.chunks
        return self._build_framework(document, framework_draft, section_results, has_error=has_error)

    def _group_chunks(self, chunks: Iterable[NormalizedChunk]) -> list[list[NormalizedChunk]]:
        """Group chunks by heading context and split them into LLM windows."""
        raw_groups: list[list[NormalizedChunk]] = []
        current_group: list[NormalizedChunk] = []
        current_key: Optional[str] = None

        for chunk in chunks:
            section_key = self._chunk_section_key(chunk)
            # Keep contiguous chunks with the same heading in one raw group.
            if current_group and section_key != current_key:
                raw_groups.append(current_group)
                current_group = []
            current_group.append(chunk)
            current_key = section_key

        if current_group:
            raw_groups.append(current_group)

        groups: list[list[NormalizedChunk]] = []
        for group in raw_groups:
            if self._should_skip_group(group):
                continue
            groups.extend(self._split_group_into_windows(group))
        if self.config.merge_small_sections:
            groups = self._merge_small_groups(groups)
        return groups

    def _chunk_section_key(self, chunk: NormalizedChunk) -> str:
        """Build grouping key and allow table captions to create subsection splits."""
        base_key = " / ".join(chunk.heading_path) if chunk.heading_path else (chunk.title or "ROOT")
        if chunk.chunk_type != "table":
            return base_key
        caption = self._normalize_subsection_caption(chunk.table_caption)
        if caption:
            # Captions like "(a)/(b)" or "（一）/（二）" usually represent true subsection boundaries.
            return f"{base_key} // {caption}"
        return base_key

    def _normalize_subsection_caption(self, caption: Optional[str]) -> str:
        """Normalize table caption when it looks like a subsection marker."""
        if not caption:
            return ""
        compact = " ".join(str(caption).split())
        patterns = [
            r"^\([a-zA-Z]\)",   # (a)
            r"^（[一二三四五六七八九十]+）",  # （一）
            r"^\([0-9]+\)",     # (1)
            r"^[（(][a-zA-Z0-9一二三四五六七八九十]+[）)]",
        ]
        if any(re.match(pattern, compact) for pattern in patterns):
            return compact
        return ""

    def _should_skip_group(self, chunks: list[NormalizedChunk]) -> bool:
        """Decide whether a group should be skipped before LLM extraction."""
        if not chunks:
            return True
        if not self.config.skip_heading_only_sections:
            return False
        return all(chunk.chunk_type == "heading" for chunk in chunks)

    def _split_group_into_windows(self, chunks: list[NormalizedChunk]) -> list[list[NormalizedChunk]]:
        """Split one group by a soft character budget with optional overlap."""
        if not chunks:
            return []

        windows: list[list[NormalizedChunk]] = []
        start = 0
        overlap = max(self.config.section_overlap_chunks, 0)

        while start < len(chunks):
            window: list[NormalizedChunk] = []
            total_chars = 0
            cursor = start

            while cursor < len(chunks):
                chunk = chunks[cursor]
                next_len = len(chunk.text)
                # Keep table-heavy windows small: avoid packing multiple table chunks together.
                if window and chunk.chunk_type == "table" and any(item.chunk_type == "table" for item in window):
                    break
                # Stop before crossing budget, but always keep at least one chunk.
                if window and total_chars + next_len > self.config.max_chars_per_section:
                    break
                window.append(chunk)
                total_chars += next_len
                cursor += 1

            if not window:
                window = [chunks[start]]
                cursor = start + 1

            windows.append(window)
            if cursor >= len(chunks):
                break

            # Overlap keeps boundary context for cross-window requirements.
            applied_overlap = overlap
            if window and overlap > 0:
                # Disable overlap around table boundaries to avoid duplicated table calls.
                boundary_has_table = window[-1].chunk_type == "table" or (
                    cursor < len(chunks) and chunks[cursor].chunk_type == "table"
                )
                if boundary_has_table:
                    applied_overlap = 0
            next_start = max(cursor - applied_overlap, start + 1)
            start = next_start

        return windows

    def _merge_small_groups(self, groups: list[list[NormalizedChunk]]) -> list[list[NormalizedChunk]]:
        """Merge adjacent small groups when merge_small_sections is enabled."""
        if not groups:
            return []

        merged_groups: list[list[NormalizedChunk]] = []
        pending = list(groups[0])

        for group in groups[1:]:
            if self._can_merge_groups(pending, group):
                pending = pending + group
                continue
            merged_groups.append(pending)
            pending = list(group)

        merged_groups.append(pending)
        return merged_groups

    def _can_merge_groups(self, left: list[NormalizedChunk], right: list[NormalizedChunk]) -> bool:
        """Return True if two groups can be merged without breaking constraints."""
        if not left or not right:
            return False

        left_chars = self._group_char_count(left)
        right_chars = self._group_char_count(right)
        combined_chars = left_chars + right_chars

        if left_chars >= self.config.min_section_chars and right_chars >= self.config.min_section_chars:
            return False
        if combined_chars > self.config.max_chars_per_section:
            return False
        if self._group_root_heading(left) != self._group_root_heading(right):
            return False
        return True

    def _group_char_count(self, chunks: list[NormalizedChunk]) -> int:
        """Calculate total text size of one group."""
        return sum(len(chunk.text) for chunk in chunks)

    def _group_root_heading(self, chunks: list[NormalizedChunk]) -> str:
        """Return first heading level used to prevent cross-topic merging."""
        if not chunks or not chunks[0].heading_path:
            return ""
        return chunks[0].heading_path[0]

    async def _extract_sections(
        self,
        document: PolicyDocument,
        groups: list[list[NormalizedChunk]],
    ) -> tuple[list[SectionExtractionResult], int]:
        """Run section extraction concurrently with a semaphore cap."""
        total_groups = len(groups)
        effective_concurrency = self._resolve_section_concurrency(total_groups)
        semaphore = asyncio.Semaphore(effective_concurrency)
        print(
            f"[policy_retractor] section scheduling: total_sections={total_groups}, "
            f"effective_concurrency={effective_concurrency}, auto={self.config.auto_section_concurrency}",
            flush=True,
        )

        async def _run_group(index: int, section_chunks: list[NormalizedChunk]) -> SectionExtractionResult:
            async with semaphore:
                return await self._extract_one_section(document, index, total_groups, section_chunks)

        tasks = [_run_group(index, section_chunks) for index, section_chunks in enumerate(groups, start=1)]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        section_results: list[SectionExtractionResult] = []
        section_error_count = 0
        for result in raw_results:
            if isinstance(result, Exception):
                section_error_count += 1
                continue
            if result.requirements:
                section_results.append(result)
        print(
            f"[policy_retractor] section extraction finished; success_sections={len(section_results)}, "
            f"error_sections={section_error_count}, total_sections={total_groups}, "
            f"effective_concurrency={effective_concurrency}",
            flush=True,
        )
        return section_results, section_error_count

    def _resolve_section_concurrency(self, total_groups: int) -> int:
        """Resolve runtime section concurrency.

        Auto mode uses a calibrated stable-max rule:
        ``effective = min(total_sections, stable_max_section_concurrency, max_section_concurrency)``.
        """
        total_groups = max(1, total_groups)
        hard_cap = max(1, self.config.max_section_concurrency)

        if not self.config.auto_section_concurrency:
            return min(hard_cap, total_groups)

        stable_max = max(1, self.config.stable_max_section_concurrency)
        return min(total_groups, hard_cap, stable_max)

    async def _extract_one_section(
        self,
        document: PolicyDocument,
        index: int,
        total_groups: int,
        section_chunks: list[NormalizedChunk],
    ) -> SectionExtractionResult:
        """Extract one section window and normalize ids/source locations."""
        section_title = self._section_title(section_chunks, index)
        section_started = time.perf_counter()
        print(
            f"[policy_retractor] section {index}/{total_groups} started; title={section_title}; "
            f"chunk_count={len(section_chunks)}",
            flush=True,
        )
        messages = self._to_lc_messages(build_section_messages(index, section_chunks))
        try:
            payload = await self._ainvoke_json_with_retry(messages)
        except Exception as exc:
            elapsed_seconds = round(time.perf_counter() - section_started, 2)
            print(
                f"[policy_retractor] section {index}/{total_groups} failed after {elapsed_seconds}s; "
                f"title={section_title}; error={type(exc).__name__}: {exc}",
                flush=True,
            )
            raise

        requirement_payloads = self._normalize_requirement_payloads(payload)
        normalized_requirements = self._finalize_section_requirements(
            section_chunks=section_chunks,
            requirement_payloads=requirement_payloads,
        )
        normalized_requirements = [
            self._attach_framework_fields(document, requirement)
            for requirement in normalized_requirements
        ]
        await self._emit_requirement_callbacks(normalized_requirements)

        elapsed_seconds = round(time.perf_counter() - section_started, 2)
        print(
            f"[policy_retractor] section {index}/{total_groups} finished in {elapsed_seconds}s; "
            f"title={section_title}; requirements={len(normalized_requirements)}",
            flush=True,
        )
        return SectionExtractionResult(
            section_title=section_title,
            section_summary=str(payload.get("sectionSummary", "") or payload.get("section_summary", "") or payload.get("summary", "")),
            requirements=normalized_requirements,
        )

    def _finalize_section_requirements(
        self,
        section_chunks: list[NormalizedChunk],
        requirement_payloads: list[Requirement],
    ) -> list[Requirement]:
        """Attach generated ids and source locations to section requirements."""

        normalized_requirements: list[Requirement] = []
        for requirement in requirement_payloads:
            req_id = self._new_requirement_id()
            requirement_source_text_candidate = requirement.source_text or requirement.description
            requirement_source_location = self._resolve_preferred_source_location(
                section_chunks=section_chunks,
                primary_reference_text=requirement_source_text_candidate,
                secondary_reference_text=self._compose_reference_text(
                    requirement.description,
                    requirement.title,
                ),
            )
            requirement_source_text = self._build_source_text_from_location(
                section_chunks=section_chunks,
                source_location=requirement_source_location,
                fallback_text=requirement_source_text_candidate,
            )
            requirement_source_text = self._normalize_percentage_notation(requirement_source_text)
            controls: list[Control] = []
            for control in requirement.controls:
                control_source_text_candidate = control.source_text or control.content
                control_source_location = self._resolve_preferred_source_location(
                    section_chunks=section_chunks,
                    primary_reference_text=control_source_text_candidate,
                    secondary_reference_text=control.content,
                    fallback=(
                        requirement_source_location
                        if self._should_fallback_to_requirement_location(control_source_text_candidate, control.content)
                        else None
                    ),
                )
                control_source_text = self._build_source_text_from_location(
                    section_chunks=section_chunks,
                    source_location=control_source_location,
                    fallback_text=control_source_text_candidate,
                )
                control_source_text = self._normalize_percentage_notation(control_source_text)
                controls.append(
                    control.model_copy(
                        update={
                            "control_id": self._new_control_id(),
                            "requirement_id": req_id,
                            "source_location": control_source_location,
                            "source_text": control_source_text,
                            "logic_description": control.logic_description or control.content,
                        }
                    )
                )

            normalized_requirements.append(
                requirement.model_copy(
                    update={
                        "requirement_id": req_id,
                        "source_text": requirement_source_text,
                        "source_location": requirement_source_location,
                        "controls": controls,
                        }
                    )
                )

        return normalized_requirements

    def _resolve_preferred_source_location(
        self,
        section_chunks: list[NormalizedChunk],
        primary_reference_text: str,
        secondary_reference_text: str = "",
        fallback: Optional[SourceLocation] = None,
    ) -> SourceLocation:
        """Locate source with sourceText-first strategy to reduce false positives."""
        primary_location = self._locate_source_location(
            section_chunks=section_chunks,
            reference_text=primary_reference_text,
            fallback=None,
        )
        if self._has_source_location(primary_location):
            return primary_location

        if self._normalize_match_text_v2(secondary_reference_text):
            secondary_location = self._locate_source_location(
                section_chunks=section_chunks,
                reference_text=secondary_reference_text,
                fallback=None,
            )
            if self._has_source_location(secondary_location):
                return secondary_location

        if fallback is not None:
            return fallback
        return SourceLocation()

    def _has_source_location(self, source_location: SourceLocation) -> bool:
        """Return True when locator points to one or more source elements."""
        return bool(source_location.element_ids)

    async def _emit_requirement_callbacks(self, requirements: list[Requirement]) -> None:
        """Emit one callback per requirement as soon as a section finishes."""
        if self.requirement_callback is None:
            return
        for requirement in requirements:
            result = self.requirement_callback(requirement)
            if inspect.isawaitable(result):
                await result

    def _attach_framework_fields(self, document: PolicyDocument, requirement: Requirement) -> Requirement:
        """Attach framework/file metadata to one requirement tree."""
        framework_id = self._build_framework_id(document.document_id)
        req_controls = [
            ctrl.model_copy(
                update={
                    "requirement_id": requirement.requirement_id,
                    "source_location": ctrl.source_location.model_copy(
                        update={"file_id": document.document_id}
                    ),
                }
            )
            for ctrl in requirement.controls
        ]
        return requirement.model_copy(
            update={
                "framework_id": framework_id,
                "source_location": requirement.source_location.model_copy(
                    update={"file_id": document.document_id}
                ),
                "controls": req_controls,
            }
        )

    async def _extract_framework(
        self,
        document: PolicyDocument,
        groups: list[list[NormalizedChunk]],
        section_results: list[SectionExtractionResult],
    ) -> dict:
        """Build and run framework-level LLM call using section summaries."""
        messages = self._to_lc_messages(build_framework_messages(document, groups, section_results))
        return await self._ainvoke_json_with_retry(messages)

    def _build_framework(
        self,
        document: PolicyDocument,
        framework_draft: dict,
        section_results: list[SectionExtractionResult],
        has_error: bool = False,
    ) -> Framework:
        """Assemble final framework object from normalized section outputs."""
        framework_id = self._build_framework_id(document.document_id)
        requirements: list[Requirement] = []

        for section in section_results:
            for req in section.requirements:
                requirements.append(self._attach_framework_fields(document, req))

        framework_name = str(framework_draft.get("title") or framework_draft.get("framework_title") or document.file_name)
        framework_desc = str(framework_draft.get("summary") or framework_draft.get("framework_summary") or "")

        return Framework(
            framework_id=framework_id,
            file_id=document.document_id,
            name=framework_name,
            status="ERROR" if has_error else "PENDING_REVIEW",
            description=framework_desc,
            requirements=requirements,
        )

    def _new_requirement_id(self) -> str:
        """Generate Java-compatible requirement ids."""
        return f"REQ-{uuid4().hex}"

    def _new_control_id(self) -> str:
        """Generate Java-compatible rule ids."""
        return f"RULE-{uuid4().hex}"

    def _build_framework_id(self, document_id: str) -> str:
        """Generate Java-compatible framework ids."""
        return f"FW-{document_id}"

    def _section_title(self, chunks: list[NormalizedChunk], index: int) -> str:
        """Resolve section title fallback for logging and output."""
        if chunks and chunks[0].chunk_type == "table":
            caption = self._normalize_subsection_caption(chunks[0].table_caption)
            if caption:
                return caption
        if chunks and chunks[0].heading_path:
            return chunks[0].heading_path[-1]
        return f"Section {index}"

    def _to_lc_messages(self, messages: list[tuple[str, str]]) -> list[SystemMessage | HumanMessage]:
        """Convert tuple-based prompts into LangChain message objects."""
        lc_messages: list[SystemMessage | HumanMessage] = []
        for role, content in messages:
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))
        return lc_messages

    async def _ainvoke_json_with_retry(self, messages) -> dict:
        """Call async model with retry and parse a JSON-like response."""
        if self.model is None:
            raise RuntimeError("LLM model is not initialized.")

        last_error = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = await self.model.ainvoke(messages)
                content = response.content if hasattr(response, "content") else str(response)
                return self._parse_json_payload(content)
            except Exception as exc:  # pragma: no cover
                last_error = exc
                if attempt >= self.config.max_retries or not self._is_retryable(exc):
                    raise
                # Backoff by attempt index to reduce thundering-herd retries.
                await asyncio.sleep(self.config.retry_delay_seconds * attempt)
        raise last_error

    def _is_retryable(self, exc: Exception) -> bool:
        """Check whether an exception looks transient and worth retrying."""
        message = str(exc).lower()
        if isinstance(exc, APIError):
            return True
        return any(token in message for token in ("502", "503", "504", "timeout", "temporarily", "bad gateway"))

    def _parse_json_payload(self, content: str) -> dict:
        """Extract and repair JSON payload from free-form model output."""
        content = content.strip()
        lower_content = content.lower()
        if "```json" in lower_content:
            start = lower_content.find("```json")
            body = content[start + 7 :]
            end = body.find("```")
            content = body[:end].strip() if end != -1 else body.strip()
        elif "```" in content:
            parts = content.split("```")
            if len(parts) >= 3:
                content = parts[1].strip()

        object_start = content.find("{")
        array_start = content.find("[")
        candidates = [idx for idx in (object_start, array_start) if idx != -1]
        if candidates:
            start = min(candidates)
            is_object = array_start == -1 or (object_start != -1 and object_start <= array_start)
            end = content.rfind("}" if is_object else "]")
            if end != -1:
                content = content[start : end + 1]

        parsed = json_repair.loads(content)
        if isinstance(parsed, list):
            return {"requirements": parsed}
        if not isinstance(parsed, dict):
            raise ValueError(f"LLM response is neither JSON object nor JSON array: {parsed}")
        return parsed

    def _normalize_requirement_payloads(self, payload: dict) -> list[Requirement]:
        """Normalize model requirement payload shape to a stable list form."""
        raw_requirements = payload.get("requirements")
        if isinstance(raw_requirements, list):
            return [self._build_requirement_from_payload(item) for item in raw_requirements if isinstance(item, dict)]

        if isinstance(payload.get("requirement"), str):
            single = {
                "title": payload.get("requirement"),
                "description": payload.get("description") or payload.get("requirement_description") or payload.get("summary") or "",
                "controls": payload.get("controls") or payload.get("control") or [],
            }
            return [self._build_requirement_from_payload(single)]
        return []

    def _build_requirement_from_payload(self, item: dict) -> Requirement:
        """Map one requirement payload dict into a Requirement model."""
        raw_controls = item.get("controls") or item.get("control") or []
        if isinstance(raw_controls, dict):
            raw_controls = [raw_controls]

        controls = [self._build_control_from_payload(ctrl) for ctrl in raw_controls if isinstance(ctrl, dict)]
        title = str(item.get("title") or item.get("requirement") or item.get("requirement_title") or "Unnamed Requirement").strip()
        description = str(item.get("description") or item.get("summary") or title).strip()
        return Requirement(
            requirement_id="",
            title=title,
            description=description,
            source_text=str(item.get("sourceText") or item.get("source_text") or description).strip(),
            controls=controls,
        )

    def _build_control_from_payload(self, item: dict) -> Control:
        """Map one control payload dict into a Control model."""
        content = str(
            item.get("content")
            or item.get("statement")
            or item.get("description")
            or item.get("text")
            or item.get("title")
            or item.get("control_title")
            or ""
        ).strip()
        raw_type = str(item.get("controlType") or item.get("control_type") or item.get("type") or "").strip().lower()
        normalized_type = self._normalize_control_type(raw_type, content)

        raw_risk = str(item.get("risk") or item.get("risk_level") or RiskLevel.MEDIUM.value).upper()
        if raw_risk not in {member.value for member in RiskLevel}:
            raw_risk = RiskLevel.MEDIUM.value

        return Control(
            control_id="",
            content=content,
            control_type=ControlType(normalized_type),
            risk=RiskLevel(raw_risk),
            source_text=str(item.get("source_text") or item.get("sourceText") or content).strip(),
            logic_description=str(item.get("logicDescription") or item.get("logic_description") or content).strip(),
        )

    def _normalize_control_type(self, raw_type: str, content: str) -> str:
        """Map loose control type aliases to supported enum values."""
        valid_types = {member.value for member in ControlType}
        if raw_type in valid_types:
            return raw_type

        alias_map = {
            "process": ControlType.QUALIFICATION_CHECK.value,
            "document": ControlType.DOCUMENT_CHECK.value,
            "prohibition": ControlType.PROHIBITED_ITEM_CHECK.value,
            "feedback": ControlType.MONITORING_CHECK.value,
            "eligibility_check": ControlType.QUALIFICATION_CHECK.value,
            "document_completeness_check": ControlType.DOCUMENT_CHECK.value,
            "document_validity_check": ControlType.DOCUMENT_CHECK.value,
            "threshold_compliance_check": ControlType.QUANTITATIVE_CHECK.value,
            "calculation_check": ControlType.QUANTITATIVE_CHECK.value,
            "timeline_check": ControlType.QUANTITATIVE_CHECK.value,
            "fund_use_compliance_check": ControlType.FUND_USE_CHECK.value,
            "collateral_compliance_check": ControlType.COLLATERAL_CHECK.value,
            "approval_authority_check": ControlType.DOCUMENT_CHECK.value,
            "account_verification_check": ControlType.DOCUMENT_CHECK.value,
            "post_loan_monitoring_check": ControlType.MONITORING_CHECK.value,
        }
        if raw_type in alias_map:
            return alias_map[raw_type]

        lowered = raw_type.lower()
        if any(token in lowered for token in ("prohibit", "forbid", "ban")):
            return ControlType.PROHIBITED_ITEM_CHECK.value
        if any(token in lowered for token in ("collateral", "mortgage", "pledge", "guarantee")):
            return ControlType.COLLATERAL_CHECK.value
        if any(token in lowered for token in ("fund", "usage", "use_of_proceeds")):
            return ControlType.FUND_USE_CHECK.value
        if any(token in lowered for token in ("threshold", "ratio", "limit", "quota", "calculation", "compute", "formula", "timeline", "timing", "deadline", "before", "after")):
            return ControlType.QUANTITATIVE_CHECK.value
        if any(token in lowered for token in ("monitor", "monitoring", "warning", "post_loan")):
            return ControlType.MONITORING_CHECK.value
        if any(token in lowered for token in ("approval", "authority", "authorization", "resolution", "validity", "license", "permit", "certificate", "document", "material", "submission", "attachment", "account", "escrow")):
            return ControlType.DOCUMENT_CHECK.value
        if any(token in lowered for token in ("qualification", "eligibility")):
            return ControlType.QUALIFICATION_CHECK.value

        normalized_content = content.lower()
        if any(token in normalized_content for token in ("禁止", "不得", "严禁", "不可")):
            return ControlType.PROHIBITED_ITEM_CHECK.value
        if any(token in normalized_content for token in ("抵押", "质押", "担保")):
            return ControlType.COLLATERAL_CHECK.value
        if any(token in normalized_content for token in ("资金用途", "专款专用", "用途")):
            return ControlType.FUND_USE_CHECK.value
        if any(token in normalized_content for token in ("比例", "不低于", "不超过", "%", "面积", "金额", "测算", "计算", "公式", "期限", "时点", "之前", "之后", "届满")):
            return ControlType.QUANTITATIVE_CHECK.value
        if any(token in normalized_content for token in ("贷后", "检查", "监控", "预警")):
            return ControlType.MONITORING_CHECK.value
        if any(token in normalized_content for token in ("提供", "提交", "资料", "材料", "报告", "合同", "有效", "证书", "执照", "批件", "证明", "审批", "审查", "批复", "决议", "授权", "账户", "开户", "专户")):
            return ControlType.DOCUMENT_CHECK.value
        if any(token in normalized_content for token in ("贷后", "检查", "监控", "预警")):
            return ControlType.MONITORING_CHECK.value
        return ControlType.QUALIFICATION_CHECK.value

    def _locate_source_location(
        self,
        section_chunks: list[NormalizedChunk],
        reference_text: str,
        fallback: Optional[SourceLocation] = None,
    ) -> SourceLocation:
        """Locate best source element id for one extracted fragment."""
        matched_chunks = self._collect_matched_chunks(section_chunks, reference_text)
        if not matched_chunks and fallback is not None:
            return fallback
        if not matched_chunks:
            return SourceLocation()

        return SourceLocation(
            element_ids=[
                int(chunk.source_child_id)
                for chunk in matched_chunks
                if str(chunk.source_child_id).isdigit()
            ],
            page_no=next((chunk.page for chunk in matched_chunks if chunk.page is not None), None),
        )

    def _compose_reference_text(self, *parts: Optional[str]) -> str:
        """Compose one matching reference from multiple candidate text fields."""
        merged_parts: list[str] = []
        seen: set[str] = set()
        for part in parts:
            normalized = " ".join(str(part or "").split()).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged_parts.append(normalized)
        return "\n".join(merged_parts)

    def _build_source_text_from_location(
        self,
        section_chunks: list[NormalizedChunk],
        source_location: SourceLocation,
        fallback_text: str = "",
    ) -> str:
        """Backfill source text from matched source elements to keep trace text canonical."""
        if not section_chunks or not source_location.element_ids:
            return str(fallback_text or "").strip()

        element_ids = {str(item) for item in source_location.element_ids}
        matched_chunks = [
            chunk for chunk in section_chunks if str(chunk.source_child_id) in element_ids
        ]
        if not matched_chunks:
            return str(fallback_text or "").strip()

        deduped_chunks = self._deduplicate_chunks(matched_chunks)
        texts: list[str] = []
        seen_texts: set[str] = set()
        for chunk in deduped_chunks:
            text = str(chunk.text or "").strip()
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            texts.append(text)

        if not texts:
            return str(fallback_text or "").strip()
        return "\n\n".join(texts)

    def _normalize_percentage_notation(self, text: str) -> str:
        """Normalize percentage tokens from latex/escaped forms to plain style."""
        normalized = str(text or "")
        if not normalized:
            return normalized
        # $10%$ -> 10%
        normalized = re.sub(r"\$([0-9]+(?:\.[0-9]+)?)%\$", r"\1%", normalized)
        # $15\%$ -> 15%
        normalized = re.sub(r"\$([0-9]+(?:\.[0-9]+)?)\\%\$", r"\1%", normalized)
        # 15\% -> 15%
        normalized = re.sub(r"([0-9]+(?:\.[0-9]+)?)\\%", r"\1%", normalized)
        return normalized

    def _should_fallback_to_requirement_location(self, *control_reference_texts: str) -> bool:
        """Fallback only when a control has no usable text of its own."""
        return not any(self._normalize_match_text_v2(text) for text in control_reference_texts)

    def _collect_matched_chunks(
        self,
        section_chunks: list[NormalizedChunk],
        reference_text: str,
    ) -> list[NormalizedChunk]:
        """Match full text first, then union per-segment matches for multi-part excerpts."""
        matches = self._match_chunks(section_chunks, reference_text)
        segment_matches: list[NormalizedChunk] = []
        for segment in self._split_reference_segments_v2(reference_text):
            segment_matches.extend(self._match_chunks(section_chunks, segment))
        return self._deduplicate_chunks([*matches, *segment_matches])

    def _match_chunks(self, section_chunks: list[NormalizedChunk], reference_text: str) -> list[NormalizedChunk]:
        """Match reference text to chunks using direct and fuzzy overlap."""
        normalized_reference = self._normalize_match_text_v2(reference_text)
        if not normalized_reference:
            return []

        # Fast path for direct containment after normalization.
        direct_matches = [
            chunk
            for chunk in section_chunks
            if normalized_reference in self._normalize_match_text_v2(chunk.text)
        ]
        if direct_matches:
            return self._deduplicate_chunks(direct_matches)

        reference_tokens = self._extract_match_tokens(normalized_reference)
        if not reference_tokens:
            return []

        scored_matches: list[tuple[float, NormalizedChunk]] = []
        for chunk in section_chunks:
            chunk_text = self._normalize_match_text_v2(chunk.text)
            chunk_tokens = self._extract_match_tokens(chunk_text)
            if not chunk_tokens:
                continue
            overlap = len(reference_tokens & chunk_tokens)
            if overlap == 0:
                continue
            # Keep robust matches by relative or absolute overlap thresholds.
            score = overlap / max(len(reference_tokens), 1)
            required_overlap = max(5, int(len(reference_tokens) * 0.5))
            if score >= 0.5 or overlap >= required_overlap:
                scored_matches.append((score, chunk))

        scored_matches.sort(key=lambda item: (-item[0], item[1].order))
        return self._deduplicate_chunks([chunk for _, chunk in scored_matches[:5]])

    def _split_reference_segments(self, reference_text: str) -> list[str]:
        """Split a merged reference into matchable clauses without over-fragmenting short text."""
        if not reference_text:
            return []
        parts = re.split(r"[\n；;。！？]+", reference_text)
        segments: list[str] = []
        seen: set[str] = set()
        for part in parts:
            normalized = " ".join(str(part).split()).strip()
            if not normalized:
                continue
            match_text = self._normalize_match_text_v2(normalized)
            if len(match_text) < 8:
                continue
            if match_text in seen:
                continue
            seen.add(match_text)
            segments.append(normalized)
        return segments

    def _split_reference_segments_v2(self, reference_text: str) -> list[str]:
        """Stable splitter for merged references with explicit punctuation set."""
        if not reference_text:
            return []
        parts = re.split(r"[\n,，。；;：:！!？?\u3001]+", reference_text)
        segments: list[str] = []
        seen: set[str] = set()
        for part in parts:
            normalized = " ".join(str(part).split()).strip()
            if not normalized:
                continue
            match_text = self._normalize_match_text_v2(normalized)
            if len(match_text) < 8:
                continue
            if match_text in seen:
                continue
            seen.add(match_text)
            segments.append(normalized)
        return segments

    def _normalize_match_text_v2(self, text: str) -> str:
        """Stable text normalization for source matching."""
        compact = re.sub(r"\s+", "", text or "")
        return compact.strip("，。；：、,.!?？！()（）[]【】\"'`")

    def _deduplicate_chunks(self, chunks: list[NormalizedChunk]) -> list[NormalizedChunk]:
        """Deduplicate chunks by source element while preserving document order."""
        best_by_child_id: dict[str, NormalizedChunk] = {}
        for chunk in chunks:
            existing = best_by_child_id.get(chunk.source_child_id)
            if existing is None or (chunk.order, chunk.chunk_id) < (existing.order, existing.chunk_id):
                best_by_child_id[chunk.source_child_id] = chunk
        return sorted(
            best_by_child_id.values(),
            key=lambda chunk: (
                chunk.page if chunk.page is not None else 10**9,
                chunk.order,
                chunk.chunk_id,
            ),
        )

    def _normalize_match_text(self, text: str) -> str:
        """Normalize text for similarity matching under OCR noise."""
        text = re.sub(r"\s+", "", text or "")
        return text.strip("，。；：、,.!！?？()（）[]【】\"'")

    def _extract_match_tokens(self, text: str) -> set[str]:
        """Tokenize text: Chinese n-grams or alnum tokens for fuzzy overlap."""
        compact_text = self._normalize_match_text_v2(text)
        if not compact_text:
            return set()
        if re.search(r"[\u4e00-\u9fff]", compact_text):
            window = min(4, len(compact_text))
            if len(compact_text) <= window:
                return {compact_text}
            return {compact_text[idx : idx + window] for idx in range(0, len(compact_text) - window + 1)}
        return {token for token in re.split(r"[^a-zA-Z0-9]+", compact_text.lower()) if token}
