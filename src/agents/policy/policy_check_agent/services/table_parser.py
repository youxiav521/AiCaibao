from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..schemas import ResourceFile
from ..schemas.material_models import DocumentManifest, EvidenceBlock, FactRecord
from ..utils import is_form_like_table, kv_pairs_from_rows, normalize_whitespace, parse_markdown_table, row_to_text, stable_id


class TableParser:
    certificate_doc_types = {"business_license", "property_certificate", "vehicle_certificate", "collateral_document"}
    detail_doc_types = {"bank_statement", "transaction_voucher", "financial_statement"}

    def parse_file(
        self,
        resource_file: ResourceFile,
        manifest: DocumentManifest,
        cleaned_items: List[Dict[str, Any]],
    ) -> Tuple[List[EvidenceBlock], List[FactRecord]]:
        blocks: List[EvidenceBlock] = []
        facts: List[FactRecord] = []
        for item in cleaned_items:
            if item["type"] not in {"table", "excel_table"}:
                continue
            rows = self._extract_rows(item)
            if not rows:
                continue
            if self._should_parse_as_kv(manifest.doc_type, rows):
                kv_pairs = kv_pairs_from_rows(rows)
                summary_values = []
                for key, value in kv_pairs:
                    evidence_id = stable_id("evidence", manifest.doc_id, item["element_id"], key, value)
                    blocks.append(
                        EvidenceBlock(
                            evidence_id=evidence_id,
                            doc_id=manifest.doc_id,
                            file_id=resource_file.fileId,
                            file_name=resource_file.fileName,
                            page=item["page"],
                            block_type="kv_fact_text",
                            source_type=item["type"],
                            text=f"{key}: {value}",
                            source_element_ids=[eid for eid in [item["element_id"]] if eid is not None],
                            bbox=item["bbox"],
                            metadata={"table_mode": "kv"},
                        )
                    )
                    facts.append(
                        FactRecord(
                            fact_id=stable_id("fact", evidence_id, key),
                            doc_id=manifest.doc_id,
                            evidence_id=evidence_id,
                            fact_type="table_kv",
                            field_name=key,
                            field_value=value,
                            normalized_value=normalize_whitespace(value),
                            confidence=0.72,
                            page=item["page"],
                            metadata={"source_element_ids": [item["element_id"]] if item["element_id"] is not None else []},
                        )
                    )
                    summary_values.append(f"{key}: {value}")
                if summary_values:
                    blocks.append(
                        EvidenceBlock(
                            evidence_id=stable_id("evidence", manifest.doc_id, item["element_id"], "table_summary"),
                            doc_id=manifest.doc_id,
                            file_id=resource_file.fileId,
                            file_name=resource_file.fileName,
                            page=item["page"],
                            block_type="raw_table_summary",
                            source_type=item["type"],
                            text="; ".join(summary_values[:10]),
                            source_element_ids=[eid for eid in [item["element_id"]] if eid is not None],
                            bbox=item["bbox"],
                            metadata={"table_mode": "summary"},
                        )
                    )
                continue

            for row_index, row in enumerate(rows):
                row_text = row_to_text(row)
                if not row_text:
                    continue
                blocks.append(
                    EvidenceBlock(
                        evidence_id=stable_id("evidence", manifest.doc_id, item["element_id"], row_index, "row"),
                        doc_id=manifest.doc_id,
                        file_id=resource_file.fileId,
                        file_name=resource_file.fileName,
                        page=item["page"],
                        block_type="table_row_text",
                        source_type=item["type"],
                        text=row_text,
                        source_element_ids=[eid for eid in [item["element_id"]] if eid is not None],
                        bbox=item["bbox"],
                        metadata={"row_index": row_index},
                    )
                )
        return blocks, facts

    def _extract_rows(self, item: Dict[str, Any]) -> List[List[str]]:
        raw_item = item["item"]
        for attr in ("rows", "table_rows"):
            rows = getattr(raw_item, attr, None)
            if rows:
                extracted: List[List[str]] = []
                for row in rows:
                    if isinstance(row, list):
                        extracted.append([normalize_whitespace(str(cell)) for cell in row])
                if extracted:
                    return extracted
        return parse_markdown_table(item.get("md") or item.get("text") or "")

    def _should_parse_as_kv(self, doc_type: str | None, rows: List[List[str]]) -> bool:
        if doc_type in self.certificate_doc_types:
            return True
        if doc_type in self.detail_doc_types:
            return False
        return is_form_like_table(rows)
