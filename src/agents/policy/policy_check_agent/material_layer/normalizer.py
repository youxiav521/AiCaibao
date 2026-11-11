from __future__ import annotations

from typing import Any, List

from src.agents.content_generate_agent.utils import count_tokens
from src.agents.policy.policy_check_agent.schemas import ResourceFile, item_content_extract

from .schemas import CanonicalChunk, CanonicalDocument, CanonicalPage, CanonicalTable, CanonicalTableRow, DocumentClassification


def _bbox_to_list(raw_bbox: Any) -> List[float] | None:
    if not raw_bbox:
        return None
    try:
        return [
            float(raw_bbox.l),
            float(raw_bbox.t),
            float(raw_bbox.r),
            float(raw_bbox.b),
        ]
    except Exception:
        return None


class MineruDocumentNormalizer:
    def normalize(
        self,
        resource_file: ResourceFile,
        classification: DocumentClassification,
    ) -> CanonicalDocument:
        pages_by_no: dict[int, CanonicalPage] = {}
        for item in resource_file.content:
            page_no = int(getattr(item, "page", 0) or 0)
            page = pages_by_no.setdefault(page_no, CanonicalPage(pageNo=page_no))
            text = item_content_extract(item)
            if text:
                chunk = CanonicalChunk(
                    chunkId=f"{resource_file.fileId}:{getattr(item, 'element_id', 'na')}",
                    pageNo=page_no,
                    elementId=getattr(item, "element_id", None),
                    chunkType="image_caption" if getattr(item, "type", "") == "image" else "text",
                    text=text,
                    bbox=_bbox_to_list(getattr(item, "bBox", None)),
                    tokensApprox=count_tokens(text),
                    metadata={
                        "sourceType": getattr(item, "type", None),
                    },
                )
                page.chunks.append(chunk)

            if getattr(item, "type", None) in {"table", "excel_table"}:
                rows = []
                for idx, row in enumerate(getattr(item, "rows", []) or []):
                    row_values = []
                    for cell in row:
                        row_values.append(str(getattr(cell, "text", "")))
                    rows.append(CanonicalTableRow(rowIndex=idx, values=row_values))
                page.tables.append(
                    CanonicalTable(
                        tableId=f"{resource_file.fileId}:table:{getattr(item, 'element_id', 'na')}",
                        pageNo=page_no,
                        elementId=getattr(item, "element_id", None),
                        headers=[],
                        rows=rows,
                        rawMarkdown=getattr(item, "md", None),
                    )
                )

        pages = [pages_by_no[page_no] for page_no in sorted(pages_by_no.keys())]
        return CanonicalDocument(
            docId=resource_file.fileId,
            fileId=resource_file.fileId,
            fileName=resource_file.fileName,
            originalFileName=resource_file.originalFileName,
            docClass=classification.docClass,
            classification=classification,
            pages=pages,
            aliases=[resource_file.fileId, resource_file.fileName, *([resource_file.originalFileName] if resource_file.originalFileName else [])],
            metadata={
                "docType": resource_file.docType,
                "businessTags": [],
                "pageCount": len(pages),
            },
        )
