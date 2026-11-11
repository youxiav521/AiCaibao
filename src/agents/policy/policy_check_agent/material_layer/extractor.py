from __future__ import annotations

import re
from typing import Iterable, List

from .schemas import CanonicalDocument, CanonicalFieldValue, EvidenceLocator


FIELD_PATTERNS = {
    "company_name": [
        r"(?:企业名称|公司名称|名称)[:：]\s*([^\n]+)",
    ],
    "uscc": [
        r"(?:统一社会信用代码|社会信用代码)[:：]\s*([0-9A-Z]{18})",
    ],
    "legal_representative": [
        r"(?:法定代表人|法人)[:：]\s*([^\n]+)",
    ],
    "expiry_date": [
        r"(?:营业期限|有效期至|经营期限)[:：]\s*([^\n]+)",
    ],
}


def _normalize_field_value(field_name: str, value: str) -> str:
    normalized = value.strip()
    if field_name == "uscc":
        return normalized.upper().replace(" ", "")
    return normalized


class CanonicalFieldExtractor:
    def extract(self, document: CanonicalDocument) -> List[CanonicalFieldValue]:
        extracted: List[CanonicalFieldValue] = []
        for page in document.pages:
            for chunk in page.chunks:
                for field_name, patterns in FIELD_PATTERNS.items():
                    for pattern in patterns:
                        for match in re.finditer(pattern, chunk.text):
                            value = match.group(1).strip()
                            extracted.append(
                                CanonicalFieldValue(
                                    entityType="company",
                                    canonicalField=field_name,
                                    value=value,
                                    normalizedValue=_normalize_field_value(field_name, value),
                                    confidence=0.88,
                                    evidence=[
                                        EvidenceLocator(
                                            fileId=document.fileId,
                                            fileName=document.fileName,
                                            pageNo=page.pageNo,
                                            elementId=chunk.elementId,
                                            chunkId=chunk.chunkId,
                                            bbox=chunk.bbox,
                                        )
                                    ],
                                    aliases=[],
                                )
                            )
        return extracted
