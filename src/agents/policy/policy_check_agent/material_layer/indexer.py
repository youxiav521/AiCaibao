from __future__ import annotations

import re
from typing import Iterable, List

from .schemas import CanonicalDocument, CanonicalFieldValue, EvidenceIndex, EvidenceLocator


def _tokenize(value: str) -> List[str]:
    return [token.lower() for token in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]+", value or "")]


class EvidenceIndexer:
    def build(self, documents: List[CanonicalDocument]) -> EvidenceIndex:
        index = EvidenceIndex()
        for document in documents:
            index.docsByClass.setdefault(document.docClass, []).append(document.docId)
            index.docSummaries[document.docId] = {
                "fileId": document.fileId,
                "fileName": document.fileName,
                "docClass": document.docClass,
                "pageCount": len(document.pages),
            }

            for alias in document.aliases:
                if not alias:
                    continue
                index.docsByEntity.setdefault(alias.lower(), []).append(document.docId)

            for field in document.canonicalFields:
                index.docsByField.setdefault(field.canonicalField, []).append(document.docId)
                index.fieldValues.setdefault(field.canonicalField, []).append(field)
                index.docsByEntity.setdefault(field.normalizedValue.lower(), []).append(document.docId)

            for page in document.pages:
                for chunk in page.chunks:
                    locator = EvidenceLocator(
                        fileId=document.fileId,
                        fileName=document.fileName,
                        pageNo=page.pageNo,
                        elementId=chunk.elementId,
                        chunkId=chunk.chunkId,
                        bbox=chunk.bbox,
                    )
                    for token in _tokenize(chunk.text):
                        index.chunkTextIndex.setdefault(token, []).append(locator)

        for mapping in [index.docsByClass, index.docsByEntity, index.docsByField]:
            for key, values in mapping.items():
                mapping[key] = list(dict.fromkeys(values))
        return index
