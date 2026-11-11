from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .schemas import CanonicalDocument, CanonicalFieldValue, EvidenceLocator, MaterialUnderstandingResult, RetrievalPlan


QUERY_CLASS_HINTS: Dict[str, List[str]] = {
    "business_license": ["营业执照", "统一社会信用代码", "工商"],
    "identity_document": ["身份证", "法人身份证", "法定代表人身份证明"],
    "bank_statement": ["银行流水", "账户明细", "流水"],
    "credit_report": ["征信", "企业信用报告", "信用报告", "贷款卡"],
    "credit_application": ["授信业务申请书", "授信申请", "申请业务概况", "申请人本行存款"],
    "due_diligence_report": ["尽职调查", "调查分析报告", "授信调查", "尽调"],
    "compliance_material": ["合规性原材料", "资料审核", "完整性", "表面真实性", "原件核验"],
    "transaction_voucher": ["借款借据", "贴现凭证", "入账凭证", "贷款借据"],
    "collateral_document": ["担保", "抵押", "质押", "他项权证", "不动产", "机动车登记证"],
    "contract": ["合同", "协议", "借款合同", "授信合同", "担保协议"],
    "authorization_letter": ["授权书", "授权委托书", "委托书"],
    "financial_statement": ["审计报告", "财报", "资产负债表", "利润表"],
    "board_resolution": ["董事会决议", "董事会"],
    "articles_of_association": ["公司章程", "章程"],
}


def _tokenize(value: str) -> List[str]:
    return [token.lower() for token in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]+", value or "")]


class MaterialEvidenceRetriever:
    def __init__(self, result: MaterialUnderstandingResult) -> None:
        self.result = result
        self.docs_by_id = {document.docId: document for document in result.documents}

    def _score_document(self, query: str, document: CanonicalDocument) -> Tuple[int, List[str]]:
        query_tokens = set(_tokenize(query))
        score = 0
        reasons: List[str] = []

        for doc_class, hints in QUERY_CLASS_HINTS.items():
            if document.docClass != doc_class:
                continue
            overlap = query_tokens & set(_tokenize(" ".join(hints)))
            if overlap:
                score += 900 + 80 * len(overlap)
                reasons.append(f"doc_class={doc_class}")

        alias_text = " ".join(document.aliases + [document.fileName, document.originalFileName or ""])
        alias_overlap = query_tokens & set(_tokenize(alias_text))
        if alias_overlap:
            score += 600 + 60 * len(alias_overlap)
            reasons.append(f"alias_overlap={','.join(sorted(alias_overlap))}")
        if query and query in alias_text:
            score += 900
            reasons.append("alias_exact_substring")

        for field in document.canonicalFields:
            field_tokens = set(_tokenize(" ".join([field.canonicalField, field.value, field.normalizedValue])))
            overlap = query_tokens & field_tokens
            if overlap:
                score += 500 + 70 * len(overlap)
                reasons.append(f"field={field.canonicalField}")

        chunk_hits = 0
        chunk_best_overlap = 0
        for page in document.pages:
            for chunk in page.chunks:
                chunk_tokens = set(_tokenize(chunk.text))
                overlap = query_tokens & chunk_tokens
                if overlap:
                    chunk_hits += 1
                    chunk_best_overlap = max(chunk_best_overlap, len(overlap))
                if query and query in chunk.text:
                    chunk_best_overlap = max(chunk_best_overlap, 6)
        if chunk_hits:
            score += min(chunk_hits * 40, 400) + min(chunk_best_overlap * 120, 600)
            reasons.append(f"chunk_hits={chunk_hits}")

        return score, reasons

    def query_documents(self, query: str, limit: int = 5) -> List[Dict[str, object]]:
        scored: List[Dict[str, object]] = []
        for document in self.result.documents:
            score, reasons = self._score_document(query, document)
            if score <= 0:
                continue
            scored.append(
                {
                    "fileId": document.fileId,
                    "fileName": document.fileName,
                    "docClass": document.docClass,
                    "score": score,
                    "reasons": reasons,
                }
            )
        scored.sort(key=lambda item: (-int(item["score"]), str(item["fileName"])))
        return scored[:limit]

    def query_chunks(self, file_id: str, query: str, limit: int = 12) -> List[Dict[str, object]]:
        document = self.docs_by_id.get(file_id)
        if document is None:
            return []
        query_tokens = set(_tokenize(query))
        ranked: List[Dict[str, object]] = []
        for page in document.pages:
            for chunk in page.chunks:
                chunk_tokens = set(_tokenize(chunk.text))
                overlap = query_tokens & chunk_tokens
                score = len(overlap) * 100
                if query and query.lower() in chunk.text.lower():
                    score += 500
                if score <= 0:
                    continue
                ranked.append(
                    {
                        "chunkId": chunk.chunkId,
                        "pageNo": chunk.pageNo,
                        "elementId": chunk.elementId,
                        "text": chunk.text,
                        "score": score,
                        "bbox": chunk.bbox,
                    }
                )
        ranked.sort(key=lambda item: (-int(item["score"]), int(item.get("elementId") or 0)))
        return ranked[:limit]

    def query_with_plan(
        self,
        plan: RetrievalPlan,
        doc_limit: int = 5,
        chunk_limit: int = 8,
    ) -> Dict[str, object]:
        candidate_scores: Dict[str, Dict[str, object]] = {}
        for doc_class in plan.docClasses:
            for doc_id in self.result.evidenceIndex.docsByClass.get(doc_class, []):
                document = self.docs_by_id.get(doc_id)
                if document is None:
                    continue
                candidate_scores[document.fileId] = {
                    "fileId": document.fileId,
                    "fileName": document.fileName,
                    "docClass": document.docClass,
                    "score": 1600,
                    "reasons": [f"planned_doc_class={doc_class}"],
                }

        for query in plan.keywordQueries or [plan.controlText]:
            for item in self.query_documents(query, limit=max(doc_limit * 2, doc_limit)):
                file_id = str(item["fileId"])
                score = int(item["score"])
                if file_id not in candidate_scores or score > int(candidate_scores[file_id]["score"]):
                    candidate_scores[file_id] = item

        ranked_docs = list(candidate_scores.values())
        if plan.docClasses:
            ranked_docs.sort(
                key=lambda item: (
                    0 if item.get("docClass") in plan.docClasses else 1,
                    -int(item["score"]),
                    str(item["fileName"]),
                )
            )
        else:
            ranked_docs.sort(key=lambda item: (-int(item["score"]), str(item["fileName"])))
        ranked_docs = ranked_docs[:doc_limit]

        doc_chunks: Dict[str, List[Dict[str, object]]] = {}
        for item in ranked_docs:
            file_id = str(item["fileId"])
            merged_chunks: List[Dict[str, object]] = []
            for query in plan.keywordQueries[:3] or [plan.controlText]:
                merged_chunks.extend(self.query_chunks(file_id, query, limit=chunk_limit))
            deduped = []
            seen = set()
            for chunk in sorted(merged_chunks, key=lambda c: (-int(c["score"]), int(c.get("elementId") or 0))):
                key = (chunk.get("chunkId"), chunk.get("elementId"))
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(chunk)
                if len(deduped) >= chunk_limit:
                    break
            doc_chunks[file_id] = deduped

        field_hits: Dict[str, List[Dict[str, object]]] = {}
        for field_name in plan.canonicalFields:
            hits = self.result.evidenceIndex.fieldValues.get(field_name, [])
            field_hits[field_name] = [
                {
                    "value": field.value,
                    "normalizedValue": field.normalizedValue,
                    "confidence": field.confidence,
                    "evidence": [e.model_dump() for e in field.evidence],
                }
                for field in hits[:10]
            ]

        return {
            "plan": plan.model_dump(),
            "docCandidates": ranked_docs,
            "chunkCandidates": doc_chunks,
            "candidateDocuments": ranked_docs,
            "candidateChunks": doc_chunks,
            "fieldHits": field_hits,
        }
