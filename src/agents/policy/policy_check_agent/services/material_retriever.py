from __future__ import annotations

from typing import Dict, List

from ..schemas.control_models import ControlRetrievalRequest
from ..schemas.material_models import MaterialStore
from ..schemas.result_models import ControlRetrievalResult, DocumentHit, EvidenceHit, FactHit
from ..utils import build_query_token_weights, keyword_tokens, normalize_text


class MaterialRetriever:
    def __init__(self, store: MaterialStore) -> None:
        self.store = store
        self.manifests_by_doc_id = {manifest.doc_id: manifest for manifest in store.manifests}
        self.blocks_by_id = {block.evidence_id: block for block in store.evidence_blocks}
        self.facts_by_id = {fact.fact_id: fact for fact in store.facts}
        self.block_keyword_index: Dict[str, List[str]] = getattr(store, "_block_keyword_index", {})
        self.fact_index: Dict[str, List[str]] = getattr(store, "_fact_index", {})

    @staticmethod
    def _weighted_overlap_score(query_weights: Dict[str, float], target_tokens: set[str]) -> tuple[float, List[str]]:
        overlap = [token for token in query_weights if token in target_tokens]
        score = sum(query_weights[token] for token in overlap)
        return score, overlap

    def find_candidate_documents(
        self,
        query: str,
        *,
        preferred_doc_types: List[str] | None = None,
        preferred_tags: List[str] | None = None,
        top_k: int = 5,
    ) -> List[DocumentHit]:
        preferred_doc_types = preferred_doc_types or []
        preferred_tags = preferred_tags or []
        query_weights = build_query_token_weights(query)
        hits: List[DocumentHit] = []
        for manifest in self.store.manifests:
            score = 0.0
            reasons: List[str] = []
            manifest_text = f"{manifest.file_name} {' '.join(manifest.candidate_tags)} {manifest.doc_type or ''}"
            overlap_score, overlap = self._weighted_overlap_score(query_weights, set(keyword_tokens(manifest_text)))
            if overlap:
                score += 2.5 + 0.4 * overlap_score
                reasons.append(f"token_overlap={','.join(sorted(overlap))}")
            if preferred_doc_types and manifest.doc_type in preferred_doc_types:
                score += 3.0
                reasons.append("preferred_doc_type")
            if preferred_tags and set(preferred_tags) & set(manifest.candidate_tags):
                score += 2.0
                reasons.append("preferred_tag")
            if score > 0:
                hits.append(DocumentHit(manifest=manifest, score=score, reasons=reasons))
        hits.sort(key=lambda item: (-item.score, item.manifest.file_name))
        return hits[:top_k]

    def search_evidence_blocks(
        self,
        query: str,
        *,
        doc_ids: List[str] | None = None,
        top_k: int = 8,
    ) -> List[EvidenceHit]:
        doc_ids_set = set(doc_ids or [])
        query_weights = build_query_token_weights(query)
        candidate_ids = set()
        for token in query_weights:
            candidate_ids.update(self.block_keyword_index.get(token, []))
        if not candidate_ids:
            candidate_ids = set(self.blocks_by_id.keys())
        hits: List[EvidenceHit] = []
        for evidence_id in candidate_ids:
            block = self.blocks_by_id[evidence_id]
            if doc_ids_set and block.doc_id not in doc_ids_set:
                continue
            overlap_score, overlap = self._weighted_overlap_score(
                query_weights,
                set(keyword_tokens(f"{block.file_name} {block.text}")),
            )
            if not overlap:
                continue
            score = overlap_score * 1.2
            reasons = [f"token_overlap={','.join(sorted(overlap))}"]
            if normalize_text(query) and normalize_text(query) in normalize_text(block.text):
                score += 3.0
                reasons.append("exact_substring")
            hits.append(EvidenceHit(block=block, score=score, reasons=reasons))
        if not hits and doc_ids_set:
            fallback_blocks = [
                block for block in self.blocks_by_id.values()
                if block.doc_id in doc_ids_set and block.block_type != "image_ref"
            ]
            fallback_blocks.sort(key=lambda block: (block.page or 0, block.file_name, block.evidence_id))
            for block in fallback_blocks[:top_k]:
                hits.append(
                    EvidenceHit(
                        block=block,
                        score=0.5,
                        reasons=["doc_fallback"],
                    )
                )
        hits.sort(key=lambda item: (-item.score, item.block.file_name, item.block.page or 0))
        return hits[:top_k]

    def search_facts(
        self,
        *,
        field_names: List[str] | None = None,
        query: str = "",
        doc_ids: List[str] | None = None,
        top_k: int = 8,
    ) -> List[FactHit]:
        field_names = field_names or []
        doc_ids_set = set(doc_ids or [])
        candidate_ids = set()
        for field_name in field_names:
            candidate_ids.update(self.fact_index.get(field_name, []))
        query_weights = build_query_token_weights(query)
        for token in query_weights:
            candidate_ids.update(self.fact_index.get(token, []))
        if not candidate_ids:
            candidate_ids = set(self.facts_by_id.keys())
        hits: List[FactHit] = []
        for fact_id in candidate_ids:
            fact = self.facts_by_id[fact_id]
            if doc_ids_set and fact.doc_id not in doc_ids_set:
                continue
            score = 0.0
            reasons: List[str] = []
            if field_names and fact.field_name in field_names:
                score += 3.0
                reasons.append("field_name_match")
            overlap_score, overlap = self._weighted_overlap_score(
                query_weights,
                set(keyword_tokens(f"{fact.field_name} {fact.field_value} {fact.normalized_value or ''}")),
            )
            if overlap:
                score += 1.5 + 0.5 * overlap_score
                reasons.append(f"token_overlap={','.join(sorted(overlap))}")
            if score > 0:
                hits.append(FactHit(fact=fact, score=score, reasons=reasons))
        hits.sort(key=lambda item: (-item.score, item.fact.field_name, item.fact.field_value))
        return hits[:top_k]

    def retrieve_for_control(self, control: ControlRetrievalRequest) -> ControlRetrievalResult:
        control = self._enrich_control_request(control)
        documents = self.find_candidate_documents(
            control.control_text,
            preferred_doc_types=control.preferred_doc_types,
            preferred_tags=control.preferred_tags,
            top_k=control.top_k_documents,
        )
        doc_ids = [hit.manifest.doc_id for hit in documents]
        query = " ".join([control.control_text, *control.keywords]).strip()
        evidence_hits = self.search_evidence_blocks(query, doc_ids=doc_ids, top_k=control.top_k_blocks)
        fact_hits = self.search_facts(
            field_names=control.fact_fields,
            query=query,
            doc_ids=doc_ids,
            top_k=control.top_k_facts,
        )
        return ControlRetrievalResult(
            control_id=control.control_id,
            control_text=control.control_text,
            candidate_documents=documents,
            evidence_hits=evidence_hits,
            fact_hits=fact_hits,
            debug={
                "doc_ids": doc_ids,
                "preferred_doc_types": control.preferred_doc_types,
                "preferred_tags": control.preferred_tags,
                "keywords": control.keywords,
                "fact_fields": control.fact_fields,
            },
        )

    def _enrich_control_request(self, control: ControlRetrievalRequest) -> ControlRetrievalRequest:
        preferred_doc_types = list(control.preferred_doc_types)
        preferred_tags = list(control.preferred_tags)
        keywords = list(control.keywords)
        fact_fields = list(control.fact_fields)
        text = control.control_text or ""

        rules = [
            (["营业执照", "工商登记", "统一社会信用代码"], ["business_license"], ["license"], ["营业执照", "统一社会信用代码"], ["uscc", "company_name"]),
            (["法人代码证书", "法定代表人身份证明", "身份证"], ["identity_document"], ["identity"], ["身份证", "法定代表人"], ["person_name"]),
            (["征信", "信用报告", "贷款卡"], ["credit_report"], ["credit"], ["征信", "信用报告", "贷款卡"], []),
            (["授信业务申请书", "授信申请", "申请业务概况"], ["credit_application"], ["application"], ["授信申请", "申请业务概况", "申请人本行存款"], []),
            (["授信调查", "调查分析报告", "尽职调查", "尽调"], ["due_diligence_report"], ["due_diligence"], ["尽职调查", "调查分析报告", "调查实施情况"], []),
            (["资料审核", "完整性", "表面真实性", "原件", "复印件", "签章清晰"], ["compliance_material"], ["compliance"], ["合规性原材料", "资料审核", "完整性", "真实性"], []),
            (["借款借据", "贴现凭证", "入账凭证", "放款凭证"], ["transaction_voucher"], ["statement"], ["借款借据", "贴现凭证", "入账凭证"], []),
            (["董事会决议", "董事会成员", "签字样本"], ["board_resolution"], ["resolution"], ["董事会决议", "董事会成员", "签字样本"], []),
            (["公司章程", "章程"], ["articles_of_association"], ["charter"], ["公司章程", "章程"], []),
            (["担保", "抵押", "质押", "他项权证", "不动产", "机动车登记证"], ["collateral_document"], ["collateral"], ["担保", "抵押", "质押", "他项权证", "不动产", "机动车登记证"], ["property_owner", "vehicle_owner", "vehicle_registration_number", "vehicle_vin"]),
            (["银行流水", "账户明细", "交易日期"], ["bank_statement"], ["statement"], ["银行流水", "账户明细"], []),
        ]

        rules.extend(
            [
                (
                    ["提货担保函", "提货担保", "trust receipt", "信托收据", "展期协议", "押汇申请书", "进口代收押汇申请书"],
                    ["transaction_voucher", "credit_application"],
                    ["statement", "application"],
                    ["提货担保函", "信托收据", "展期协议", "押汇申请书"],
                    [],
                ),
                (
                    ["担保书", "担保协议", "授权委托书", "法定代表人身份证明", "签字样本", "董事会成员名单"],
                    ["compliance_material", "board_resolution", "identity_document"],
                    ["compliance", "resolution", "identity"],
                    ["担保书", "担保协议", "授权委托书", "法定代表人身份证明", "签字样本", "董事会成员名单"],
                    ["person_name"],
                ),
                (
                    ["抵押登记证明", "他项权证", "抵押登记权利证明", "收妥凭证"],
                    ["collateral_document"],
                    ["collateral"],
                    ["抵押登记证明", "他项权证", "抵押登记权利证明", "收妥凭证"],
                    ["property_owner", "vehicle_owner", "vehicle_registration_number"],
                ),
            ]
        )

        for trigger_terms, doc_types, tags, extra_keywords, fields in rules:
            if not any(term in text for term in trigger_terms):
                continue
            for doc_type in doc_types:
                if doc_type not in preferred_doc_types:
                    preferred_doc_types.append(doc_type)
            for tag in tags:
                if tag not in preferred_tags:
                    preferred_tags.append(tag)
            for keyword in extra_keywords:
                if keyword not in keywords:
                    keywords.append(keyword)
            for field in fields:
                if field not in fact_fields:
                    fact_fields.append(field)

        return ControlRetrievalRequest(
            control_id=control.control_id,
            control_text=control.control_text,
            preferred_doc_types=preferred_doc_types,
            preferred_tags=preferred_tags,
            keywords=keywords,
            fact_fields=fact_fields,
            top_k_documents=control.top_k_documents,
            top_k_blocks=control.top_k_blocks,
            top_k_facts=control.top_k_facts,
            requirement_id=control.requirement_id,
        )
