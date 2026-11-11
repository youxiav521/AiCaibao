from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..schemas import ResourceFile
from ..schemas.material_models import DocumentManifest, EvidenceBlock
from ..utils import guess_candidate_tags, normalize_whitespace, stable_id, unique_preserve_order


class MaterialNormalizer:
    paragraph_types = {"text", "list"}

    def normalize_file(
        self,
        resource_file: ResourceFile,
        cleaned_items: List[Dict[str, Any]],
    ) -> Tuple[DocumentManifest, List[EvidenceBlock]]:
        sample_text = " ".join(item["text"] for item in cleaned_items if item["text"])[:1000]
        manifest = DocumentManifest(
            doc_id=stable_id("doc", resource_file.fileId),
            file_id=resource_file.fileId,
            file_name=resource_file.fileName,
            doc_type=self._guess_doc_type(resource_file.fileName, sample_text),
            page_count=max((int(item["page"] or 0) for item in cleaned_items), default=0),
            has_table=any(item["type"] == "table" for item in cleaned_items),
            has_excel_table=any(item["type"] == "excel_table" for item in cleaned_items),
            candidate_tags=guess_candidate_tags(resource_file.fileName, sample_text),
            metadata={},
        )
        blocks: List[EvidenceBlock] = []
        buffer: List[Dict[str, Any]] = []
        for item in cleaned_items:
            if item["type"] in self.paragraph_types:
                buffer.append(item)
                continue
            if buffer:
                blocks.append(self._build_paragraph_block(manifest, resource_file, buffer))
                buffer = []
            if item["type"] == "image":
                blocks.append(
                    EvidenceBlock(
                        evidence_id=stable_id("evidence", manifest.doc_id, item["element_id"], "image"),
                        doc_id=manifest.doc_id,
                        file_id=resource_file.fileId,
                        file_name=resource_file.fileName,
                        page=item["page"],
                        block_type="image_ref",
                        source_type="image",
                        text=normalize_whitespace(item["text"] or resource_file.fileName),
                        source_element_ids=[eid for eid in [item["element_id"]] if eid is not None],
                        bbox=item["bbox"],
                        metadata={"img_path": getattr(item["item"], "img_path", None)},
                    )
                )
        if buffer:
            blocks.append(self._build_paragraph_block(manifest, resource_file, buffer))
        return manifest, blocks

    def _build_paragraph_block(
        self,
        manifest: DocumentManifest,
        resource_file: ResourceFile,
        items: List[Dict[str, Any]],
    ) -> EvidenceBlock:
        text = normalize_whitespace("\n".join(item["text"] for item in items if item["text"]))
        pages = unique_preserve_order(str(item["page"]) for item in items if item["page"] is not None)
        page = int(pages[0]) if pages else None
        element_ids = [int(item["element_id"]) for item in items if item["element_id"] is not None]
        return EvidenceBlock(
            evidence_id=stable_id("evidence", manifest.doc_id, *element_ids[:10], "paragraph"),
            doc_id=manifest.doc_id,
            file_id=resource_file.fileId,
            file_name=resource_file.fileName,
            page=page,
            block_type="paragraph",
            source_type="merged_text_list",
            text=text,
            source_element_ids=element_ids,
            bbox=items[0]["bbox"] if items else None,
            metadata={"merged_item_count": len(items)},
        )

    def _guess_doc_type(self, file_name: str, sample_text: str) -> str | None:
        content = f"{file_name} {sample_text}"
        rules = {
            "business_license": ["营业执照", "统一社会信用代码"],
            "identity_document": ["身份证", "公民身份号码"],
            "credit_report": ["征信", "信用报告"],
            "credit_application": ["授信申请", "申请业务概况", "申请书"],
            "due_diligence_report": ["尽职调查", "调查分析", "尽调"],
            "compliance_material": ["合规性原材料", "资料审核", "真实性", "完整性"],
            "transaction_voucher": ["借款借据", "贴现凭证", "入账凭证"],
            "board_resolution": ["董事会决议"],
            "articles_of_association": ["公司章程", "章程"],
            "collateral_document": ["抵押", "质押", "不动产", "机动车登记证"],
            "bank_statement": ["账户明细", "银行流水", "交易日期"],
            "financial_statement": ["审计报告", "资产负债表", "利润表"],
        }
        for doc_type, keywords in rules.items():
            if any(keyword in content for keyword in keywords):
                return doc_type
        return None
