from __future__ import annotations

import re
from typing import Iterable, List

from ..schemas import ResourceFile
from ..schemas.material_models import DocumentManifest, EvidenceBlock, FactRecord
from ..utils import normalize_whitespace, stable_id


FACT_PATTERNS = {
    "company_name": [re.compile(r"(?:公司名称|企业名称)[:：]\s*([^\n，。,；;]{2,80})")],
    "person_name": [re.compile(r"(?:姓名|法人代表|法定代表人)[:：]\s*([^\n，。,；;]{2,30})")],
    "uscc": [re.compile(r"(?:统一社会信用代码|社会信用代码)[:：]?\s*([0-9A-Z]{18})")],
    "property_owner": [re.compile(r"(?:权利人|所有权人|不动产权人)[:：]\s*([^\n，。,；;]{2,50})")],
    "property_usage_term": [re.compile(r"(?:使用期限|土地使用期限|使用权期限)[:：]\s*([^\n]{4,80})")],
    "vehicle_owner": [re.compile(r"(?:机动车所有人|所有人)[:：]\s*([^\n，。,；;]{2,50})")],
    "vehicle_registration_number": [re.compile(r"(?:号牌号码|车牌号码|注册登记编号)[:：]\s*([A-Z0-9\u4e00-\u9fff\-]{4,30})")],
    "vehicle_registration_date": [re.compile(r"(?:注册日期|注册登记日期)[:：]\s*([0-9]{4}[-./年][0-9]{1,2}[-./月][0-9]{1,2}日?)")],
    "vehicle_vin": [re.compile(r"(?:车辆识别代号|VIN)[:：]?\s*([A-HJ-NPR-Z0-9]{8,20})")],
    "vehicle_issue_date": [re.compile(r"(?:发证日期|签发日期)[:：]\s*([0-9]{4}[-./年][0-9]{1,2}[-./月][0-9]{1,2}日?)")],
}


class FactExtractor:
    def extract_for_document(
        self,
        resource_file: ResourceFile,
        manifest: DocumentManifest,
        evidence_blocks: Iterable[EvidenceBlock],
    ) -> List[FactRecord]:
        facts: List[FactRecord] = []
        for block in evidence_blocks:
            for field_name, patterns in FACT_PATTERNS.items():
                for pattern in patterns:
                    for match in pattern.finditer(block.text):
                        field_value = normalize_whitespace(match.group(1))
                        if not field_value:
                            continue
                        facts.append(
                            FactRecord(
                                fact_id=stable_id("fact", block.evidence_id, field_name, field_value),
                                doc_id=manifest.doc_id,
                                evidence_id=block.evidence_id,
                                fact_type="rule_extracted",
                                field_name=field_name,
                                field_value=field_value,
                                normalized_value=self._normalize_field_value(field_name, field_value),
                                confidence=0.8,
                                page=block.page,
                                metadata={
                                    "file_name": resource_file.fileName,
                                    "source_element_ids": block.source_element_ids,
                                },
                            )
                        )
        return facts

    def _normalize_field_value(self, field_name: str, value: str) -> str | None:
        value = normalize_whitespace(value)
        if field_name in {"uscc", "vehicle_vin"}:
            return value.upper().replace(" ", "")
        return value
