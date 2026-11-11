from __future__ import annotations

from typing import Any, Dict, List

from ..schemas import ResourceFile, item_content_extract
from ..utils import looks_empty


class MaterialCleaner:
    def clean_file(self, resource_file: ResourceFile) -> List[Dict[str, Any]]:
        cleaned: List[Dict[str, Any]] = []
        for item in resource_file.content:
            item_type = getattr(item, "type", None)
            if item_type == "page_number":
                continue
            text = item_content_extract(item) or ""
            if item_type != "image" and looks_empty(text):
                continue
            cleaned.append(
                {
                    "item": item,
                    "element_id": getattr(item, "element_id", None),
                    "page": getattr(item, "page", None),
                    "type": item_type,
                    "bbox": self._normalize_bbox(getattr(item, "bBox", None)),
                    "text": text,
                    "md": getattr(item, "md", None),
                    "html": getattr(item, "html", None),
                }
            )
        return cleaned

    @staticmethod
    def _normalize_bbox(value: Any) -> Dict[str, Any] | None:
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump()
        return None
