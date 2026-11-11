"""
ADE 结构化提取 Service — VLM 模式（OCR 辅助）

流程：
  1. 文件渲染为图片
  2. OCR 识别（OpenAI 兼容接口）→ 文字 + 坐标
  3. VLM 看图 + OCR 编号列表 → 提取字段值 + ocrIndices
  4. 根据 ocrIndices 查 OCR bbox → 构建溯源 FieldReference
"""
import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from src.ade.extract.schema.extract_schema import (
    ExtractRule,
    ExtractionItem,
    FieldMeta,
    FieldReference,
)
from src.utils.model_factory import ModelFactory
from src.ade.extract.utils.vlm_utils import (
    PageImage,
    OCRItem,
    render_pages_from_bytes,
    image_bytes_to_data_url,
    build_resized_page_info,
    compact_schema,
    get_json_text,
    try_parse_json,
    split_ocr_indices,
    collect_fields_for_grounding,
    build_bbox_from_match,
    inject_bboxes_into_result,
    merge_leaf_values,
    parse_ocr_response,
    apply_orientation_to_pages,
    bbox_rotated_to_original,
)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

_OCR_BASE_URL = os.getenv("ADE_OCR_BASE_URL", "http://10.192.72.11:8083/v1")
_OCR_MODEL = os.getenv("ADE_OCR_MODEL", "HunyuanOCR")
_OCR_API_KEY = os.getenv("ADE_OCR_API_KEY", "EMPTY")
_ORIENTATION_API_URL = os.getenv(
    "ADE_ORIENTATION_API_URL", "http://10.192.72.11:8082/api/v1/predict/orientation"
)

_MAX_RETRIES = 3
_INITIAL_DELAY = 15.0
_BATCH_SIZE = 5


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class VlmExtractService:
    """ADE 结构化提取服务 — VLM 模式（OCR 辅助一步提取+对齐）"""

    def __init__(self):
        self.vlm_model = ModelFactory.get_model(
            key="doc_extract_vl",
            vision=True,
        )

    # ------------------------------------------------------------------ #
    # 公开入口
    # ------------------------------------------------------------------ #
    async def extract(
        self,
        file_bytes: bytes,
        filename: str,
        rules: List[ExtractRule],
    ) -> Dict[str, ExtractionItem]:
        """
        VLM 模式提取主入口。

        Returns:
            {extractName: ExtractionItem, ...}
        """
        # 1. 渲染页面（原始图片）
        orig_pages = await asyncio.to_thread(
            render_pages_from_bytes, file_bytes, filename, 160, 0
        )
        if not orig_pages:
            raise ValueError("未渲染出任何页面")
        logger.info("VLM 提取: 渲染 {} 页, 文件={}", len(orig_pages), filename)

        # 2. 方向检测 + 回正
        angles = await self._call_orientation_api(orig_pages)
        pages = await asyncio.to_thread(apply_orientation_to_pages, orig_pages, angles)

        # 3. OCR 识别（基于回正图）
        ocr_items = await self._call_ocr(pages)
        logger.info(f"VLM 提取: OCR 识别 {len(ocr_items)} 个文字块")

        # 3. 对每条规则：按批次 VLM 提取 + 溯源
        result: Dict[str, ExtractionItem] = {}
        for rule in rules:
            extract_output, match_result = await self._vlm_extract_rule_in_batches(
                pages, rule, ocr_items
            )

            # 4. 提取纯值（去掉 metadata 包装）
            result_node = extract_output.get("result")
            if not isinstance(result_node, dict):
                result_node = extract_output
            extracted = merge_leaf_values(result_node)
            if not isinstance(extracted, dict):
                extracted = {"_raw": extracted}

            # 5. 构建溯源 bbox（逆旋转回原图坐标），按 extraction 结构构建同构 metadata 树
            metadata = self._build_vlm_metadata(
                extract_output, match_result, ocr_items, orig_pages, angles,
                extracted,
            )

            result[rule.extractName] = ExtractionItem(
                extraction=extracted,
                extraction_metadata=metadata,
            )

        return result

    # ------------------------------------------------------------------ #
    # 方向检测
    # ------------------------------------------------------------------ #
    async def _call_orientation_api(self, pages: List[PageImage]) -> List[int]:
        """批量调用方向分类接口，返回每页旋转角度列表。

        score > 0.5 时取 orientation（0/90/180/270），否则视为 0。
        接口异常时降级返回全 0，不中断提取流程。
        """
        try:
            files = []
            for i, page in enumerate(pages):
                files.append(
                    ("files", (f"page_{i}.png", page.image_bytes, "image/png"))
                )
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(_ORIENTATION_API_URL, files=files)
                resp.raise_for_status()
                data = resp.json().get("data", [])
            angles: List[int] = []
            for item in data:
                score = item.get("score", 0.0)
                orientation = item.get("orientation", 0)
                angles.append(int(orientation) if score > 0.5 else 0)
            # 若返回数量与页数不一致，补全为 0
            while len(angles) < len(pages):
                angles.append(0)
            return angles[:len(pages)]
        except Exception as e:
            logger.warning(f"方向分类接口调用失败，降级为不旋转: {e}")
            return [0] * len(pages)

    # ------------------------------------------------------------------ #
    # OCR 调用
    # ------------------------------------------------------------------ #
    async def _call_ocr(self, pages: List[PageImage]) -> List[OCRItem]:
        """逐页调用 OCR OpenAI 兼容接口，带重试。"""
        all_items: List[OCRItem] = []

        for page in pages:
            data_url = await asyncio.to_thread(
                image_bytes_to_data_url, page.image_bytes, page.image_format
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                        {"type": "text", "text": "识别图片中的文字，并输出坐标"},
                    ],
                }
            ]
            payload = {
                "model": _OCR_MODEL,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 4096,
            }

            raw_text = ""
            for attempt in range(_MAX_RETRIES + 1):
                try:
                    async with httpx.AsyncClient(timeout=120) as client:
                        resp = await client.post(
                            f"{_OCR_BASE_URL}/chat/completions",
                            json=payload,
                            headers={"Authorization": f"Bearer {_OCR_API_KEY}"},
                        )
                        resp.raise_for_status()
                        body = resp.json()
                        raw_text = (
                            body.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                    break
                except Exception as e:
                    if attempt < _MAX_RETRIES:
                        delay = _INITIAL_DELAY * (2 ** attempt)
                        logger.warning(
                            f"OCR 调用失败（第 {attempt + 1} 次），{delay}s 后重试: {e}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"OCR 调用失败，已重试 {_MAX_RETRIES} 次: {e}")
                        raise

            page_items = await asyncio.to_thread(parse_ocr_response, raw_text, page)
            all_items.extend(page_items)
            logger.debug(
                f"OCR page {page.page_index + 1}: {len(page_items)} items"
            )

        return all_items

    # ------------------------------------------------------------------ #
    # VLM 提取 + OCR 索引对齐
    # ------------------------------------------------------------------ #
    async def _vlm_extract_rule_in_batches(
        self,
        pages: List[PageImage],
        rule: ExtractRule,
        ocr_items: List[OCRItem],
    ) -> Tuple[Dict[str, Any], Dict[str, List[int]]]:
        """按页批次提取；无顶层数组字段时，字段提取完整后提前停止。"""
        has_array = self._has_top_level_array(rule.jsonSchema)
        batches = [
            pages[i:i + _BATCH_SIZE]
            for i in range(0, len(pages), _BATCH_SIZE)
        ]
        logger.info(
            "ADE VLM 提取任务 {}：共 {} 页，分 {} 批，有数组字段: {}",
            rule.extractName,
            len(pages),
            len(batches),
            has_array,
        )

        merged_output: Dict[str, Any] = {"result": {}}
        merged_match: Dict[str, List[int]] = {}

        for batch_idx, batch_pages in enumerate(batches):
            batch_output, batch_match = await self._vlm_extract_with_alignment(
                batch_pages,
                rule,
                ocr_items,
            )
            merged_output, merged_match = self._merge_vlm_batch_result(
                merged_output,
                merged_match,
                batch_output,
                batch_match,
            )

            if not has_array:
                result_node = merged_output.get("result")
                if not isinstance(result_node, dict):
                    result_node = merged_output
                extracted = merge_leaf_values(result_node)
                if isinstance(extracted, dict):
                    missing = self._get_missing_fields(rule.jsonSchema, extracted)
                    if not missing:
                        logger.info(
                            "ADE VLM 提取任务 {}：第 {} 批后所有字段已提取完毕，提前结束",
                            rule.extractName,
                            batch_idx + 1,
                        )
                        break
                    logger.info(
                        "ADE VLM 提取任务 {}：第 {} 批完成，剩余未提取字段: {}",
                        rule.extractName,
                        batch_idx + 1,
                        missing,
                    )

        return merged_output, merged_match

    async def _vlm_extract_with_alignment(
        self,
        pages: List[PageImage],
        rule: ExtractRule,
        ocr_items: List[OCRItem],
    ) -> Tuple[Dict[str, Any], Dict[str, List[int]]]:
        """VLM 一步提取：看图 + OCR 编号列表 → 字段值 + ocrIndices。"""
        resized = await asyncio.to_thread(build_resized_page_info, pages)

        # 构建当前批次页面的 OCR 编号列表，编号保持全局 OCR 索引，便于后续溯源匹配
        page_numbers = {page.page_index for page in pages}
        ocr_index_lines = [
            f"[{i}] (页{item.page_index + 1}) {item.text}"
            for i, item in enumerate(ocr_items)
            if item.page_index in page_numbers
        ]
        ocr_list_str = "\n".join(ocr_index_lines)

        # 构建规则 prompt
        schema_obj = json.loads(rule.jsonSchema)
        compact = compact_schema(schema_obj)
        rule_prompt = json.dumps(
            {
                "extractName": rule.extractName,
                "extractPrompt": rule.extractPrompt or "",
                "schema": compact,
            },
            ensure_ascii=False,
            indent=2,
        )

        # 构建消息内容
        content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "请根据给定规则，从文档图片中提取字段值，并将每个字段对齐到 OCR 识别结果。\n\n"
                    f"OCR 识别结果（带编号）：\n{ocr_list_str}\n\n"
                    "每个叶子字段必须返回 {value, pageIndex, sourceText, ocrIndices}。\n"
                    "- value: 字段值，未找到则整个字段返回 null\n"
                    "- pageIndex: 所在页码，从 1 开始\n"
                    "- sourceText: 图中该字段对应的原始文字（仅包含该字段自身的文字，"
                    "不要混入相邻列或其他字段的文字。注意：文档可能存在左右分栏或多列布局，"
                    "请严格按字段的空间位置区分，只提取属于该字段区域内的文字）\n"
                    "- ocrIndices: 该字段值对应的 OCR 编号列表（从上面的编号中选取，"
                    "仅选取属于该字段自身区域的编号，不要选取相邻列或其他字段的编号）\n"
                    "- 只有当所选 ocrIndices 对应的 OCR 文本，与 value 或 sourceText 至少有一部分明确重合、"
                    "包含关系、或去掉空格/标点后仍能部分匹配时，才能返回这些 ocrIndices；"
                    "如果无法满足这个条件，ocrIndices 必须返回空数组 []，不要为了给框而强行选附近编号\n"
                    "- 如果 value 主要来自印章、手写、模糊区域、视觉判断，或者图片里能看见但 OCR 列表中没有对应文字，"
                    "则可以保留 value 和 sourceText，但 ocrIndices 必须为空 []\n"
                    "- 严禁因为版面接近、位于同一行/同一区域，就选择字段名、相邻字段、相邻列、登记机关、日期等无关 OCR 编号\n"
                    "如果字段是数组，请返回数组，数组中每个元素的叶子字段同样需要 "
                    "{value, pageIndex, sourceText, ocrIndices}。\n"
                    "不要返回任何解释，不要使用 Markdown。\n\n"
                    "返回结构要求：\n"
                    "{{\n"
                    '  "extractName": "规则名称",\n'
                    '  "result": 与 schema 同结构，叶子节点为 '
                    "{{value, pageIndex, sourceText, ocrIndices}} 或 null\n"
                    "}}\n\n"
                    f"提取规则：\n{rule_prompt}"
                ),
            }
        ]

        # 添加页面图片
        for img_bytes, fmt, rw, rh, sx, sy in resized:
            data_url = await asyncio.to_thread(image_bytes_to_data_url, img_bytes, fmt)
            content.append(
                {"type": "image_url", "image_url": {"url": data_url}}
            )

        messages = [
            SystemMessage(
                content="你是一个严谨的结构化字段提取与定位助手，只输出合法 JSON。"
            ),
            HumanMessage(content=content),
        ]

        # 调用 VLM，带重试
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await self.vlm_model.ainvoke(
                    messages, response_format={"type": "json_object"}
                )
                break
            except Exception as e:
                if attempt < _MAX_RETRIES:
                    delay = _INITIAL_DELAY * (2 ** attempt)
                    logger.warning(
                        f"VLM 调用失败（第 {attempt + 1} 次），{delay}s 后重试: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"VLM 调用失败，已重试 {_MAX_RETRIES} 次: {e}")
                    raise

        raw = try_parse_json(get_json_text(response.content))
        extract_output, match_result = split_ocr_indices(raw)
        return extract_output, match_result

    @staticmethod
    def _has_top_level_array(json_schema: str) -> bool:
        """检查 jsonSchema 顶层 properties 是否有 type=array 的字段。"""
        try:
            schema = json.loads(json_schema)
            properties = schema.get("properties", {})
            return any(prop.get("type") == "array" for prop in properties.values())
        except Exception:
            return False

    @staticmethod
    def _get_missing_fields(json_schema: str, extracted: Dict[str, Any]) -> List[str]:
        """返回 jsonSchema 顶层 properties 中缺失或为空的字段名列表。"""
        try:
            schema = json.loads(json_schema)
            properties = schema.get("properties", {})
            missing = []
            for field_name in properties:
                val = extracted.get(field_name)
                if val is None or val == "" or val == []:
                    missing.append(field_name)
            return missing
        except Exception:
            return []

    @classmethod
    def _merge_vlm_batch_result(
        cls,
        base_output: Dict[str, Any],
        base_match: Dict[str, List[int]],
        patch_output: Dict[str, Any],
        patch_match: Dict[str, List[int]],
    ) -> Tuple[Dict[str, Any], Dict[str, List[int]]]:
        """合并批次提取结果，已有非空字段不被后续空值覆盖。"""
        base_result = base_output.get("result")
        if not isinstance(base_result, dict):
            base_result = base_output
        patch_result = patch_output.get("result")
        if not isinstance(patch_result, dict):
            patch_result = patch_output

        remapped_patch_match = {
            cls._shift_match_path_for_appended_arrays(base_result, patch_result, key): indices
            for key, indices in patch_match.items()
        }
        merged_result = cls._merge_result_node(base_result, patch_result)
        merged_match = dict(base_match)
        for key, indices in remapped_patch_match.items():
            if not cls._is_empty_value(cls._get_node_by_path(base_result, key)):
                continue
            if cls._get_node_by_path(merged_result, key) in (None, "", []):
                continue
            merged_match.setdefault(key, [])
            for idx in indices:
                if idx not in merged_match[key]:
                    merged_match[key].append(idx)

        return {"result": merged_result}, merged_match

    @classmethod
    def _merge_result_node(cls, base: Any, patch: Any) -> Any:
        if cls._is_empty_value(base):
            return patch
        if cls._is_empty_value(patch):
            return base
        if cls._is_field_leaf(base) and cls._is_field_leaf(patch):
            return base if not cls._is_empty_value(base.get("value")) else patch
        if isinstance(base, dict) and isinstance(patch, dict):
            result = dict(base)
            for key, patch_value in patch.items():
                result[key] = cls._merge_result_node(result.get(key), patch_value)
            return result
        if isinstance(base, list) and isinstance(patch, list):
            return base + patch
        return base

    @staticmethod
    def _is_field_leaf(node: Any) -> bool:
        return isinstance(node, dict) and {"value", "pageIndex", "sourceText"}.issubset(node.keys())

    @staticmethod
    def _is_empty_value(value: Any) -> bool:
        if value is None or value == "" or value == []:
            return True
        if isinstance(value, dict) and {"value", "pageIndex", "sourceText"}.issubset(value.keys()):
            return value.get("value") is None or value.get("value") == "" or value.get("value") == []
        return False

    @staticmethod
    def _get_node_by_path(node: Any, path: str) -> Any:
        current = node
        for token in path.split(".") if path else []:
            if isinstance(current, dict):
                current = current.get(token)
            elif isinstance(current, list) and token.isdigit():
                idx = int(token)
                current = current[idx] if 0 <= idx < len(current) else None
            else:
                return None
        if isinstance(current, dict) and {"value", "pageIndex", "sourceText"}.issubset(current.keys()):
            return current.get("value")
        return current

    @classmethod
    def _shift_match_path_for_appended_arrays(cls, base: Any, patch: Any, path: str) -> str:
        tokens = path.split(".") if path else []
        shifted: List[str] = []
        base_current = base
        patch_current = patch

        for token in tokens:
            if isinstance(patch_current, dict):
                shifted.append(token)
                patch_current = patch_current.get(token)
                base_current = base_current.get(token) if isinstance(base_current, dict) else None
                continue

            if isinstance(patch_current, list) and token.isdigit():
                idx = int(token)
                offset = len(base_current) if isinstance(base_current, list) else 0
                shifted.append(str(offset + idx))
                patch_current = patch_current[idx] if 0 <= idx < len(patch_current) else None
                base_current = None
                continue

            shifted.append(token)
            patch_current = None
            base_current = None

        return ".".join(shifted)

    # ------------------------------------------------------------------ #
    # 构建溯源 metadata
    # ------------------------------------------------------------------ #
    def _build_vlm_metadata(
        self,
        extract_output: Dict[str, Any],
        match_result: Dict[str, List[int]],
        ocr_items: List[OCRItem],
        orig_pages: List[PageImage],
        angles: List[int],
        extracted: Dict[str, Any],
    ) -> Dict[str, Any]:
        """从 VLM 提取结果 + OCR match 构建溯源 metadata（与 extraction 同构嵌套树）。

        bbox 坐标经逆旋转后换算回原图坐标系（PDF 为 points，图片为像素）。
        """
        result_node = extract_output.get("result")
        if not isinstance(result_node, dict):
            result_node = extract_output

        # 收集需要溯源的字段
        fields_to_ground: List[Dict[str, Any]] = []
        collect_fields_for_grounding(result_node, [], fields_to_ground)

        # 构建 bbox_map（坐标基于回正图像素）
        bbox_map = build_bbox_from_match(fields_to_ground, ocr_items, match_result)

        # 逆旋转 + 换算到原图坐标系（PDF points 或图片像素）
        field_page_index: Dict[str, int] = {
            f["field"]: (f.get("pageIndex") or 1) for f in fields_to_ground
        }
        for field_path, bbox in list(bbox_map.items()):
            page_idx = field_page_index.get(field_path, 1) - 1  # 1-based → 0-based
            if 0 <= page_idx < len(orig_pages):
                orig_page = orig_pages[page_idx]
                angle = angles[page_idx] if page_idx < len(angles) else 0
                rotated_back = bbox_rotated_to_original(
                    bbox, angle, orig_page.width, orig_page.height
                )
                if orig_page.orig_width and orig_page.orig_height:
                    sx = orig_page.orig_width / orig_page.width
                    sy = orig_page.orig_height / orig_page.height
                    x1, y1, x2, y2 = rotated_back
                    bbox_map[field_path] = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
                else:
                    bbox_map[field_path] = rotated_back

        # 注入 bbox 到结果树
        inject_bboxes_into_result(result_node, [], bbox_map)

        # 组装嵌套 metadata 树（与 extraction 同构，叶子节点为 FieldMeta）
        # 先构建扁平 path -> FieldMeta 映射
        flat_meta: Dict[str, FieldMeta] = {}
        for field_info in fields_to_ground:
            field_path = field_info["field"]
            field_value = field_info["value"]
            page_index = field_info.get("pageIndex")
            bbox = bbox_map.get(field_path)
            flat_meta[field_path] = FieldMeta(
                value=field_value,
                references=FieldReference(bbox=bbox, pageIndex=page_index),
            )

        # 按 extraction 结构递归构建同构 metadata 树
        meta_tree = self._build_meta_tree(extracted, flat_meta, [])
        if not isinstance(meta_tree, dict):
            meta_tree = {}
        return meta_tree

    @staticmethod
    def _build_meta_tree(
        ext_node: Any,
        flat_meta: Dict[str, "FieldMeta"],
        path_tokens: List[str],
    ) -> Any:
        """按 extraction 结构递归构建同构 metadata 树。"""
        path_key = ".".join(path_tokens)

        if path_key in flat_meta:
            return flat_meta[path_key]

        if isinstance(ext_node, dict):
            result = {}
            for k, v in ext_node.items():
                result[k] = VlmExtractService._build_meta_tree(
                    v, flat_meta, path_tokens + [k]
                )
            return result

        if isinstance(ext_node, list):
            result_list = []
            for i, item in enumerate(ext_node):
                result_list.append(
                    VlmExtractService._build_meta_tree(
                        item, flat_meta, path_tokens + [str(i)]
                    )
                )
            return result_list

        return None
