"""
图片类型识别与印章标记服务

流程：
1. 从 MinIO 加载 converted.json (items)
2. 提取所有 type=="image" 且有 img_path 的元素
3. 从 MinIO 下载对应图片
4. 调用 /api/v1/predict/layout 接口批量识别图片类型
5. 将 label_types 包含 "seal" 的 image 元素 type 改为 "seal"
6. 返回处理后的 items
"""
import os
from typing import Any, Dict, List, Tuple

import httpx
from loguru import logger

from src.applications.parsing.utils.mineru_storage import (
    MINERU_OUTPUT_BUCKET,
    load_mineru_items,
)
from src.base.storage_service import storage_service

# 图片类型识别服务地址
LAYOUT_PREDICT_API_BASE = os.environ.get(
    "LAYOUT_PREDICT_API_BASE",
    "http://10.192.72.12:8082/api/v1/predict/layout",
)

# 请求超时（秒）
LAYOUT_PREDICT_TIMEOUT = int(os.environ.get("LAYOUT_PREDICT_TIMEOUT", "120"))


class ImageClassifyService:
    """图片类型识别与印章标记服务"""

    async def tag_seals(self, file_id: str) -> Dict[str, Any]:
        """对 MinerU 解析结果中的 image 元素进行印章识别并标记 type。

        Args:
            file_id: MinerU 解析结果的 file_id。

        Returns:
            包含处理统计和 items 的字典。
        """
        # 1. 加载 converted.json
        items = await load_mineru_items(file_id)
        logger.info("加载解析结果 | file_id={}, 元素数={}", file_id, len(items))

        # 2. 提取 image 元素
        image_items = [
            it for it in items
            if isinstance(it, dict)
            and it.get("type") == "image"
            and it.get("img_path")
        ]

        if not image_items:
            logger.info("未发现 image 元素，跳过印章识别 | file_id={}", file_id)
            return {
                "file_id": file_id,
                "seal_element_ids": [],
                "items": items,
            }

        logger.info("发现 image 元素 | file_id={}, count={}", file_id, len(image_items))

        # 3. 下载图片
        downloaded = await self._download_images(file_id, image_items)
        if not downloaded:
            logger.warning("所有图片下载失败 | file_id={}", file_id)
            return {
                "file_id": file_id,
                "seal_element_ids": [],
                "items": items,
            }

        # 4. 调用类型识别接口
        id_to_labels = await self._classify_images(downloaded)

        # 5. 根据结果改 type
        seal_element_ids: List[int] = []
        for it in image_items:
            eid = str(it.get("element_id", ""))
            labels = id_to_labels.get(eid, [])
            if "seal" in labels:
                it["type"] = "seal"
                seal_element_ids.append(it.get("element_id"))
                logger.info("标记为印章 | file_id={}, eid={}, labels={}", file_id, eid, labels)

        logger.info(
            "印章识别完成 | file_id={}, seal_count={}",
            file_id, len(seal_element_ids),
        )

        return {
            "file_id": file_id,
            "seal_element_ids": seal_element_ids,
            "items": items,
        }

    async def _download_images(
        self,
        file_id: str,
        image_items: List[Dict[str, Any]],
    ) -> List[Tuple[str, bytes]]:
        """从 MinIO 下载所有 image 元素对应的图片。

        Returns:
            (element_id, image_bytes) 列表。
        """
        downloaded: List[Tuple[str, bytes]] = []
        for it in image_items:
            eid = str(it.get("element_id", ""))
            img_path = it.get("img_path", "")
            object_name = f"{file_id}/{img_path}"
            try:
                data = await storage_service.download(MINERU_OUTPUT_BUCKET, object_name)
                downloaded.append((eid, data))
            except Exception:
                logger.exception("下载图片失败 | file_id={}, eid={}, path={}", file_id, eid, img_path)

        return downloaded

    async def _classify_images(
        self,
        downloaded: List[Tuple[str, bytes]],
    ) -> Dict[str, List[str]]:
        """调用 /api/v1/predict/layout 接口批量识别图片类型。

        Returns:
            element_id -> label_types 的映射。
        """
        ids = ",".join(eid for eid, _ in downloaded)
        files = [
            ("files", (f"{eid}.jpg", data, "image/jpeg"))
            for eid, data in downloaded
        ]
        data_payload = {"ids": ids}

        logger.info("调用类型识别接口 | url={}, ids={}", LAYOUT_PREDICT_API_BASE, ids)

        async with httpx.AsyncClient(timeout=LAYOUT_PREDICT_TIMEOUT) as client:
            resp = await client.post(LAYOUT_PREDICT_API_BASE, files=files, data=data_payload)

        if resp.status_code != 200:
            logger.error("类型识别接口返回错误 | status={}, body={}", resp.status_code, resp.text[:500])
            return {}

        results = resp.json()
        if not isinstance(results, list):
            logger.error("类型识别接口返回格式异常 | result={}", str(results)[:500])
            return {}

        id_to_labels: Dict[str, List[str]] = {}
        for r in results:
            if not isinstance(r, dict):
                continue
            rid = str(r.get("id", ""))
            labels = r.get("label_types", [])
            if isinstance(labels, list):
                id_to_labels[rid] = labels

        logger.info("类型识别完成 | results={}", id_to_labels)
        return id_to_labels
