"""Step1: 通过远程 OCR API 执行表格单元格检测（按 BATCH_SIZE 分批，每批一次请求多图）

服务端 /predict/model 已内置方向检测与回正（use_orientation=true），
客户端直接发原始图片，从响应的 orientation 字段构建 orient_map，省去额外的方向检测请求。
检测结果直接以 dict 列表返回，不落盘，供 filter 步骤在内存中处理。
input_path 始终记录原始文件路径，下游 fill.py 解析文件名不受影响。
"""
import asyncio
import os
import time
from pathlib import Path

import httpx
from loguru import logger
from PIL import Image as PILImage
import io
from src.utils import json_utils

BATCH_SIZE = int(os.getenv("OCR_DETECT_BATCH_SIZE", "128"))
THRESHOLD = float(os.getenv("CELL_LOCATION_THRESHOLD", "0.85"))
CELL_LOCATION_API_BASE = os.getenv("CELL_LOCATION_API_BASE", "http://10.192.72.12:8082/api/v1/predict/model")
OCR_DETECT_TIMEOUT = float(os.getenv("OCR_DETECT_TIMEOUT", "120"))


def _get_image_size(raw: bytes) -> tuple[int, int]:
    """从原始图片字节读取 (width, height)，用于构建 orient_map。"""
    with PILImage.open(io.BytesIO(raw)) as im:
        return im.size  # (width, height)


async def _request_batch(client: httpx.AsyncClient, paths: list[Path]) -> tuple[list[dict], dict]:
    """一次请求多张原始图片，服务端负责方向检测与回正。

    返回 (detect_results, orient_map)：
      - detect_results: [{input_path, page_index, boxes}, ...]
      - orient_map: {stem -> {angle, orig_width, orig_height}}，仅含 angle!=0 的条目
    """
    if not paths:
        return [], {}

    t_read = time.perf_counter()
    raws = await asyncio.gather(*[asyncio.to_thread(p.read_bytes) for p in paths])
    logger.info("溯源 batch 读文件耗时: {:.2f}s, 文件数={}", time.perf_counter() - t_read, len(paths))

    files = [
        ("files", (p.name, raw, "image/png" if p.suffix.lower() == ".png" else "image/jpeg"))
        for p, raw in zip(paths, raws)
    ]
    t_post = time.perf_counter()
    resp = await client.post(
        CELL_LOCATION_API_BASE,
        files=files,
        data={"threshold": float(THRESHOLD), "use_orientation": "true"},
    )
    logger.info("溯源 batch 请求耗时: {:.2f}s, 文件数={}", time.perf_counter() - t_post, len(paths))
    resp.raise_for_status()
    t_content = time.perf_counter()
    content = resp.content
    content_elapsed = time.perf_counter() - t_content

    t_json = time.perf_counter()
    body = json_utils.loads(content or b"null")
    json_elapsed = time.perf_counter() - t_json

    t_data = time.perf_counter()
    if not isinstance(body, dict):
        logger.warning("溯源 API 返回非对象响应: {}", type(body).__name__)
        data_list = []
    else:
        status = body.get("status")
        if status and status != "success":
            logger.warning("溯源 API 返回非 success 状态: status={}, task_id={}", status, body.get("task_id"))
        raw_data = body.get("data", [])
        data_list = raw_data if isinstance(raw_data, list) else []
    total_boxes = sum(
        len(item.get("boxes", []))
        for item in data_list
        if isinstance(item, dict)
    )
    data_elapsed = time.perf_counter() - t_data
    logger.info(
        "溯源 batch 响应解析详情 | content获取={:.2f}s json反序列化={:.2f}s data统计={:.2f}s size={:.2f}MB 文件数={} data数={} boxes数={} content_length={} content_encoding={} transfer_encoding={}",
        content_elapsed,
        json_elapsed,
        data_elapsed,
        len(content) / 1024 / 1024,
        len(paths),
        len(data_list),
        total_boxes,
        resp.headers.get("content-length"),
        resp.headers.get("content-encoding"),
        resp.headers.get("transfer-encoding"),
    )
    if len(data_list) < len(paths):
        logger.warning(
            f"溯源 API 返回 data 数量({len(data_list)}) < 请求图片数({len(paths)})，"
            f"多出的 {len(paths) - len(data_list)} 张将得到空 boxes"
        )

    # 从响应里提取方向信息，构建 orient_map（只记录 angle!=0 的条目）
    # 需要原始图片尺寸用于后续逆旋转，对有旋转的图片读一次尺寸
    t_orientation = time.perf_counter()
    orient_map: dict = {}
    for i, p in enumerate(paths):
        if i >= len(data_list):
            break
        angle = int(data_list[i].get("orientation", 0))
        if angle != 0:
            orig_w, orig_h = _get_image_size(raws[i])
            orient_map[p.stem] = {"angle": angle, "orig_width": orig_w, "orig_height": orig_h}
    logger.info(
        "溯源 batch 方向信息处理耗时: {:.2f}s, 文件数={}, 旋转图片数={}",
        time.perf_counter() - t_orientation,
        len(paths),
        len(orient_map),
    )

    detect_results = [
        {
            "input_path": str(p),
            "page_index": None,
            "boxes": data_list[i].get("boxes", []) if i < len(data_list) else [],
        }
        for i, p in enumerate(paths)
    ]
    return detect_results, orient_map


async def execute(images=None) -> tuple[list[dict], dict]:
    """检测图片中的表格单元格，返回内存结果，不落盘。

    返回 (detect_results, orient_map)：
      - detect_results: [{input_path, page_index, boxes}, ...]
      - orient_map: {stem -> {angle, orig_width, orig_height}}，仅含 angle!=0 的条目
    """
    if not images:
        return [], {}
    jpgs = [Path(p) for p in images]
    if not jpgs:
        return [], {}

    batches = [jpgs[i: i + BATCH_SIZE] for i in range(0, len(jpgs), BATCH_SIZE)]

    async def _one_batch(batch):
        t0 = time.perf_counter()
        results, batch_orient_map = await _request_batch(client, batch)
        logger.info(f"溯源一个batch结束，处理 {len(batch)} 个图，耗时 {time.perf_counter() - t0:.2f}s")
        return results, batch_orient_map

    async with httpx.AsyncClient(timeout=OCR_DETECT_TIMEOUT) as client:
        t0 = time.perf_counter()
        batch_results = await asyncio.gather(*[_one_batch(b) for b in batches])
        logger.info(f"溯源模型调用耗时: {time.perf_counter() - t0:.2f}s, 图片数={len(jpgs)}, 批次数={len(batches)}")

    detect_results = [r for results, _ in batch_results for r in results]
    orient_map = {}
    for _, bom in batch_results:
        orient_map.update(bom)
    return detect_results, orient_map
