"""
ADE Parse 服务层

职责：
  1. 接收上传文件，持久化到 MinIO，并创建短生命周期临时文件
  2. 调用 MinerU 解析服务得到 items
  3. 将 MinerU items 转换为 ADE ParseResponse 结构
"""

import mimetypes
import os
import tempfile
import time
import uuid
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from src.ade.parse.schema.parse_response_schema import (
    NonTableCellGroundingType,
    ParseChunk,
    ParseChunkGrounding,
    ParseChunkType,
    ParseGroundingBox,
    ParseMetadata,
    ParseResponse,
    ParseResponseGrounding,
    ParseResponseGroundingEntry,
    ParseResponseTableCellGrounding,
    ParseResponseTableCellGroundingPosition,
    ParseSplit,
)
from src.applications.parsing.service.parse_minerU_service import MinerUParsingService
from src.applications.parsing.utils.mineru_storage import (
    extract_pdf_info_from_mineru_result,
    upload_original_file,
)

# MinerU item.type → ParseChunkType
_ITEM_TYPE_TO_CHUNK_TYPE: Dict[str, ParseChunkType] = {
    "text": ParseChunkType.text,
    "table": ParseChunkType.table,
    "image": ParseChunkType.figure,
    "list": ParseChunkType.text,
    "page_number": ParseChunkType.page_number,
    "title": ParseChunkType.title,
    "form": ParseChunkType.form,
}

# ParseChunkType → NonTableCellGroundingType
_CHUNK_TYPE_TO_GROUNDING_TYPE: Dict[ParseChunkType, NonTableCellGroundingType] = {
    ParseChunkType.text: NonTableCellGroundingType.chunkText,
    ParseChunkType.table: NonTableCellGroundingType.chunkTable,
    ParseChunkType.figure: NonTableCellGroundingType.chunkFigure,
    ParseChunkType.title: NonTableCellGroundingType.chunkTitle,
    ParseChunkType.marginalia: NonTableCellGroundingType.chunkMarginalia,
    ParseChunkType.form: NonTableCellGroundingType.chunkForm,
    ParseChunkType.logo: NonTableCellGroundingType.chunkLogo,
    ParseChunkType.card: NonTableCellGroundingType.chunkCard,
    ParseChunkType.attestation: NonTableCellGroundingType.chunkAttestation,
    ParseChunkType.scan_code: NonTableCellGroundingType.chunkScanCode,
    ParseChunkType.page_header: NonTableCellGroundingType.chunkPageHeader,
    ParseChunkType.page_footer: NonTableCellGroundingType.chunkPageFooter,
    ParseChunkType.page_number: NonTableCellGroundingType.chunkPageNumber,
    ParseChunkType.key_value: NonTableCellGroundingType.chunkKeyValue,
}


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _extract_bbox(item: Dict[str, Any]) -> ParseGroundingBox:
    """从 MinerU item 的 bBox 字段提取边界框。"""
    bbox = item.get("bBox") or item.get("bbox") or {}
    if isinstance(bbox, dict):
        return ParseGroundingBox(
            left=_safe_float(bbox.get("l")),
            top=_safe_float(bbox.get("t")),
            right=_safe_float(bbox.get("r")),
            bottom=_safe_float(bbox.get("b")),
        )
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return ParseGroundingBox(
            left=_safe_float(bbox[0]),
            top=_safe_float(bbox[1]),
            right=_safe_float(bbox[2]),
            bottom=_safe_float(bbox[3]),
        )
    return ParseGroundingBox(left=0, top=0, right=0, bottom=0)


def _resolve_chunk_type(item: Dict[str, Any]) -> ParseChunkType:
    """根据 MinerU item 确定 ParseChunkType。"""
    raw_type = str(item.get("type", "text")).lower().strip()
    heading_level = item.get("headingLevel") or item.get("textLevel")
    if heading_level is not None and _safe_int(heading_level, 0) > 0 and raw_type == "text":
        return ParseChunkType.title
    return _ITEM_TYPE_TO_CHUNK_TYPE.get(raw_type, ParseChunkType.text)


def _get_page_dimensions(
    page_sizes: Optional[List[Dict[str, Any]]],
    page_idx: int,
) -> Tuple[float, float]:
    """获取指定页的宽高（默认 A4: 595×842 pt）。"""
    if page_sizes and 0 <= page_idx < len(page_sizes):
        ps = page_sizes[page_idx]
        if isinstance(ps, dict):
            page_size = ps.get("page_size")
            if isinstance(page_size, (list, tuple)) and len(page_size) >= 2:
                return _safe_float(page_size[0], 595.0), _safe_float(page_size[1], 842.0)
            return (
                _safe_float(ps.get("width") or ps.get("w"), 595.0),
                _safe_float(ps.get("height") or ps.get("h"), 842.0),
            )
        if isinstance(ps, (list, tuple)) and len(ps) >= 2:
            return _safe_float(ps[0], 595.0), _safe_float(ps[1], 842.0)
    return 595.0, 842.0


def _build_response_markdown_segment(
    chunk_id: str,
    chunk_type: ParseChunkType,
    chunk_markdown: str,
    item: Dict[str, Any],
    page_0based: int,
) -> str:
    """
    组装 ParseResponse.markdown 的单段文本。

    目标格式：
    - 非表格：
      `chunk_id:{chunk_id}|chunk_type:{chunk_type}\n{chunk_text}\n`
    - 表格：
      `chunk_id:{chunk_id}|chunk_type:{chunk_type}\n<{cell_id}|{cell_text}><{cell_id}|{cell_text}>\n`
    """
    header = f"chunk_id:{chunk_id}|chunk_type:{chunk_type.value}\n"
    if chunk_type != ParseChunkType.table:
        text = (chunk_markdown or "").strip()
        return f"{header}{text}\n"

    rows = item.get("rows")
    if not isinstance(rows, list) or not rows:
        table_text = (chunk_markdown or "").strip()
        return f"{header}{table_text}\n"

    cell_parts: List[str] = []
    for row_idx, row in enumerate(rows):
        if not isinstance(row, list):
            continue
        for col_idx, cell in enumerate(row):
            if not isinstance(cell, dict):
                continue
            cell_id = f"{page_0based}-{chunk_id}-{row_idx}-{col_idx}"
            cell_text = str(cell.get("text") or "").strip()
            cell_parts.append(f"<{cell_id}|{cell_text}>")

    return f"{header}{''.join(cell_parts)}\n"


def _build_table_cell_groundings(
    chunk_id: str,
    item: Dict[str, Any],
    page_0based: int,
    chunk_box: ParseGroundingBox,
) -> Dict[str, ParseResponseTableCellGrounding]:
    """为表格的每个单元格生成 tableCell grounding 条目。"""
    rows = item.get("rows")
    if not isinstance(rows, list) or not rows:
        return {}

    entries: Dict[str, ParseResponseTableCellGrounding] = {}
    for row_idx, row in enumerate(rows):
        if not isinstance(row, list):
            continue
        for col_idx, cell in enumerate(row):
            if not isinstance(cell, dict):
                continue

            # 需要保证全局唯一，避免同页多表格的单元格 id 覆盖
            cell_id = f"{page_0based}-{chunk_id}-{row_idx}-{col_idx}"

            cell_box = ParseGroundingBox(
                left=chunk_box.left,
                top=chunk_box.top,
                right=chunk_box.right,
                bottom=chunk_box.bottom,
            )

            entries[cell_id] = ParseResponseTableCellGrounding(
                box=cell_box,
                page=page_0based,
                type="tableCell",
                confidence=None,
                markdown=str(cell.get("text") or "").strip() or None,
                position=ParseResponseTableCellGroundingPosition(
                    row=_safe_int(cell.get("rowstart", row_idx)),
                    col=_safe_int(cell.get("colstart", col_idx)),
                    rowspan=max(1, _safe_int(cell.get("rowspan"), 1)),
                    colspan=max(1, _safe_int(cell.get("colspan"), 1)),
                    chunk_id=chunk_id,
                ),
            )

    return entries


def convert_items_to_parse_response(
    items: List[Dict[str, Any]],
    filename: str,
    duration_ms: int,
    page_sizes: Optional[List[Dict[str, Any]]] = None,
    split_by_page: bool = False,
    file_id: Optional[str] = None,
    job_id: Optional[str] = None,
) -> ParseResponse:
    """
    将 MinerU 解析后的 items 列表转换为 ADE ParseResponse 结构。

    Args:
        items: MinerU 转换后的 item 列表
        filename: 原始文件名
        duration_ms: 解析耗时（毫秒）
        page_sizes: 每页 {width, height} 或 [width, height] 列表（从 middle_json.pdf_info 中获取）
        split_by_page: 是否按页分段
    """
    chunks: List[ParseChunk] = []
    grounding_map: Dict[str, ParseResponseGroundingEntry] = {}
    page_set: set[int] = set()
    page_chunks_map: Dict[int, List[str]] = {}
    markdown_segments: List[str] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        chunk_type = _resolve_chunk_type(item)
        chunk_id = str(uuid.uuid4())
        page_1based = _safe_int(item.get("page"), 1)
        page_0based = max(0, page_1based - 1)
        page_set.add(page_0based)

        md_text = item.get("md") or item.get("value") or ""

        bbox = _extract_bbox(item)
        chunk = ParseChunk(
            markdown=md_text,
            type=chunk_type,
            id=chunk_id,
            grounding=ParseChunkGrounding(box=bbox, page=page_0based),
        )
        chunks.append(chunk)

        markdown_segments.append(
            _build_response_markdown_segment(
                chunk_id=chunk_id,
                chunk_type=chunk_type,
                chunk_markdown=md_text,
                item=item,
                page_0based=page_0based,
            )
        )

        page_chunks_map.setdefault(page_0based, []).append(chunk_id)

        grounding_type = _CHUNK_TYPE_TO_GROUNDING_TYPE.get(
            chunk_type, NonTableCellGroundingType.chunkText
        )
        chunk_markdown = (md_text or item.get("value") or "").strip() or None

        grounding_map[chunk_id] = ParseResponseGrounding(
            box=bbox,
            page=page_0based,
            type=grounding_type,
            confidence=None,
            markdown=chunk_markdown,
            low_confidence_spans=[],
        )

        if chunk_type == ParseChunkType.table:
            cell_groundings = _build_table_cell_groundings(
                chunk_id, item, page_0based, bbox,
            )
            grounding_map.update(cell_groundings)

    splits: List[ParseSplit] = []
    if split_by_page:
        for page_idx in sorted(page_set):
            page_chunk_ids = page_chunks_map.get(page_idx, [])
            page_md_parts = [
                c.markdown for c in chunks
                if c.grounding.page == page_idx and c.markdown
            ]
            splits.append(
                ParseSplit(
                    class_="page",
                    identifier=f"page_{page_idx}",
                    pages=[page_idx],
                    markdown="\n\n".join(page_md_parts),
                    chunks=page_chunk_ids,
                )
            )

    page_count = max(page_set, default=0) + 1 if page_set else 0

    metadata = ParseMetadata(
        filename=filename,
        org_id=None,
        page_count=page_count,
        duration_ms=duration_ms,
        credit_usage=float(page_count),
        job_id=(job_id or str(uuid.uuid4())),
        file_id=file_id,
        version="mineru",
        failed_pages=[],
    )

    return ParseResponse(
        markdown="".join(markdown_segments),
        chunks=chunks,
        splits=splits,
        grounding=grounding_map if grounding_map else None,
        metadata=metadata,
    )


def _guess_content_type(filename: str) -> str:
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


class AdeParseService:
    """ADE Parse 同步解析服务：MinIO 持久化 → 临时文件解析 → ParseResponse。"""

    @staticmethod
    def _write_temp_file_sync(
        file_bytes: bytes,
        filename: str,
        file_id: str,
    ) -> Path:
        """写入解析器使用的短生命周期临时文件。"""
        ext = Path(filename).suffix
        fd, file_path = tempfile.mkstemp(prefix=f"{file_id}_", suffix=ext)
        try:
            with os.fdopen(fd, "wb") as buffer:
                buffer.write(file_bytes)
        except Exception:
            os.close(fd)
            raise
        return Path(file_path)

    async def parse_document(
        self,
        file_bytes: bytes,
        filename: str,
        split_by_page: bool = False,
        file_id: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> ParseResponse:
        """
        同步解析文档的完整流程。

        1. 原始文件上传 MinIO
        2. 写入短生命周期临时文件
        3. 调用 MinerU 解析得到 items
        4. 转换 items → ParseResponse
        """
        resolved_file_id = file_id or uuid.uuid4().hex
        content_type = _guess_content_type(filename)
        await upload_original_file(
            resolved_file_id,
            filename,
            file_bytes,
            content_type,
        )
        file_path = await asyncio.to_thread(
            self._write_temp_file_sync,
            file_bytes,
            filename,
            resolved_file_id,
        )
        logger.info(
            "ADE parse: file uploaded and temp file created, file_id={}, temp_path={}, size={}",
            resolved_file_id, file_path, len(file_bytes),
        )

        start_ms = time.time()
        try:
            try:
                items_paths = await MinerUParsingService().parse_to_items_and_save_async(
                    str(file_path),
                    file_id=resolved_file_id,
                    classify_tables=True,
                    original_filename=filename,
                )
            except Exception:
                logger.exception("MinerU parsing failed | file_id={}", resolved_file_id)
                raise

            items = items_paths.get("items")
            mineru_result = items_paths.get("mineru_result")

            duration_ms = int((time.time() - start_ms) * 1000)
            logger.info(
                "ADE parse: MinerU returned {} items in {}ms",
                len(items) if isinstance(items, list) else 0,
                duration_ms,
            )

            if not isinstance(items, list):
                items = []

            page_sizes = extract_pdf_info_from_mineru_result(
                mineru_result,
                file_id=resolved_file_id,
            )

            response = convert_items_to_parse_response(
                items=items,
                filename=filename,
                duration_ms=duration_ms,
                page_sizes=page_sizes,
                split_by_page=split_by_page,
                file_id=resolved_file_id,
                job_id=job_id,
            )

            return response
        finally:
            self._cleanup(file_path)

    @staticmethod
    def _cleanup(file_path: Path) -> None:
        """清理临时上传文件。"""
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            logger.warning("Failed to cleanup temp file {}: {}", file_path, e)
