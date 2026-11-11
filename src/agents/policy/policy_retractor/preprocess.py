import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    from .schemas import NormalizedChunk, PolicyDocument
except ImportError:  # pragma: no cover
    from schemas import NormalizedChunk, PolicyDocument


class PolicyDocumentLoader:
    """Load MinerU output and normalize it into model-ready chunks."""
    MAX_TABLE_ROWS_PER_CHUNK = 20
    MAX_TABLE_CHARS_PER_CHUNK = 2200

    def load(self, input_path: str | Path) -> PolicyDocument:
        """Load a single MinerU json file or pick one from a directory."""
        path = Path(input_path)
        if path.is_dir():
            path = self._resolve_directory_input(path)
        return self._load_file(path)

    def load_payload(self, payload: Any) -> PolicyDocument:
        """Load a Java-provided JSON payload directly into a PolicyDocument."""
        return self._load_payload(payload)

    def _resolve_directory_input(self, directory: Path) -> Path:
        """Resolve directory input to the first supported MinerU json file."""
        files = sorted(directory.glob("*_mineru_converted.json"))
        if not files:
            raise FileNotFoundError(f"No supported MinerU json found under {directory}")
        return files[0]

    def _load_file(self, path: Path) -> PolicyDocument:
        """Load and normalize one MinerU json file into a PolicyDocument."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        return self._load_payload(payload, source_path=str(path))

    def _load_payload(self, payload: Any, source_path: str = "[java-payload]") -> PolicyDocument:
        """Load and normalize one in-memory payload into a PolicyDocument."""
        items = self._extract_document_items(payload)
        chunks = self._normalize_raw(items)

        file_name = self._extract_file_name(payload) or "java_payload.json"
        document_id = self._extract_document_id(payload) or "UNKNOWN_DOC"
        return PolicyDocument(
            document_id=document_id,
            file_name=file_name,
            source_path=source_path,
            chunks=self._compact_chunks(chunks),
        )

    def _extract_document_items(self, payload: Any) -> list[dict[str, Any]]:
        """Extract MinerU-like element list from supported payload shapes."""
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            doc_content = payload.get("docContent")
            if isinstance(doc_content, list):
                return [item for item in doc_content if isinstance(item, dict)]
        raise ValueError("Unsupported policy payload shape: expected list or object with docContent")

    def _extract_document_id(self, payload: Any) -> Optional[str]:
        """Try to extract upstream file id before falling back to file name."""
        if isinstance(payload, dict):
            for key in ("fileId", "file_id", "document_id", "documentId"):
                value = payload.get(key)
                if value:
                    return str(value).strip()

        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                for container in (item, item.get("metadata", {})):
                    if not isinstance(container, dict):
                        continue
                    for key in ("fileId", "file_id", "document_id", "documentId"):
                        value = container.get(key)
                        if value:
                            return str(value).strip()
        return None

    def _extract_file_name(self, payload: Any) -> Optional[str]:
        """Try to extract file name from MinerU metadata payload."""
        if isinstance(payload, list):
            for item in payload:
                metadata = item.get("metadata", {})
                if isinstance(metadata, dict) and metadata.get("file_name"):
                    return metadata["file_name"]
                if item.get("file_name"):
                    return item["file_name"]
        return None

    def _normalize_raw(self, items: list[dict[str, Any]]) -> list[NormalizedChunk]:
        """Convert raw MinerU elements to normalized chunks with heading context."""
        chunks: list[NormalizedChunk] = []
        heading_stack: dict[int, str] = {}
        order = 0

        for item in items:
            item_type = item.get("type", "text")
            if item_type == "page_number":
                continue
            text = (item.get("value") or item.get("md") or "").strip()
            heading_level = item.get("headingLevel")

            if heading_level and text:
                heading_stack[heading_level] = text  # 若新标题出现，取代现在相同层级的标题
                # Drop deeper heading levels when a higher level appears.
                for level in list(heading_stack):
                    if level > heading_level:
                        del heading_stack[level]

            heading_path = [heading_stack[level] for level in sorted(heading_stack)]
            title = text if heading_level else (heading_path[-1] if heading_path else None)

            if item_type == "table":
                rows = item.get("rows") or []
                table_caption = self._resolve_table_caption(item, items)
                table_segments = self._split_table_rows(
                    rows,
                    max_rows=self.MAX_TABLE_ROWS_PER_CHUNK,
                    max_chars=self.MAX_TABLE_CHARS_PER_CHUNK,
                )
                if not table_segments:
                    table_segments = [rows]
                chunk_type = "table"
            else:
                rows = []
                table_caption = None
                chunk_text = text
                chunk_type = "heading" if heading_level else "text"

            if item_type == "table":
                element_id = item.get("element_id", order)
                source_child_id = str(item.get("element_id", order))
                for segment_idx, segment_rows in enumerate(table_segments, start=1):
                    markdown_table = self._rows_to_markdown(segment_rows)
                    chunk_text = markdown_table or text
                    if not chunk_text:
                        continue
                    segment_suffix = (
                        f":part{segment_idx:02d}" if len(table_segments) > 1 else ""
                    )
                    chunks.append(
                        NormalizedChunk(
                            chunk_id=f"element:{element_id}{segment_suffix}",
                            source_child_id=source_child_id,
                            page=item.get("page"),
                            order=order,
                            chunk_type=chunk_type,
                            heading_path=heading_path,
                            title=title,
                            text=chunk_text,
                            table_caption=table_caption,
                            table_rows=segment_rows,
                            raw_md=item.get("md"),
                        )
                    )
                    order += 1
            else:
                if not chunk_text:
                    continue
                chunks.append(
                    NormalizedChunk(
                        chunk_id=f"element:{item.get('element_id', order)}",
                        source_child_id=str(item.get("element_id", order)),
                        page=item.get("page"),
                        order=order,
                        chunk_type=chunk_type,
                        heading_path=heading_path,
                        title=title,
                        text=chunk_text,
                        table_caption=table_caption,
                        table_rows=rows,
                        raw_md=item.get("md"),
                    )
                )
                order += 1
        return chunks

    def _compact_chunks(self, chunks: Iterable[NormalizedChunk]) -> list[NormalizedChunk]:
        """Sort and clean chunks, then drop empty text chunks."""
        compacted: list[NormalizedChunk] = []
        for chunk in sorted(chunks, key=lambda item: (item.page or 0, item.order, item.chunk_id)):
            text = self._clean_text(chunk.text)
            if not text:
                continue
            compacted.append(chunk.model_copy(update={"text": text}))
        return compacted

    def _clean_text(self, text: str) -> str:
        """Normalize text artifacts produced by OCR / conversion layers."""
        text = text.replace("\r\n", "\n")
        text = text.lstrip("\ufeff")
        text = re.sub(r"^[\u200b\u200c\u200d\u2060]+", "", text)
        text = re.sub(r"^\[H\d+\]\s*[^\n]+\n?", "", text)
        text = re.sub(r"(?m)^\s*-\d+\s*-\s*$", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _rows_to_markdown(self, rows: Any) -> str:
        """Convert MinerU table rows to markdown (coordinate-first strategy).

        Preferred path:
        - Use ``rowstart``/``colstart`` as authoritative coordinates.
        - Expand merged cells with ``rowspan``/``colspan``.

        Fallback path:
        - When coordinates are missing/invalid/colliding, place the cell by
          row-order and first available column on the current physical row.
        """
        if not isinstance(rows, list):
            return ""

        grid: dict[tuple[int, int], str] = {}
        occupied: set[tuple[int, int]] = set()
        max_row = -1
        max_col = -1

        for physical_row_idx, row in enumerate(rows):
            if isinstance(row, (str, bytes)):
                row = [row]
            elif not isinstance(row, list):
                row = [row]

            col_cursor = 0
            for cell in row:
                while (physical_row_idx, col_cursor) in occupied:
                    col_cursor += 1

                if isinstance(cell, dict):
                    row_start = self._safe_int(cell.get("rowstart"), default=physical_row_idx, minimum=0)
                    col_start = self._safe_int(cell.get("colstart"), default=col_cursor, minimum=0)
                    rowspan = self._safe_int(cell.get("rowspan"), default=1, minimum=1)
                    colspan = self._safe_int(cell.get("colspan"), default=1, minimum=1)
                else:
                    row_start = physical_row_idx
                    col_start = col_cursor
                    rowspan = 1
                    colspan = 1

                text = self._normalize_table_cell(cell)

                target_row = row_start
                target_col = col_start
                if self._area_has_collision(occupied, target_row, target_col, rowspan, colspan):
                    # Fallback for malformed/inconsistent coordinates.
                    target_row = physical_row_idx
                    target_col = self._first_free_col(occupied, target_row, col_cursor, colspan)

                for rr in range(target_row, target_row + rowspan):
                    for cc in range(target_col, target_col + colspan):
                        occupied.add((rr, cc))
                        if rr == target_row and cc == target_col:
                            grid[(rr, cc)] = text
                        elif rr > target_row and cc == target_col:
                            # Propagate vertical merged labels to preserve row semantics.
                            grid[(rr, cc)] = text
                        else:
                            grid[(rr, cc)] = ""
                        max_row = max(max_row, rr)
                        max_col = max(max_col, cc)

                if target_row == physical_row_idx:
                    col_cursor = max(col_cursor, target_col + colspan)

        if max_row < 0 or max_col < 0:
            return ""

        matrix: list[list[str]] = []
        for row_idx in range(max_row + 1):
            width = max_col + 1
            rendered_row = [grid.get((row_idx, col), "") for col in range(width)]
            if any(rendered_row):
                matrix.append(rendered_row)

        if not matrix:
            return ""

        header = matrix[0]
        divider = ["---"] * width
        body = matrix[1:]
        markdown_lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(divider) + " |",
        ]
        markdown_lines.extend("| " + " | ".join(row) + " |" for row in body)
        return "\n".join(markdown_lines)

    def _area_has_collision(
        self,
        occupied: set[tuple[int, int]],
        row_start: int,
        col_start: int,
        rowspan: int,
        colspan: int,
    ) -> bool:
        """Return True if target area overlaps existing occupied cells."""
        for rr in range(row_start, row_start + rowspan):
            for cc in range(col_start, col_start + colspan):
                if (rr, cc) in occupied:
                    return True
        return False

    def _first_free_col(
        self,
        occupied: set[tuple[int, int]],
        row_idx: int,
        start_col: int,
        colspan: int,
    ) -> int:
        """Find first column on a row that can fit a cell with given colspan."""
        candidate = max(start_col, 0)
        while self._area_has_collision(occupied, row_idx, candidate, 1, colspan):
            candidate += 1
        return candidate

    def _safe_int(self, value: Any, default: int, minimum: int) -> int:
        """Parse int safely with lower-bound constraint."""
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed >= minimum else default

    def _normalize_table_cell(self, cell: Any) -> str:
        """Normalize one table cell into a markdown-safe single-line string."""
        if isinstance(cell, dict):
            raw = (
                cell.get("text")
                or cell.get("value")
                or cell.get("md")
                or cell.get("html")
                or ""
            )
        elif cell is None:
            raw = ""
        else:
            raw = cell

        text = str(raw).replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\s*\n\s*", " ", text)
        text = re.sub(r"[ \t]{2,}", " ", text).strip()
        # Escape markdown pipe to keep table column boundaries stable.
        return text.replace("|", r"\|")

    def _resolve_table_caption(self, item: dict[str, Any], items: list[dict[str, Any]]) -> Optional[str]:
        """Resolve table caption by preferring OCR-provided table_caption field."""
        raw_caption = item.get("table_caption")
        if isinstance(raw_caption, list):
            joined = " ".join(str(part).strip() for part in raw_caption if str(part).strip())
            if joined:
                return joined
        elif isinstance(raw_caption, str):
            cleaned = raw_caption.strip()
            if cleaned:
                return cleaned
        return self._find_nearby_caption(items, item.get("element_id"))

    def _split_table_rows(self, rows: Any, max_rows: int, max_chars: int) -> list[list[list[dict[str, Any]]]]:
        """Split oversized tables into row blocks while preserving merged-cell blocks.

        The first row is treated as header and repeated in each split segment.
        """
        if not isinstance(rows, list) or len(rows) <= 1:
            return [rows] if isinstance(rows, list) and rows else []

        header = rows[0]
        body = rows[1:]
        if len(body) <= max_rows:
            full_md = self._rows_to_markdown(rows)
            if len(full_md) <= max_chars:
                return [rows]

        blocks = self._build_row_blocks(body)
        segments: list[list[list[dict[str, Any]]]] = []
        current_segment: list[Any] = [header]
        current_rows = 0
        current_chars = 0

        for block in blocks:
            block_rows = len(block)
            block_chars = sum(self._estimate_row_chars(row) for row in block)
            should_split = (
                current_rows > 0
                and (
                    current_rows + block_rows > max_rows
                    or current_chars + block_chars > max_chars
                )
            )
            if should_split:
                segments.append(current_segment)
                current_segment = [header]
                current_rows = 0
                current_chars = 0

            current_segment.extend(block)
            current_rows += block_rows
            current_chars += block_chars

        if len(current_segment) > 1:
            segments.append(current_segment)
        return segments if segments else [rows]

    def _build_row_blocks(self, body_rows: list[Any]) -> list[list[Any]]:
        """Build non-breakable row blocks based on rowspan coverage."""
        if not body_rows:
            return []

        span_end = [idx for idx in range(len(body_rows))]
        for row_idx, row in enumerate(body_rows):
            if not isinstance(row, list):
                continue
            max_end = row_idx
            for cell in row:
                if not isinstance(cell, dict):
                    continue
                rowspan = self._safe_int(cell.get("rowspan"), default=1, minimum=1)
                max_end = max(max_end, row_idx + rowspan - 1)
            span_end[row_idx] = max_end

        blocks: list[list[Any]] = []
        start = 0
        n = len(body_rows)
        while start < n:
            end = span_end[start]
            cursor = start
            while cursor <= end and cursor < n:
                end = max(end, span_end[cursor])
                cursor += 1
            blocks.append(body_rows[start : end + 1])
            start = end + 1
        return blocks

    def _estimate_row_chars(self, row: Any) -> int:
        """Estimate row text size for splitting decisions."""
        if isinstance(row, list):
            return sum(len(self._normalize_table_cell(cell)) for cell in row) + max(len(row) - 1, 0) * 3
        return len(self._normalize_table_cell(row))

    def _find_nearby_caption(self, items: list[dict[str, Any]], element_id: Any) -> Optional[str]:
        """Find a nearby short text as a candidate table caption."""
        if element_id is None:
            return None
        candidates: list[str] = []
        for item in items:
            current_id = item.get("element_id")
            if not isinstance(current_id, int) or not isinstance(element_id, int):
                continue
            if item.get("type") != "text":
                continue
            if abs(current_id - element_id) <= 1:
                text = (item.get("value") or "").strip()
                if text and len(text) <= 80:
                    candidates.append(text)
        return candidates[-1] if candidates else None
