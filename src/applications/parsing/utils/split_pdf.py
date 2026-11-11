"""PDF 按页数平均拆分服务。

把一个 PDF 平均拆成 N 份（份数由 ``num_parts`` 指定，默认 3）。
均分策略：按页数，前 N%total 份多 1 页，保证每份页数差不超过 1，且无丢页。

依赖：pymupdf。
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from loguru import logger


class PdfSplitInfeasible(Exception):
    """拆分不可行：候选边界上无法避开割裂图表，前后向均无解。

    路由层捕获后应回退到"原文件"处理路径（同其它拆分异常的处理方式）。
    """

    pass


class PdfSplitService:
    """PDF 按页数平均拆分服务。

    用于将单个 PDF 文件按页数均分为多份，便于并发解析或按段处理。
    """

    DEFAULT_NUM_PARTS: int = 3

    def split(
        self,
        input_pdf: str,
        num_parts: int = DEFAULT_NUM_PARTS,
        output_dir: Optional[str] = None,
        output_prefix: Optional[str] = None,
        avoid_split_through_figure: bool = True,
        pages_per_part: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """将 PDF 拆分为多份。

        支持两种互斥的拆分模式（``pages_per_part`` 不为 ``None`` 时优先）：

        - **按份数均分**（默认）：拆成 ``num_parts`` 份，按页数均分，
          前 ``total % n`` 份多 1 页。
        - **按每份页数切分**：每份 ``pages_per_part`` 页，最后一份取余。
          例如 250 页 / 每份 100 页 → 100/100/50。

        Args:
            input_pdf: 输入 PDF 文件路径。
            num_parts: 拆成几份，默认 3；会按总页数自动截断。
                仅在 ``pages_per_part`` 为 ``None`` 时生效。
            output_dir: 输出目录；为 ``None`` 时与输入文件同目录。
            output_prefix: 输出文件名前缀；为 ``None`` 时使用
                ``"{输入文件stem}_part"``。
            avoid_split_through_figure: 是否避开"图表被分页割裂"的边界；
                启用后会用启发式检查每个分页边界，若发现该边界可能把图表
                从中间切开（页底/页顶无文字 + 邻接页有较大图像或图形），
                就把该边界向后推 1 页，直到避开或无法再推为止。
            pages_per_part: 每份页数；不为 ``None`` 时按此模式切分，
                忽略 ``num_parts``。必须 >= 1。

        Returns:
            每份的元数据列表，元素字段：
              - ``path`` (str): 输出 PDF 绝对路径
              - ``index`` (int): 1-based 序号
              - ``start_page`` (int): 起始页码（1-based，闭区间）
              - ``end_page`` (int): 结束页码（1-based，闭区间）
              - ``page_count`` (int): 份内页数
              - ``size_bytes`` (int): 输出文件字节数
        """
        in_path = Path(input_pdf)
        if not in_path.is_file():
            raise FileNotFoundError(f"输入 PDF 不存在: {input_pdf}")

        if num_parts < 1:
            raise ValueError("份数 num_parts 必须 >= 1。")

        if pages_per_part is not None and pages_per_part < 1:
            raise ValueError("pages_per_part 必须 >= 1。")

        out_dir = Path(output_dir) if output_dir else in_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        prefix = output_prefix if output_prefix else f"{in_path.stem}_part"

        results = self._split_sync(
            in_path, num_parts, out_dir, prefix, avoid_split_through_figure, pages_per_part
        )
        logger.info(
            "PDF 拆分完成 | input={}, num_parts={}, output_dir={}, parts={}",
            in_path,
            num_parts,
            out_dir,
            len(results),
        )
        return results

    @staticmethod
    def _has_large_content_in_region(
        page: "fitz.Page",
        region: str,
        min_size_pt: float = 100.0,
    ) -> bool:
        """检查页面指定半区是否存在较大的图像或图形。

        Args:
            page: fitz.Page 对象。
            region: ``"top"`` 检查页面顶部 50%，``"bottom"`` 检查底部 50%。
            min_size_pt: 最小包围盒尺寸（点）；低于此值视为 logo/小图标忽略。

        Returns:
            是否存在满足尺寸的非文字内容。
        """
        if region not in ("top", "bottom"):
            raise ValueError(f"region 必须是 'top' 或 'bottom'，当前: {region}")
        height = page.rect.height
        threshold = height * 0.5

        for info in page.get_image_info(xrefs=True):
            bbox = info.get("bbox")
            if not bbox:
                continue
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w < min_size_pt or h < min_size_pt:
                continue
            if region == "bottom" and bbox[3] > threshold:
                return True
            if region == "top" and bbox[1] < threshold:
                return True

        for d in page.get_drawings():
            rect = d.get("rect")
            if not rect:
                continue
            w = rect[2] - rect[0]
            h = rect[3] - rect[1]
            if w < min_size_pt or h < min_size_pt:
                continue
            if region == "bottom" and rect[3] > threshold:
                return True
            if region == "top" and rect[1] < threshold:
                return True

        return False

    @staticmethod
    def _is_risky_boundary(
        src: "fitz.Document",
        page_a_idx: int,
        page_b_idx: int,
    ) -> bool:
        """检查 ``page_a -> page_b`` 之间的分页边界是否是"图表被割裂"风险边界。

        启发式规则（同时满足才视为风险）：
          1. ``page_a`` 的底部 30% 没有任何文字块；
          2. ``page_b`` 的顶部 30% 没有任何文字块；
          3. ``page_a`` 的底部 50% 或 ``page_b`` 的顶部 50% 存在较大
             （>=100pt）的图像或矢量图形。

        注意局限：
          - 无法严格区分"图"和"表"——表通常以 vector drawing 表示，与图触发同一规则；
          - 不能识别"无可见边框的纯文本表格"（本质是文字块）；
          - 极个别情况可能误判（如页底大图 + 页顶小 logo 仍会被判为风险）；
            误判仅导致"多推一页"，不会损坏正确性。
        """
        page_a = src[page_a_idx]
        page_b = src[page_b_idx]

        height_a = page_a.rect.height
        height_b = page_b.rect.height

        text_blocks_a = page_a.get_text("blocks")
        text_blocks_b = page_b.get_text("blocks")

        # 1) page_a 底部 30% 内是否有文字
        bottom_has_text = any(
            b[3] > height_a * 0.7 and b[6] == 0 and b[4].strip()
            for b in text_blocks_a
        )
        # 2) page_b 顶部 30% 内是否有文字
        top_has_text = any(
            b[1] < height_b * 0.3 and b[6] == 0 and b[4].strip()
            for b in text_blocks_b
        )

        if bottom_has_text or top_has_text:
            return False

        # 3) 两侧都没有文字分隔时，再看是否邻接较大非文字内容
        bottom_has_content = PdfSplitService._has_large_content_in_region(page_a, "bottom")
        if bottom_has_content:
            return True
        return PdfSplitService._has_large_content_in_region(page_b, "top")

    @staticmethod
    def _split_sync(
        in_path: Path,
        num_parts: int,
        out_dir: Path,
        prefix: str,
        avoid_split_through_figure: bool = True,
        pages_per_part: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        src = fitz.open(str(in_path))
        try:
            total = src.page_count
            if total == 0:
                raise ValueError("PDF 没有任何页面。")

            boundaries: List[Tuple[int, int]] = []  # 元素为 (start, end) 闭区间，0-based
            if pages_per_part is not None:
                # 按每份 pages_per_part 页切分，最后一份取余
                start = 0
                while start < total:
                    end = min(start + pages_per_part - 1, total - 1)
                    boundaries.append((start, end))
                    start = end + 1
            else:
                n = min(num_parts, total)
                base, extra = divmod(total, n)
                start = 0
                for i in range(n):
                    count = base + (1 if i < extra else 0)
                    end = start + count - 1  # insert_pdf 的 to_page 是闭区间
                    boundaries.append((start, end))
                    start = end + 1

            # 调整边界：避免在图表被割裂处划分
            # 流程：
            #   1) 对每对相邻 part，先尝试向后推 1 页；
            #   2) 推到 next part 不足 1 页（max_push 用尽）时边界仍 risk，
            #      改为从最初那个 boundary 向前找非风险位置；
            #   3) 向前找到：这一对相邻 part 涉及的页范围只切 1 刀，输出 2 份；
            #   4) 向前也找不到：抛 PdfSplitInfeasible，由路由层回退到"原文件"处理。
            if avoid_split_through_figure and len(boundaries) >= 2:
                new_boundaries: List[Tuple[int, int]] = []
                skip_next = False
                for i in range(len(boundaries)):
                    if skip_next:
                        skip_next = False
                        continue

                    current_start, current_end = boundaries[i]

                    # 最后一个 boundary 没有后续 part 可调整
                    if i == len(boundaries) - 1:
                        new_boundaries.append((current_start, current_end))
                        continue

                    next_start, next_end = boundaries[i + 1]
                    original_boundary = current_end  # 记下最初边界位置，供前向回退使用

                    # === 第一阶段：向后推（沿用旧逻辑）===
                    max_push = next_end - next_start
                    pushed = 0
                    while (
                        pushed < max_push
                        and current_end + 1 < next_end
                        and PdfSplitService._is_risky_boundary(src, current_end, next_start)
                    ):
                        logger.info(
                            "PDF 拆分：检测到风险边界（可能割裂图表），向后推 1 页 | "
                            "boundary={}, part_{} last_page {}->{}, part_{} first_page {}->{}",
                            i, i + 1, current_end, current_end + 1,
                            i + 2, next_start, next_start + 1,
                        )
                        current_end += 1
                        next_start += 1
                        pushed += 1

                    # 检查向后推之后是否仍为风险
                    still_risky = PdfSplitService._is_risky_boundary(
                        src, current_end, next_start
                    )
                    if not still_risky:
                        new_boundaries.append((current_start, current_end))
                        new_boundaries.append((next_start, next_end))
                        skip_next = True
                        continue

                    # === 第二阶段：从最初边界向前找非风险位置 ===
                    # boundary = b 表示 current part = [current_start, b]，next part = [b+1, next_end]
                    # 范围是 [current_start, original_boundary - 1]（含 current_start，最退化情形下
                    # current part 只剩 1 页，也是合法 2 份切分）。
                    found_boundary: Optional[int] = None
                    for b in range(original_boundary - 1, current_start - 1, -1):
                        if not PdfSplitService._is_risky_boundary(src, b, b + 1):
                            found_boundary = b
                            break

                    if found_boundary is not None:
                        logger.warning(
                            "PDF 拆分：向后推无解，从原始边界向前找到非风险边界，"
                            "该对相邻 part 退化为 2 份 | "
                            "original_boundary_page={}, new_boundary_page={}, "
                            "part1=[{}, {}], part2=[{}, {}]",
                            original_boundary + 1,
                            found_boundary + 1,
                            current_start + 1, found_boundary + 1,
                            found_boundary + 2, next_end + 1,
                        )
                        new_boundaries.append((current_start, found_boundary))
                        new_boundaries.append((found_boundary + 1, next_end))
                        skip_next = True
                    else:
                        # 前后向均找不到非风险边界 → 直接抛异常，让路由层走"原文件"回退路径
                        # （与其它 PDF 拆分失败时同一条处理链路）
                        logger.warning(
                            "PDF 拆分：前向也找不到非风险边界，"
                            "放弃本次拆分，回退到原文件处理 | "
                            "covered_pages=[{}, {}]",
                            current_start + 1, next_end + 1,
                        )
                        raise PdfSplitInfeasible(
                            f"PDF 拆分不可行：相邻 part 范围 "
                            f"[{current_start + 1}, {next_end + 1}] 内"
                            f"前后向均找不到非风险边界"
                        )

                boundaries = new_boundaries

            parts: List[Dict[str, Any]] = []
            for i, (s, e) in enumerate(boundaries):
                out_path = out_dir / f"{prefix}_{i + 1}.pdf"
                part_doc = fitz.open()
                try:
                    part_doc.insert_pdf(src, from_page=s, to_page=e)
                    part_doc.save(str(out_path))
                finally:
                    part_doc.close()

                parts.append(
                    {
                        "path": str(out_path.resolve()),
                        "index": i + 1,
                        "start_page": s + 1,
                        "end_page": e + 1,
                        "page_count": e - s + 1,
                        "size_bytes": out_path.stat().st_size,
                    }
                )
            return parts
        finally:
            src.close()


def split_pdf(input_pdf: Path, n: int, output_prefix: str) -> List[Path]:
    """向后兼容的函数式入口，等价于 ``PdfSplitService().split(...)``。

    返回输出文件路径列表（与原实现保持一致）。
    """
    parts = PdfSplitService().split(
        input_pdf=str(input_pdf),
        num_parts=n,
        output_dir=str(input_pdf.parent),
        output_prefix=output_prefix,
    )
    return [Path(p["path"]) for p in parts]


def split_pdf_bytes(
    pdf_bytes: bytes,
    num_parts: int = PdfSplitService.DEFAULT_NUM_PARTS,
    avoid_split_through_figure: bool = True,
    pages_per_part: Optional[int] = None,
) -> List[Tuple[bytes, Dict[str, Any]]]:
    """将 PDF 字节流拆分为多份，返回 ``(字节流, 元数据)`` 列表。

    与 :meth:`PdfSplitService.split` 等价，但接受字节流输入，调用方无需自行落盘；
    函数内部会创建临时目录并在返回前清理所有临时文件。

    Args:
        pdf_bytes: 输入 PDF 字节流。
        num_parts: 拆成几份，默认 3；会按总页数自动截断。
            仅在 ``pages_per_part`` 为 ``None`` 时生效。
        avoid_split_through_figure: 是否避开"图表被分页割裂"的边界，透传给
            :meth:`PdfSplitService.split`；默认为 ``True``。
        pages_per_part: 每份页数；不为 ``None`` 时按此模式切分，忽略 ``num_parts``。
            必须为 ``None`` 或 >= 1。

    Returns:
        每份的 ``(pdf_bytes, metadata)`` 元组列表，``metadata`` 字段同 :meth:`PdfSplitService.split`。
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_in:
        tmp_in.write(pdf_bytes)
        tmp_in_path = Path(tmp_in.name)

    tmp_out_dir = Path(tempfile.mkdtemp(prefix="pdf_split_bytes_"))
    parts: List[Dict[str, Any]] = []
    try:
        parts = PdfSplitService().split(
            input_pdf=str(tmp_in_path),
            num_parts=num_parts,
            output_dir=str(tmp_out_dir),
            avoid_split_through_figure=avoid_split_through_figure,
            pages_per_part=pages_per_part,
        )

        results: List[Tuple[bytes, Dict[str, Any]]] = []
        for p in parts:
            part_path = Path(p["path"])
            try:
                results.append((part_path.read_bytes(), p))
            finally:
                try:
                    part_path.unlink()
                except Exception:
                    logger.warning("拆分临时文件清理失败 | path={}", part_path)
        return results
    finally:
        try:
            tmp_in_path.unlink(missing_ok=True)
        except Exception:
            logger.warning("拆分输入临时文件清理失败 | path={}", tmp_in_path)
        try:
            for p in parts:
                Path(p["path"]).unlink(missing_ok=True)
            tmp_out_dir.rmdir()
        except Exception:
            logger.warning("拆分输出临时目录清理失败 | dir={}", tmp_out_dir)


__all__ = ["PdfSplitService", "split_pdf", "split_pdf_bytes"]
