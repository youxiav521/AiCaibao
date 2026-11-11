from src.ade.knowledge_graph.schema.kg_schema import FileInput, Piece
from typing import List
import hashlib
from loguru import logger


class SplittingService:
    """文档切片服务"""

    MAX_PIECE_CHARS = 50000

    @classmethod
    def split_piece(cls, piece: Piece) -> List[Piece]:
        """沿最接近字符中点的结构边界将 Piece 二分。"""
        elems = piece.elements
        if len(elems) < 2:
            return []

        page_boundaries = [
            index
            for index in range(1, len(elems))
            if elems[index - 1].page != elems[index].page
        ]
        boundaries = page_boundaries or list(range(1, len(elems)))
        total_chars = sum(len(elem.md) for elem in elems)
        prefix_chars = 0
        imbalances: dict[int, int] = {}
        boundary_set = set(boundaries)
        for index, elem in enumerate(elems[:-1], 1):
            prefix_chars += len(elem.md)
            if index in boundary_set:
                left_chars = prefix_chars + index - 1
                right_chars = total_chars - prefix_chars + len(elems) - index - 1
                imbalances[index] = abs(left_chars - right_chars)

        split_at = min(boundaries, key=lambda index: (imbalances[index], index))
        children: list[Piece] = []
        for branch, child_elems in enumerate((elems[:split_at], elems[split_at:])):
            children.append(
                Piece(
                    id=f"{piece.id}__split_{branch}",
                    content="\n".join(elem.md for elem in child_elems),
                    file_id=piece.file_id,
                    file_name=piece.file_name,
                    elements=child_elems,
                )
            )
        return children

    @classmethod
    def split(cls, file: FileInput) -> List[Piece]:
        """将文件内容切片

        以 element 为最小单元，逐个累计字符长度，达到 MAX_PIECE_CHARS 则回退一个 element 作为分片边界。
        单个超长 element 强制入片（不可再分）。

        Args:
            file: 文件输入，含元素元数据列表

        Returns:
            List[Piece]: 切片列表
        """
        elems = file.elements
        if not elems:
            return []

        pieces: list[Piece] = []
        start = 0

        while start < len(elems):
            length = 0
            end = start

            for i in range(start, len(elems)):
                add = len(elems[i].md) + (1 if i > start else 0)  # +1 对应 "\n" 分隔符

                if length + add > cls.MAX_PIECE_CHARS and length > 0:
                    break

                length += add
                end = i + 1

            piece_elems = elems[start:end]
            piece_content = "\n".join(e.md for e in piece_elems)
            multi = len(pieces) > 0 or end < len(elems)
            piece_id = f"piece_{file.id}_{len(pieces)}" if multi else f"piece_{file.id}"

            pieces.append(Piece(
                id=piece_id,
                content=piece_content,
                content_hash=hashlib.md5(piece_content.encode("utf-8")).hexdigest(),
                file_id=file.id,
                file_name=file.name,
                elements=piece_elems,
            ))

            start = end

        if len(pieces) == 1:
            logger.info("SplittingService | 单片切片 | file_id={} file_name={} content_len={}", file.id, file.name, len(pieces[0].content))
        else:
            logger.info("SplittingService | 多片切片 | file_id={} file_name={} pieces={}", file.id, file.name, len(pieces))

        return pieces
