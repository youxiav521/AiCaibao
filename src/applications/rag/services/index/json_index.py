from src.applications.rag.chunking import make_chunks_for_txt_items
from src.applications.rag.config.rag_settings import RAG_SETTINGS
from src.applications.rag.chunking.plain_text import (
    build_grandchildren_from_children_text,
)
from .index_writer import write_index_payloads

from src.applications.rag.chunking import make_chunks_for_json_items


async def build(
    file_id: str,
    mineru_json: list[dict],
    folder_id: str | None = None,
    file_name: str | None = None,
):
    parents, children = make_chunks_for_json_items(
        mineru_json,
        file_id,
        parent_chunk_size=RAG_SETTINGS.p_max,
        child_chunk_size=RAG_SETTINGS.c_max,
        orphan_min=RAG_SETTINGS.orphan_min,
        folder_id=folder_id,
        file_name=file_name,
    )
    grandchildren = build_grandchildren_from_children_text(
        children,
        file_id,
        chunk_size=RAG_SETTINGS.c_min,
        chunk_overlap=RAG_SETTINGS.child_overlap,
        strategy="json",
        folder_id=folder_id,
        file_name=file_name,
    )
    return await write_index_payloads(
        file_id=file_id, parents=parents, children=children, grandchildren=grandchildren
    )
