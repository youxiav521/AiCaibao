import asyncio

import pytest
from langchain_core.documents import Document

from src.applications.rag.services import image_query_service


class _Retriever:
    def __init__(self, documents):
        self.documents = documents

    async def aget_relevant_documents(self, question):
        return self.documents


def test_query_images_returns_three_unique_signed_image_candidates(monkeypatch):
    captured = {}
    documents = [
        Document(page_content="old", metadata={"type": "image", "file_id": "f1", "file_name": "a.pdf", "element_id": 8, "page": 42, "image_storage_path": "f1/images/a.jpg", "score": 0.2}),
        Document(page_content="best", metadata={"type": "image", "file_id": "f1", "file_name": "a.pdf", "element_id": 8, "page": 42, "image_storage_path": "f1/images/a.jpg", "score": 0.9}),
        Document(page_content="two", metadata={"type": "image", "file_id": "f1", "element_id": 9, "image_storage_path": "f1/images/b.jpg", "score": 0.8}),
        Document(page_content="three", metadata={"type": "image", "file_id": "f2", "element_id": 8, "image_storage_path": "f2/images/c.jpg", "score": 0.7}),
    ]
    def get_retriever(**kwargs):
        captured.update(kwargs)
        return _Retriever(documents)
    monkeypatch.setattr(image_query_service.VectorStoresService, "get_grandchildren_retriever", get_retriever)
    async def exists(*args): return True
    monkeypatch.setattr(image_query_service.storage_service, "exists", exists)
    async def sign(bucket, path, *, expires_s): return f"https://signed/{path}"
    monkeypatch.setattr(image_query_service.storage_service, "presigned_url", sign)

    result = asyncio.run(image_query_service.query_images("股权结构图"))

    assert [(item.file_id, item.element_id) for item in result] == [("f1", 8), ("f1", 9), ("f2", 8)]
    assert result[0].summary == "best"
    assert result[0].image_url == "https://signed/f1/images/a.jpg"
    assert captured["doc_kinds"] == ["chunk", ""]


def test_query_images_skips_invalid_or_unsigned_candidates(monkeypatch):
    docs = [Document(page_content="x", metadata={"type": "image", "file_id": "f1", "element_id": 1, "image_storage_path": "missing.jpg", "score": 1})]
    monkeypatch.setattr(image_query_service.VectorStoresService, "get_grandchildren_retriever", lambda **kwargs: _Retriever(docs))
    async def missing(*args): return False
    monkeypatch.setattr(image_query_service.storage_service, "exists", missing)
    assert asyncio.run(image_query_service.query_images("图")) == []


def test_search_image_candidates_deduplicates_and_preserves_score(monkeypatch):
    documents = [
        Document(page_content="旧摘要", metadata={"type": "image", "file_id": "f1", "element_id": 8, "image_storage_path": "f1/images/a.png", "score": 0.20}),
        Document(page_content="最佳摘要", metadata={"type": "image", "file_id": "f1", "file_name": "a.pdf", "element_id": 8, "page": 42, "image_storage_path": "f1/images/a.png", "score": 0.91}),
        Document(page_content="第二张", metadata={"type": "image", "file_id": "f1", "element_id": 9, "image_storage_path": "f1/images/b.jpg", "score": 0.80}),
    ]
    monkeypatch.setattr(image_query_service.VectorStoresService, "get_grandchildren_retriever", lambda **kwargs: _Retriever(documents))

    result = asyncio.run(
        image_query_service.search_image_candidates("股权结构图", top_k=5, file_ids=["f1"])
    )

    assert [(item.file_id, item.element_id) for item in result] == [("f1", 8), ("f1", 9)]
    assert result[0].score == pytest.approx(0.91)
    assert result[0].summary == "最佳摘要"


def test_search_image_candidates_resolves_missing_path_from_mineru_items(monkeypatch):
    """旧图片索引缺少路径时，按 file_id/element_id 回查 MinerU 产物。"""
    documents = [
        Document(
            page_content="股权结构图",
            metadata={"type": "image", "file_id": "f1", "element_id": 8, "score": 0.91},
        )
    ]
    monkeypatch.setattr(
        image_query_service.VectorStoresService,
        "get_grandchildren_retriever",
        lambda **kwargs: _Retriever(documents),
    )

    async def load_items(file_id):
        assert file_id == "f1"
        return [{"type": "image", "element_id": 8, "img_path": "images/equity.jpg"}]

    monkeypatch.setattr(image_query_service, "load_mineru_items", load_items)

    result = asyncio.run(
        image_query_service.search_image_candidates("股权结构图", top_k=5, file_ids=["f1"])
    )

    assert len(result) == 1
    assert result[0].image_storage_path == "f1/images/equity.jpg"


def test_load_image_as_data_uri_downloads_original_image(monkeypatch):
    candidate = image_query_service.ImageSearchCandidate(
        file_id="f1",
        element_id=8,
        image_storage_path="f1/images/a.png",
        score=0.91,
    )

    async def download(bucket, object_name):
        assert object_name == "f1/images/a.png"
        return b"png-bytes"

    monkeypatch.setattr(image_query_service.storage_service, "download", download)
    result = asyncio.run(image_query_service.load_image_as_data_uri(candidate))

    assert result == "data:image/png;base64,cG5nLWJ5dGVz"
