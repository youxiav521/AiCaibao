from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException
from loguru import logger

from ..config.rag_settings import RAG_SETTINGS
from ..schemas import (
    BatchQueryRequestHttp,
    DeleteRequestHttp,
    DeleteResponse,
    RagBackfillJobResponse,
    RagBackfillRequestHttp,
    RagBackfillSubmitResponse,
    UploadIndexResponse,
    UploadIndexRequestHttp,
    BatchQueryItemHttp,
    batch_query_item_to_http,
    batchQueryRequestHttp_to_batchQueryRequest,
    FolderFileItem,
    FolderFileCountResponse,
)
from src.base.task_manager import TaskManager
from ..services.index_service import delete_document_by_file_id, index_document_items
from ..services.vectorstore import VectorStoresService
from ..services.query_service import run_batch_query
from ..utils.validators import validate_file_id
import time

router = APIRouter()
_RAG_BACKFILL_TASK_TYPE = "rag_legacy_backfill"

@router.post("/index", response_model=UploadIndexResponse)
async def index_document(body: UploadIndexRequestHttp) -> UploadIndexResponse:
    """上传结构化元素对象并进行切片入库。

    步骤：解析 docParseContent → 提取/生成 file_id → 检查去重 → 切片与入库。

    Args:
        body: JSON 请求体，包含 `docParseContent`、`fileName`、`docFolderId`。

    Returns:
        上传入库的执行结果。
    """
    doc_parse_items = [dict(item) for item in body.docParseContent]
    
    start_time = time.time()
    try:
        await index_document_items(
            doc_parse_items,
            body.fileName,
            body.docFolderId,
        )
    except (ValueError, FileExistsError) as exc:
        elapsed = time.time() - start_time
        logger.warning("document indexing failed: elapsed={:.1f}s, fileName={}, error={}", elapsed, body.fileName, exc)
        raise HTTPException(status_code=400, detail=f"{body.fileName} 处理失败: {exc}") from exc
    except RuntimeError as exc:
        elapsed = time.time() - start_time
        logger.error("document indexing failed: elapsed={:.1f}s, fileName={}, error={}", elapsed, body.fileName, exc)
        raise HTTPException(status_code=500, detail=f"{body.fileName} 处理失败: {exc}") from exc

    elapsed = time.time() - start_time
    logger.info("document indexing success: elapsed={:.1f}s, fileName={}", elapsed, body.fileName)
    return UploadIndexResponse(success=True)


@router.post("/doc/delete", response_model=DeleteResponse)
async def delete_document(body: DeleteRequestHttp) -> DeleteResponse:
    """删除指定 file_id 的索引记录。"""
    try:
        file_id = validate_file_id(body.fileId)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"fileId 非法: {e}")
    ok = await delete_document_by_file_id(
        file_id
    )
    return DeleteResponse(file_id=body.fileId, deleted=ok)


@router.delete("/database/reset", response_model=bool, summary="重置数据库")
async def reset_database(password: str) -> bool:
    """危险：清空 Milvus 中当前 DB 的所有集合，并清空检索缓存。
    
    操作包括：
    1. 删除 Milvus 中的所有集合。
    2. 尝试删除 Milvus 数据库（如果非默认）。
    3. 清空 BM25 缓存。
    4. 重建数据库和集合（初始化）。
    """

    if not RAG_SETTINGS.rag_reset_enabled:
        raise HTTPException(status_code=404, detail="Not found")

    expected = RAG_SETTINGS.rag_reset_password
    if not expected:
        raise HTTPException(status_code=403, detail="Reset not configured")
    if password != expected:
        raise HTTPException(status_code=401, detail="密码错误")
    result = await VectorStoresService.reset_all()
    return result


@router.post("/query", response_model=List[BatchQueryItemHttp])
async def batch_query(body: BatchQueryRequestHttp) -> List[BatchQueryItemHttp]:
    """批量 RAG 检索接口。"""

    items = await run_batch_query(batchQueryRequestHttp_to_batchQueryRequest(body))
    return [batch_query_item_to_http(item) for item in items]


@router.post("/backfill", response_model=RagBackfillSubmitResponse, summary="提交旧 RAG 集合回填任务")
async def submit_legacy_backfill(body: RagBackfillRequestHttp) -> RagBackfillSubmitResponse:
    """提交后台任务，将旧三集合数据回填到 MinIO 和新 LangChain 向量集合。"""
    job_id = await TaskManager.submit(
        "process_rag_legacy_backfill",
        task_type=_RAG_BACKFILL_TASK_TYPE,
        kwargs={
            "old_parent_collection": body.oldParentCollection,
            "old_child_collection": body.oldChildCollection,
            "old_grandchild_collection": body.oldGrandchildCollection,
        },
        metadata={"source": "rag_router"},
    )
    return RagBackfillSubmitResponse(jobId=job_id, taskType=_RAG_BACKFILL_TASK_TYPE)


@router.get("/backfill/{job_id}", response_model=RagBackfillJobResponse, summary="查询旧 RAG 集合回填任务")
async def get_legacy_backfill_job(job_id: str) -> RagBackfillJobResponse:
    record = await TaskManager.get(job_id)
    if record is None or record.task_type != _RAG_BACKFILL_TASK_TYPE:
        raise HTTPException(status_code=404, detail="回填任务不存在")
    return RagBackfillJobResponse(
        jobId=record.job_id,
        status=record.status.value,
        result=record.result,
        error=record.error,
        attempts=record.attempts,
    )


@router.get("/folder/{folder_id}/files", response_model=FolderFileCountResponse, summary="查询文件夹下的文件数量")
async def get_folder_file_count(folder_id: str) -> FolderFileCountResponse:
    """根据 folder_id 查询该文件夹下有多少个文件。

    Args:
        folder_id: 文件夹 ID

    Returns:
        文件数量以及文件 ID、文件名列表
    """
    count, files = await VectorStoresService.count_files_by_folder_id(folder_id)
    return FolderFileCountResponse(
        folderId=folder_id,
        fileCount=count,
        fileIds=[item["file_id"] for item in files],
        files=[
            FolderFileItem(
                fileId=item["file_id"],
                fileName=item["file_name"] or None,
            )
            for item in files
        ],
    )
