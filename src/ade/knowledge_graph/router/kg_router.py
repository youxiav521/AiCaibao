import os
import uuid
from typing import List, Optional
from fastapi import APIRouter, Query, Path, HTTPException, Body
from loguru import logger

from src.ade.knowledge_graph.schema.kg_schema import (
    JobCreateRequest,
    JobCreateResponse,
    JobCreateData,
    JobListResponse,
    JobListData,
    JobItem,
    JobDetailResponse,
    JobDetailData,
    JobDetailMetadata,
    FileMetadata,
    GraphSummary,
    GraphData,
    GraphGetResponse,
    QueryRequest,
    QueryResponse,
)
from src.ade.knowledge_graph.service.knowledge_graph_service import GraphStorageService
from src.ade.knowledge_graph.query.query_service_v2 import QueryServiceV2
from src.ade.knowledge_graph.query.query_service_v3 import QueryServiceV3
from src.base.task_manager import TaskManager, TaskStatus
import asyncio

router = APIRouter(prefix="/api/kg", tags=["graph"])


@router.post("/jobs", response_model=JobCreateResponse)
async def create_job(request: JobCreateRequest = Body(...)):
    """创建知识图谱生成任务"""
    job_id = str(uuid.uuid4())

    # 🔍 调试日志：记录接收到的文件详情
    logger.info("KG创建任务请求 | job_id={} folder_id={} folder_name={} files_count={}",
                job_id, request.folderId, request.folderName, len(request.files))
    for f in request.files:
        logger.info("  - 文件: id={} name={} elements_count={}", f.id, f.name, len(f.elements))
        if f.elements:
            total_md_len = sum(len(e.md) for e in f.elements)
            logger.info("    - 总md长度: {} chars", total_md_len)
            for i, elem in enumerate(f.elements[:3]):  # 只打印前3个
                logger.info("    - elem[{}]: type={} md_len={}", i, elem.type, len(elem.md))

    kwargs = {
        "folder_id": request.folderId,
        "folder_name": request.folderName,
        "files_data": [f.model_dump() for f in request.files],
        "graph_id": job_id,
    }

    metadata = {
        "folder_id": request.folderId,
        "folder_name": request.folderName,
        "files": [{"id": f.id, "name": f.name} for f in request.files],
    }

    kg_job_id = await TaskManager.submit_with_job_id(
        job_id,
        "create_knowledge_graph",
        task_type="KnowledgeGraph",
        kwargs=kwargs,
        metadata=metadata,
    )

    return JobCreateResponse(
        code="0",
        message="success",
        data=JobCreateData(jobId=kg_job_id),
    )


@router.post("/jobs/incremental", response_model=JobCreateResponse)
async def create_incremental_job(request: JobCreateRequest = Body(...)):
    """创建知识图谱增量更新任务

    入参与全量接口一致，调用方传最新全量文件列表即可。
    系统内部与已存 graph 的 file_index 做 diff，自动识别新增/删除文件。
    若 graph 不存在则自动降级为全量生成。
    """
    job_id = str(uuid.uuid4())

    logger.info(
        "KG增量任务请求 | job_id={} folder_id={} folder_name={} files_count={}",
        job_id, request.folderId, request.folderName, len(request.files),
    )

    kwargs = {
        "folder_id": request.folderId,
        "folder_name": request.folderName,
        "files_data": [f.model_dump() for f in request.files],
        "graph_id": job_id,
    }

    metadata = {
        "folder_id": request.folderId,
        "folder_name": request.folderName,
        "files": [{"id": f.id, "name": f.name} for f in request.files],
        "incremental": True,
    }

    kg_job_id = await TaskManager.submit_with_job_id(
        job_id,
        "update_knowledge_graph",
        task_type="KnowledgeGraphIncremental",
        kwargs=kwargs,
        metadata=metadata,
    )

    return JobCreateResponse(
        code="0",
        message="success",
        data=JobCreateData(jobId=kg_job_id),
    )


@router.get("/jobs", response_model=JobListResponse)
async def get_jobs(
    page: int = Query(0, description="页码索引，0-based"),
    limit: int = Query(10, ge=1, le=100, description="每页记录数"),
    status: Optional[str] = Query(None, description="任务状态过滤: cancelled, success, failed, pending, running"),
    folderId: Optional[str] = Query(None, description="知识库ID过滤"),
    ids: Optional[str] = Query(None, description="任务ID列表，逗号分隔"),
):
    """获取任务列表和状态

    同时查询全量(KnowledgeGraph)和增量(KnowledgeGraphIncremental)任务，合并后按创建时间倒序排序。
    支持按 folderId 过滤，便于用户退出重进后凭知识库ID找回最新任务继续轮询。
    """
    if ids:
        id_list = [i.strip() for i in ids.split(",") if i.strip()]
        tasks = [TaskManager.get(job_id) for job_id in id_list]
        records = await asyncio.gather(*tasks)

        for job_id, record in zip(id_list, records):
            if record is None:
                raise HTTPException(status_code=404, detail=f"任务 {job_id} 不存在或已过期")
        if status:
            records = [r for r in records if r.status.value == status]

    else:
        # 两个 task_type 各查一页，合并后由 router 层做全局排序，再按 limit 截断
        full_records, incr_records = await asyncio.gather(
            TaskManager.get_all(task_type="KnowledgeGraph", limit=limit, offset=page * limit),
            TaskManager.get_all(task_type="KnowledgeGraphIncremental", limit=limit, offset=page * limit),
        )
        records = list(full_records) + list(incr_records)
        # 全局按 created_at 倒序
        records.sort(key=lambda r: r.created_at, reverse=True)
        # 合并后两路总数可能超过 limit，按 limit 截断
        records = records[:limit]

    jobs = []
    for record in records:
        # folderId 过滤
        if folderId is not None and record.metadata.get("folder_id") != folderId:
            continue
        # status 过滤
        if status is not None and record.status.value != status:
            continue
        job_item = JobItem(
            job_id=record.job_id,
            status=record.status.value,
            failureReason=record.error,
            folderId=record.metadata.get("folder_id", ""),
            incremental=record.task_type == "KnowledgeGraphIncremental",
        )
        jobs.append(job_item)

    return JobListResponse(
        code="0",
        message="success",
        total=len(jobs),
        data=JobListData(content=jobs),
    )


@router.get("/jobs/{job_id}", response_model=JobDetailResponse)
async def get_job_detail(job_id: str):
    """获取任务详情"""
    record = await TaskManager.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")

    metadata = None
    if record.metadata:
        files_meta = record.metadata.get("files", [])
        metadata = JobDetailMetadata(
            folderId=record.metadata.get("folder_id", ""),
            folderName=record.metadata.get("folder_name", ""),
            files=[FileMetadata(**f) for f in files_meta],
        )

    summary = None
    graph_id = None
    failure_reason = None

    if record.status == TaskStatus.SUCCESS and record.result:
        graph_id = record.result.get("graph_id")
        entity_count = record.result.get("entity_count", 0)
        relation_count = record.result.get("relation_count", 0)
        summary = GraphSummary(entities=entity_count, relations=relation_count)

    if record.status == TaskStatus.FAILED:
        failure_reason = record.error

    detail_data = JobDetailData(
        status=record.status.value,
        failureReason=failure_reason,
        graphId=graph_id,
        metadata=metadata,
        summary=summary,
    )

    return JobDetailResponse(
        code="0",
        message="success",
        data=detail_data,
    )


@router.get("/{folderId}", response_model=GraphGetResponse, response_model_exclude_none=True)
async def get_graph(folderId: str):
    """获取图谱"""
    graph = await GraphStorageService.get(folderId)
    if graph is None:
        raise HTTPException(status_code=404, detail=f"知识库 {folderId} 下不存在图谱")

    return GraphGetResponse(
        code="0",
        message="success",
        data=GraphData(
            totalEntities=len(graph.entities),
            totalRelations=len(graph.relations),
            entities=graph.entities,
            relations=graph.relations,
        ),
    )

@router.post("/query", response_model=QueryResponse)
async def query_knowledge_graph(request: QueryRequest = Body(...)):
    """知识图谱查询"""
    query_version = os.environ.get("KG_QUERY_VERSION", "v2").lower()
    if query_version not in {"v2", "v3"}:
        raise RuntimeError(f"不支持的 KG_QUERY_VERSION: {query_version}")
    service = QueryServiceV3 if query_version == "v3" else QueryServiceV2
    data = await service.query(
        folder_id=request.folderId,
        query_text=request.query,
    )
    return QueryResponse(
        code="0",
        message="success",
        data=data,
    )
