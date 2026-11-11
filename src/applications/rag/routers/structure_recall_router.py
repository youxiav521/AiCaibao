from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException
from loguru import logger

from ..schemas import StructureBatchRecallRequest, BatchQueryItemHttp
from ..services.structure_recall_service import recall_structure_batch

router = APIRouter()


@router.post(
    "/structure/recall-batch",
    response_model=List[BatchQueryItemHttp],
    summary="文档结构批量召回",
    description="将多个问题对同一批文件的结构召回合并为一次大模型调用，使文档结构的 KV cache 在多个问题间共享，减少重复计算。",
)
async def structure_recall_batch(body: StructureBatchRecallRequest) -> List[BatchQueryItemHttp]:
    if not body.questions:
        raise HTTPException(status_code=400, detail="questions 不能为空")

    file_ids = [fid.strip() for fid in body.fileIds if fid.strip()]
    if not file_ids:
        raise HTTPException(status_code=400, detail="fileIds 不能为空")

    questions = [
        {"question_id": q.questionId, "question_text": q.questionText}
        for q in body.questions
    ]

    logger.info(
        "文档结构批量召回开始 | question_count={} file_count={}",
        len(questions), len(file_ids),
    )

    try:
        result = await recall_structure_batch(
            questions=questions,
            file_ids=file_ids,
            max_tokens=body.maxTokens,
            max_structure_num=body.maxStructureNum,
        )
    except Exception:
        logger.exception("文档结构批量召回失败 | question_count={}", len(questions))
        raise HTTPException(status_code=500, detail="文档结构批量召回失败")

    logger.info(
        "文档结构批量召回完成 | question_count={} result_count={}",
        len(questions), len(result),
    )

    return result
