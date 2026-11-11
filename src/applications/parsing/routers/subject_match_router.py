"""财报科目匹配API路由"""

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from pathlib import Path
from typing import Optional, List, Any

from ..schemas.subject_match_schema import SubjectMatchRequest
from ..service.subject_match_service import SubjectMatchService
from ..service.subject_matching.response_builder import build_subject_match_rows_response
from src.utils import json_utils

# 创建路由
router = APIRouter(
    prefix="/api/parsing/subject-match",
    tags=["财报科目匹配"]
)

# subject_templates.json 相对于本文件的路径：src/applications/parsing/routers/ → 项目根 → evaluator/
_TEMPLATES_FILE = Path(__file__).parent.parent.parent.parent.parent / "evaluator" / "subject_templates.json"

# 实例化服务（默认启用LLM，因为Java端已经过滤了能通过字符串匹配的科目）
subject_match_service = SubjectMatchService(use_llm=True, llm_max_concurrency=20)


@router.get(
    "/templates",
    summary="获取标准科目模板",
    description=(
        "返回 evaluator/subject_templates.json 中存储的标准科目模板列表，"
        "供评估脚本在无需 Swagger 的情况下获取含真实 subjectId 和同义词的 reportList。"
    ),
)
async def get_subject_templates() -> List[Any]:
    """
    读取并返回标准科目模板。

    模板文件位于 evaluator/subject_templates.json，格式为 reportList 数组。
    每条记录包含：reportName, reportCaliber, reportId, subjectList（含 subjectId/synonymSubject）。

    若文件不存在或尚未填充，返回 404 并给出填充方法。
    """
    if not _TEMPLATES_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"模板文件不存在：{_TEMPLATES_FILE}。"
                "请运行 python evaluator/batch_parse_images.py --export-template <request_raw_*.json> 生成。"
            ),
        )
    try:
        templates = json_utils.loads(_TEMPLATES_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取模板文件失败：{e}")

    # 过滤掉占位条目（reportId == "__placeholder__"）
    real_templates = [t for t in templates if isinstance(t, dict) and t.get("reportId") != "__placeholder__"]
    if not real_templates:
        raise HTTPException(
            status_code=404,
            detail=(
                "模板文件存在但尚未填充真实数据。"
                "请运行 python evaluator/batch_parse_images.py --export-template <request_raw_*.json> 生成。"
            ),
        )
    logger.info(f"返回标准科目模板 {len(real_templates)} 条")
    return real_templates


@router.post(
    "/match",
    response_model=List[List[Any]],  # 返回二维数组
    summary="财报科目匹配",
    description=(
        "根据报表定义和表格数据，进行科目匹配并填充 subjectId，返回二维数组（rows）。"
        "每个单元格额外包含：caliberLabel（合并/单体/混合/未知）、"
        "accountingYear（表头解析到的年份列，首列为 null）、statementType、gaapType，"
        "以及原有 bBox/pageAt 等。"
    ),
)
async def match_subjects(
    request: SubjectMatchRequest,
    use_llm: Optional[bool] = Query(True, description="是否使用LLM进行语义匹配，默认True。Java端已过滤字符串匹配的科目，建议使用LLM"),
) -> List[List[Any]]:
    """
    财报科目匹配接口
    
    匹配逻辑：
    1. 根据 reportList 的 (reportName + reportCaliber) 语义匹配 tables 的 reportName
    2. 根据 subjectName 精确匹配 rows[][0].text
    3. 根据 synonymSubject 精确匹配 rows[][0].text
    4. 根据 summaryRelation 精确匹配 rows[][0].text
    5. 以上未命中时，交由 LLM 进行语义匹配
    
    Returns:
        匹配结果，包含填充了subjectId的表格列表（二维数组）
    """
    try:
        logger.info(f"接收财报科目匹配请求 - fileId: {request.fileId}, use_llm: {use_llm}")
        # 根据参数创建服务实例
        service = subject_match_service
        if use_llm is not None and use_llm != subject_match_service.use_llm:
            service = SubjectMatchService(use_llm=use_llm)
        
        # 调用服务层
        response = await service.match_subjects(request)
        
        all_rows = build_subject_match_rows_response(request, response)
        
        logger.info(f"财报科目匹配完成 - fileId: {request.fileId}, "
                   f"表格数: {len(response.tables)}, 总行数: {len(all_rows)}")
        
        return all_rows
        
    except Exception as e:
        logger.error(f"财报科目匹配API异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"财报科目匹配失败: {str(e)}")
