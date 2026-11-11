"""科目匹配（国际化）路由"""

from __future__ import annotations

from fastapi import APIRouter

from src.applications.subject_match_i18n.schemas.match_schema import (
    MatchRequest,
    MatchResponse,
)
from src.applications.subject_match_i18n.services import match_service

router = APIRouter(prefix="/api/subject-match", tags=["科目匹配（国际化）"])


@router.post("/match", response_model=MatchResponse)
async def match_subjects(request: MatchRequest) -> MatchResponse:
    return await match_service.match(
        table_id=request.id,
        template=request.template,
        parse_result=request.parseResult,
    )
