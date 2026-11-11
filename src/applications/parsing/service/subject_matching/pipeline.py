from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.applications.parsing.service.subject_match_service import SubjectMatchService
from src.applications.parsing.schemas.subject_match_schema import SubjectMatchRequest, SubjectMatchResponse
from src.applications.parsing.service.subject_matching.prompts import (
    SubjectMatchPromptProvider,
    choose_subject_prompt_provider,
)


@dataclass(frozen=True)
class SubjectMatchPipelineConfig:
    strategy: str = "china"
    use_llm: bool = True
    llm_max_concurrency: int = 10


class SubjectMatchPipeline:
    """Strategy-aware subject matching pipeline.

    The China path delegates to the existing service unchanged. International
    prompt/rule providers are exposed here so future matching logic can be
    developed without modifying the domestic route behavior.
    """

    def __init__(self, config: Optional[SubjectMatchPipelineConfig] = None):
        self.config = config or SubjectMatchPipelineConfig()
        self.prompt_provider: SubjectMatchPromptProvider = choose_subject_prompt_provider(
            self.config.strategy
        )
        self.service = SubjectMatchService(
            use_llm=self.config.use_llm,
            llm_max_concurrency=self.config.llm_max_concurrency,
        )

    async def match_subjects(self, request: SubjectMatchRequest) -> SubjectMatchResponse:
        return await self.service.match_subjects(request)
