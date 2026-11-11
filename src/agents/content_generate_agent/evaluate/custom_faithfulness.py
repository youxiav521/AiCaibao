from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, ConfigDict
from pydantic.alias_generators import to_camel

from src.base.langfuse_observe import trace_span

from ..config import create_default_configs
from loguru import logger
from ..utils import (
    ainvoke_llm_with_structured_output,
    ensure_within_token_limit,
    apply_prompt_template,
    log_error_type_and_message,
)


class ExtractedStatement(BaseModel):
    """模型抽取的原子声明"""

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    id: int = Field(..., description="自1开始的声明序号")
    statement: str = Field(..., description="单条原子声明，保持原意不扩写")


class StatementExtractionResponse(BaseModel):
    """LLM 结构化输出：声明列表"""

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    statements: List[ExtractedStatement] = Field(
        default_factory=list, description="拆分后的原子声明列表"
    )


class VerifiedStatement(BaseModel):
    """单条声明的验证结果"""

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    id: int = Field(..., description="对应的声明序号")
    statement: str = Field(..., description="原始声明内容")
    supported: bool = Field(..., description="上下文是否支持/能推断该声明")
    rationale: str = Field(
        ...,
        description="简要说明判定依据，可引用上下文中的关键信息或指出缺失/矛盾",
    )


class StatementVerificationResponse(BaseModel):
    """LLM 结构化输出：声明验证结果列表"""

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    verifications: List[VerifiedStatement] = Field(
        default_factory=list, description="每条声明的验证结论"
    )


class FaithfulnessEvaluationResult(BaseModel):
    """最终忠实度评估结果"""

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    statements: List[ExtractedStatement] = Field(
        default_factory=list, description="提取出的声明"
    )
    verifications: List[VerifiedStatement] = Field(
        default_factory=list, description="声明验证结果"
    )
    faithfulness: float = Field(..., description="0-1 之间的忠实度分数")


class CustomFaithfulnessEvaluator:
    """
    基于“声明提取-声明验证-打分”流程的忠实度评估器。
    - 使用 utils.ainvoke_llm_with_structured_output 进行两次模型调用
    - 评分公式：Faithfulness = 支持的声明数 / 总声明数
    """

    def __init__(self, model_id: str, context_token_limit: int | None = None):
        cfg = create_default_configs()
        self.model_id = model_id
        self.context_token_limit = context_token_limit or cfg.context_token_limit

    @trace_span("_extract_statements")
    async def _extract_statements(self, question: str, answer: str) -> List[ExtractedStatement]:
        """步骤1：声明提取"""
        prompt = self._build_extraction_prompt(question, answer)
        try:
            resp = await ainvoke_llm_with_structured_output(
                prompt, model_id=self.model_id, response_type=StatementExtractionResponse
            )
            return resp.statements
        except Exception as e:
            log_error_type_and_message(
                logger=logger,
                context="声明提取失败",
                e=e,
                exc_info=False,
                max_message_chars=200,
            )
            return []

    @trace_span("_verify_statements")
    async def _verify_statements(
        self, statements: List[ExtractedStatement], context_text: str
    ) -> List[VerifiedStatement]:
        """步骤2：声明验证"""
        prompt = self._build_verification_prompt(statements, context_text)
        try:
            resp = await ainvoke_llm_with_structured_output(
                prompt,
                model_id=self.model_id,
                response_type=StatementVerificationResponse,
            )
            return resp.verifications
        except Exception as e:
            log_error_type_and_message(
                logger=logger,
                context="声明验证失败",
                e=e,
                exc_info=False,
                max_message_chars=200,
            )
            return []

    def _build_extraction_prompt(self, question: str, answer: str) -> str:
        """构造声明提取提示词"""
        prompt = apply_prompt_template(
            "faithfulness_statement_extraction",
            {"question": question, "answer": answer},
        )
        return ensure_within_token_limit(
            prompt, max_tokens=self.context_token_limit
        )

    def _build_verification_prompt(
        self, statements: List[ExtractedStatement], context_text: str
    ) -> str:
        """构造声明验证提示词，并控制上下文长度"""
        prompt = apply_prompt_template(
            "faithfulness_statement_verification",
            {
                "context_text": context_text,
                "statements": statements,
            },
        )

        return ensure_within_token_limit(
            prompt, max_tokens=self.context_token_limit
        )

    @trace_span("evaluate_faithfulness")
    async def evaluate(
        self, question: str, answer: str, context_texts: List[str]
    ) -> FaithfulnessEvaluationResult:
        """外部主入口：完成提取、验证并计算分数"""
        statements = await self._extract_statements(question, answer)
        verifications = await self._verify_statements(statements, "\n".join(context_texts))

        total = len(statements)
        supported_cnt = sum(1 for v in verifications if v.supported)
        score = supported_cnt / total if total > 0 else 0.0

        return FaithfulnessEvaluationResult(
            statements=statements, verifications=verifications, faithfulness=score
        )

