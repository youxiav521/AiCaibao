from __future__ import annotations

from typing import Any, Dict
from .base import BaseSectionExtractor


class AuditExtractor(BaseSectionExtractor):
    @classmethod
    def section_key(cls) -> str:
        return "assurance"

    @classmethod
    def recall_keywords(cls):
        return [
        "审计报告", "审计意见", "审计意见类型", "非标准",
        "保留意见", "否定意见", "无法表示意见", "标准无保留意见",
        "强调事项", "与持续经营相关的重大不确定性",
        "关键审计事项",
        "审阅报告", "审阅结论", "未经审计"
    ]

    @classmethod
    def recall_params(cls):
        return {"max_hits": 6, "max_per_hit": 120}
    def title(self) -> str:
        return "审计/审阅信息"

    def schema_hint(self) -> Dict[str, Any]:
        return {"审计": {"value": None, "source_element_id": None}}

    def extra_rules(self) -> str:
        return (
f"""具体指令：
1. **核心任务**: 明确识别并提取本期财报中记载的审计意见类型，需从"标准无保留意见"、"带强调事项段的无保留意见"、"保留意见"、"否定意见"、"无法表示意见"五个标准选项中选择最精确的匹配项。
2. **查找优先级 (从高到低)**:
    - **最高优先级**: 提取包含"已出具[审计意见类型]审计报告"或"审计意见为[具体类型]"等明确表述的正文段落，且需确认该表述出现在审计报告正文主体部分而非附录或注释中。
    - **次高优先级**: 若未发现直接结论，需识别包含"非标准审计报告"或"特殊审计意见"等特定表述的标题段落，且该标题对应的正文内容需包含具体类型的说明。
    - **辅助判断**: 当出现"持续经营能力存在重大不确定性"、"重大事项需特别说明"等负面描述时，需记录该描述所属段落作为辅助判断依据。
3. **排除要素**: 必须过滤掉所有与审计程序执行、审计委员会职能、内控测试过程等无关的程序性内容，仅处理审计结论陈述部分。
4. **取值规范**: 当文本描述为"非标准审计报告"且解释具体原因为"持续经营能力问题"时，value字段必须赋值为"带持续经营重大不确定性段落的无保留意见"；若原文包含多个审计意见表述，取值应优先使用最明确且完整的描述。
5. **来源定位**: source_element_id参数必须映射到包含最终审计结论的正文段落，且该段落需包含完整的审计意见表述，而非仅作为章节标题的概括性描述。"""
)

    def parse(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {"审计": raw.get("审计", {"value": None, "source_element_id": None})}


