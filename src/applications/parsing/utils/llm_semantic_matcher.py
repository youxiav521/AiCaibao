"""基于大模型的语义匹配工具（选择题模式）"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from loguru import logger

from src.applications.parsing.service.subject_matching.strategy import FinancialStatementLanguage
from src.utils.model_factory import ModelFactory

# 获取本模块的标准 logger
_logger = logging.getLogger(__name__)


class LLMSemanticMatcher:
    """基于大模型的语义匹配器。
    
    核心方法：select_best_match（选择题模式）
    使用短ID映射（1,2,3...）代替真实subjectId，节省token并减少幻觉。
    """
    
    def __init__(self, model_id: str = "model_for_table_classify", temperature: float = 0.01):
        """
        初始化语义匹配器
        
        Args:
            model_id: 模型ID
            temperature: 模型温度参数，越低越确定
        """
        self.model_id = model_id
        self.temperature = temperature
        self.model = None
        self._init_model()
    
    def _init_model(self):
        """初始化模型"""
        try:
            self.model = ModelFactory.get_model(
                key=self.model_id,
                temperature=self.temperature
            )
            _logger.info(f"LLM语义匹配器初始化成功，使用模型: {self.model_id}")
        except Exception as e:
            _logger.error(f"LLM语义匹配器初始化失败: {e}")
            # 不抛出异常，允许降级
            self.model = None

    async def select_best_match(
        self, 
        subject_text: str, 
        candidates: List[Dict[str, Any]], 
        context: Optional[str] = None,
        language: FinancialStatementLanguage = FinancialStatementLanguage.ZH,
    ) -> Optional[str]:
        """
        从候选列表中选出最匹配的科目ID（选择题模式，使用短ID映射）。
        
        Args:
            subject_text: 待匹配的科目文本
            candidates: 候选科目列表，每项包含 id（真实subjectId）和 name
            context: 上下文信息（如报表类型、口径等）
            
        Returns:
            匹配的真实 subjectId，无匹配返回 None
        """
        if not candidates:
            return None
            
        # === 构建短ID映射：短编号 -> 真实subjectId ===
        short_to_real = {}
        cand_lines = []
        for i, c in enumerate(candidates, 1):
            short_id = str(i)
            short_to_real[short_id] = c['id']
            source_tag = " [同义词]" if c.get('source') == 'synonym' else ""
            cand_lines.append(f"{short_id}. {c['name']}{source_tag}")
        
        cand_str = "\n".join(cand_lines)
        
        prompt = self._build_select_best_match_prompt(
            subject_text=subject_text,
            candidates_text=cand_str,
            context=context,
            language=language,
        )

        content = ""
        try:
            # 兼容 langchain 或直接调用
            if self.model and hasattr(self.model, "ainvoke"):
                response = await self.model.ainvoke(prompt)
                content = response.content if hasattr(response, 'content') else str(response)
            else:
                # 模型工厂初始化失败时，退回到统一的 JSON 调用方式。
                # _call_llm_json 内部使用 chain.invoke()（同步），必须隔离到线程以避免阻塞事件循环。
                from src.applications.parsing.extractor.base import _call_llm_json
                res_json = await asyncio.to_thread(_call_llm_json, self.model_id, prompt)
                if isinstance(res_json, dict):
                    short_id = str(res_json.get("id", ""))
                    confidence = float(res_json.get("confidence", 0.0))
                    return self._process_match_result(subject_text, short_id, confidence, short_to_real, candidates)
                return None

            content = content.strip()
            
            # 提取JSON
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            
            result = json.loads(content)
            if not isinstance(result, dict):
                _logger.debug(f"LLM返回非JSON对象({type(result).__name__}), 视为无匹配: '{subject_text}'")
                return None
            short_id = str(result.get("id", ""))
            confidence = float(result.get("confidence", 0.0))
            
            return self._process_match_result(subject_text, short_id, confidence, short_to_real, candidates)
            
        except json.JSONDecodeError as e:
            _logger.warning(f"LLM响应JSON解析失败: {content[:200]}, 错误: {e}")
            return None
        except Exception as e:
            _logger.warning(f"LLM科目匹配返回异常: subject='{subject_text}', error={type(e).__name__}: {e}")
            return None

    def _build_select_best_match_prompt(
        self,
        *,
        subject_text: str,
        candidates_text: str,
        context: Optional[str],
        language: FinancialStatementLanguage,
    ) -> str:
        if language == FinancialStatementLanguage.EN:
            return f"""You are a financial reporting expert familiar with IFRS and common English annual-report terminology. {f"Context: {context}" if context else ""}

Line item to match: "{subject_text}"

Candidate standard subjects ([synonym] means the candidate is an accepted alias of a standard subject):
{candidates_text}

Task: choose the one candidate that has the same accounting meaning as the line item.
Treat the following as a match:
- Exact wording or only minor wording/order/punctuation differences.
- Accepted accounting synonyms or common abbreviations, such as PPE -> Property, plant and equipment.
- IFRS/common annual-report terms that refer to the same concept.

Treat the following as not a match and return null:
- Opposite direction or nature, such as receivable vs payable, asset vs liability, income vs expense.
- Different maturity or classification, such as current vs non-current, operating vs investing vs financing.
- Different statement families.
- A broad heading or section title matched to a specific line item.
- A parent line item matched to an unrelated child line item, or the reverse.
- Similar words but different accounting meaning.
- Uncertain cases.

Always return a complete JSON object. Do not return a bare null literal.

Return JSON only: {{"id": candidate_number_or_null, "confidence": 0.0-1.0, "reason": "reason"}}
"""

        return f"""你是一位精通中国会计准则（CAS）的财务专家。{f"背景: {context}" if context else ""}

待匹配科目："{subject_text}"

候选标准科目（[同义词] 表示该项是标准科目的公认别名）：
{candidates_text}

任务：判断待匹配科目与哪个候选在会计语义上完全对应。
以下情况视为"对应"：
- 字面相同或仅有细微表述差异（如标点、顺序）
- 中国会计准则新旧版本的同一概念（如"递延税款借项" -> "递延所得税资产"）
- 公认的同义词或简称（如"应付工资" -> "应付职工薪酬"）

以下情况视为"不对应"，返回null：
- 方向相反（应收 vs 应付，资产 vs 负债）
- 期限不同（短期 vs 长期）
- 报表分类不同（经营活动 vs 投资活动 vs 筹资活动，三者之间不可互匹配）
- 资产负债表并列大类不可互匹配（流动资产、长期投资、固定资产、无形资产、其他长期资产各自独立）
- 仅共享宽泛词汇但含义不同（如"其他应收款" -> "其他流动资产"）
- 现金流量表科目必须严格区分业务性质，仅共享"收到""支付""现金""利息"等片段不构成匹配：
  · "支付利息、手续费及佣金的现金" ≠ "分配股利、利润或偿付利息支付的现金"（前者为经营活动，后者为筹资活动）
  · "收到担保费取得的现金" ≠ "销售商品、提供劳务收到的现金"（业务性质不同：担保费≠销售/劳务）
  · "收回投资收到的现金" ≠ "收到其他与经营活动有关的现金"（活动类别不同）
  · "经营/投资/筹资活动产生的现金流量"是大类活动标题行，不等价于该大类下任何具体收付款科目
- 净利润层级严格区分：
  · "净利润"（汇总行）≠ "持续经营净利润"（子项）：二者是父子包含关系，不可互相替代
  · "净利润" ≠ "非持续经营净利润" ； "持续经营净利润" ≠ "非持续经营净利润"
- 每股收益类科目与利润类科目截然不同，不可互匹配：
  · "基本每股收益"/"稀释每股收益" ≠ "净利润"/"营业利润"/"利润总额"/"归属于母公司股东的净利润"等任何利润类科目
  · 反之，任何净利润/利润类科目 ≠ 每股收益类科目
- “其中”“含”“附注”等前缀的明细行与其汇总科目不等价
- 不确定时

注意：无论是否匹配到，都必须返回完整的JSON对象，禁止直接返回 null 字面量。

请以JSON格式返回：{{"id": 编号或null, "confidence": 0.0-1.0, "reason": "匹配原因"}}

只返回JSON，不要返回其他内容。"""

    def _process_match_result(
        self, 
        subject_text: str, 
        short_id: str, 
        confidence: float, 
        short_to_real: dict, 
        candidates: list
    ) -> Optional[str]:
        """处理模型返回的匹配结果并进行置信度过滤"""
        # 置信度过滤：低于0.75认为不可靠
        if not short_id or short_id in ("None", "null", "") or confidence < 0.75:
            _logger.debug(f"LLM匹配置信度不足: '{subject_text}', id={short_id}, confidence={confidence:.2f}")
            return None
        
        # 短ID映射回真实subjectId
        real_id = short_to_real.get(short_id)
        if real_id:
            _logger.debug(f"LLM选择匹配: '{subject_text}' -> 编号{short_id} (confidence={confidence:.2f})")
            return real_id
        
        # 健壮性处理：模型可能返回了名称而非编号
        for c in candidates:
            if short_id.upper() == str(c['name']).strip().upper() or short_id == str(c['id']):
                _logger.debug(f"LLM返回了名称/原始ID而非编号，已纠正: '{subject_text}' -> '{c['name']}'")
                return c['id']
        
        _logger.warning(f"LLM返回的编号无效: '{short_id}', 有效范围: 1-{len(candidates)}")
        return None
