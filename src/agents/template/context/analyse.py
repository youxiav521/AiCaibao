import re
from loguru import logger
from src.agents.template.context.abc import BaseNode
from src.agents.template.schema import Context, Question, QuestionDetail
from src.agents.template.utils import to_int
from src.agents.template.llm_utils import abatch_question_analysis

class AnalyseQuestion(BaseNode):
    """基于模板全文，对指定id位置进行明细分析。"""
    def __init__(self,context:Context=None, model_id: str = "model_bailian_80b") -> None:
        self.context = context
        self.model_id = model_id

    async def analyse_batch_async(self, 
    questions: list[Question], 
    use_english: bool = False,
    model_id: str = "model_bailian_80b") -> list[QuestionDetail]:
        """批量处理问题分析（异步版本）
        Args:
            questions: 问题列表
            model_id: 模型ID，默认为 "model_bailian_80b"
        Returns:
            处理后的问题详情列表（保证返回数量等于输入数量）
        """
        question_details = []
        texts_to_analyse = []
        questions_mapping = []  # 存储 (question, d, html_or_text) 的映射
        
        # 第一步：准备所有需要分析的文本
        for q in questions:
            d = QuestionDetail(**q.model_dump())
            description_text = q.description
            texts_to_analyse.append(description_text)

            questions_mapping.append(d)
            
        # 第二步：批量调用abatch_question_analysis
        if texts_to_analyse:
            logger.info(f"批量分析 {len(texts_to_analyse)} 个问题...")
            analysis_results = await abatch_question_analysis(
                texts_to_analyse, 
                use_english=use_english,
                model_id=model_id
            )
            
            # 第三步：处理分析结果
            for idx, d in enumerate(questions_mapping):
                analysis = analysis_results[idx]
                d.type = analysis.type
                d.description = analysis.description
                d.prompt = analysis.prompt
                logger.info(f"分析结果 {d.id} - 类型: {d.type}, 描述: {d.description}, 提示: {d.prompt[:50]}")
                question_details.append(d)
        else:
            # 如果没有需要分析的文本，仍然需要返回所有问题的基本信息
            logger.warning(f"批次中没有需要分析的文本，返回 {len(questions)} 个基本问题信息")
            for d in questions_mapping:
                d.type = 'text'  # 默认类型
                question_details.append(d)
            raise RuntimeError(f"texts_to_analyse len:{len(texts_to_analyse)};something went wrong while parsing context ")
        # 将所有处理后的问题详情加入到context中
        self.context.questions_detail.extend(question_details)
        
        # 确保返回数量与输入数量一致
        if len(question_details) != len(questions):
            logger.error(f"返回数量不匹配！输入: {len(questions)}, 返回: {len(question_details)}")
        
        return question_details
            
    def extract_html_qtag(self,id: str,full_text:str):
        _id = to_int(id)
        pattern = rf"&lt;q_{_id}&gt;(.*?)&lt;/q_{_id}&gt;"
        match = re.search(pattern, full_text, re.DOTALL)
        return match.group(1) if match else ""

    def get_sub_text(self,id: str,document_text:str) -> str:
        """
        通过id获取目标文本内容
        Args:
            id(str): 目标id
        """
        import re
        escaped_id = re.escape(id)
        # 构造正则表达式提取对应的内容
        tag_pattern = re.compile(rf"<{escaped_id}>(.*?)</{escaped_id}>", re.DOTALL | re.IGNORECASE)
        match = tag_pattern.search(document_text)
        if not match:
            return ""
        # 找到 <id> 的起始
        id_start = match.start()
        # 截取 <id> 之前的内容
        before = document_text[:id_start]
        # 查找 before 中最后一个类似 </q_xxx> 的标记
        close_tag_pattern = re.compile(r"</q_\d+>", re.IGNORECASE)
        prev_close_tag = None
        for m in close_tag_pattern.finditer(before):
            prev_close_tag = m
        # 计算片段开始
        chunk_start = prev_close_tag.end() if prev_close_tag else 0
        # 目标片段为 片段开始 到 </id> 结尾
        chunk_end = match.end()
        return document_text[chunk_start:chunk_end]

