import re
from loguru import logger
from src.agents.template.context.abc import BaseNode
from src.agents.template.schema import Context, Question
from src.agents.template.utils import calculate_similarity, remove_q_tags, split_by_q_tags, to_int, to_int

class DocumentContext(BaseNode):
    def __init__(self, context: Context = None, model_id: str = "model_bailian_80b"):
        self.chunk_size = 20000
        self.context = context
        self.lines_q = None
        self.lines_toc = None
        self.model_id = model_id
        self._initialized = False

    async def init(self):
        """异步初始化方法"""
        if self._initialized:
            return
        await self.write_markdown(self.context.name, self.context.markdown)
        
        from src.agents.template.context.toc.toc import TocExtractor
        logger.info(f'开始结构提取目录，using model_id: {self.model_id}')
        self.context.toc = await TocExtractor().toc_level_async(
            remove_q_tags(self.context.markdown),
            model_id=self.model_id
        )
        await self.write_jsonl(f'TOC-{self.context.name}', self.context.toc, model_id=self.model_id)
        self._initialized = True

    def questions(self):
        questions:list[Question] = []
        
        for q in split_by_q_tags(self.context.markdown):
            q_id = to_int(q['id'])
            content = q['content']
            in_q_tag = self.extract_html_qtag(q['id'],self.context.html)
            description = f"""
单独提取<q_XX>标签中的内容如下：
<q_{q_id}>
{in_q_tag}
</q_{q_id}>

完整前后上下文内容如下，请准确判断类型，并完成提示信息生成任务
{content}
"""
            questions.append(Question(id=q['id'],description=description))
        
        self.context.questions = questions
        return questions

    def line_num_question(self):
        """问题行号信息"""
        self.lines_q = {}
        for idx,line in enumerate(self.context.markdown.split('\n')):
            if line.__contains__('q_'):
                for q in self.context.questions:
                    if line.__contains__(f'<{q.id}>'):
                        self.lines_q[q.id] = idx+1
        return self.lines_q

    def extract_html_qtag(self,id: str,full_text:str):
        _id = to_int(id)
        pattern = rf"&lt;q_{_id}&gt;(.*?)&lt;/q_{_id}&gt;"
        match = re.search(pattern, full_text, re.DOTALL)
        return match.group(1) if match else "" 

    def line_num_toc(self):
        """目录行号信息"""
        from src.agents.template.context.toc.find_line import match_array_to_lines_keywords
        self.lines_toc = []
        for i,r in enumerate(match_array_to_lines_keywords([t.text for t in self.context.toc],self.context.markdown)):
            self.lines_toc.append((self.context.toc[i],r[1]))
        return self.lines_toc

    def question_top_toc(self,q:Question):
        """方便UI从上到下展示:问题及上方目录,没有则返回None"""
        if not self.lines_q:
            self.lines_q = self.line_num_question()
            self.lines_toc = self.line_num_toc()
        n = self.lines_q.get(q.id)
        if not n:
            return

        toc = [t[0] for t in self.lines_toc if t[1] < n and calculate_similarity(t[0].text,q.description) < 0.5]
        if toc:
            self.lines_toc = [t for t in self.lines_toc if t[1] > n]
        return toc
    

