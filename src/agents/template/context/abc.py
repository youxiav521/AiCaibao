from abc import ABC
import aiofiles
from loguru import logger
from src.agents.template.utils import openai_api, tmp_path, write_jsonl
class BaseNode(ABC):

    async def write_jsonl(self, key: str, datas: list, model_id: str = None)->str:
        model_name = model_id or openai_api()[0]
        jsonl_path = tmp_path("dataset/jsonl/" + model_name) / f"{key}.jsonl"
        logger.info(f"Write jsonl: \n{jsonl_path}")
        await write_jsonl(jsonl_path, datas)
        return jsonl_path.as_posix()

    async def write_markdown(self, key: str, markdown: str):
        if not markdown:
            return
        p = tmp_path("dataset/markdown") / f"{key}.md"
        logger.info(f"Write markdown: {p}")
        async with aiofiles.open(p, "w", encoding="utf-8") as f:
            await f.write(markdown)

    def get_chunks(self,text:str,chunk_size=15000):
        """文本拆分，保证表格语义完整性"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter,MarkdownHeaderTextSplitter
        from langchain_core.documents import Document
        # 1) 按标题初步切块，保持章节语义
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
        )
        header_docs = header_splitter.split_text(text)

        # 2) 对每个章节做细分，避免超过模型上下限，同时保留表格
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=30,
            separators=[
                "\n\n|", "\n\n", "\n",  # 优先保留表格与段落
                "。", "！", "？", ".", "!", "?",  # 句子层面
                "；", ";", "，", ",", " "        # 词/空格
            ],
        )

        chunks: list[Document] = []
        for doc in header_docs:
            sub_docs = splitter.split_documents([doc])
            chunks.extend(sub_docs)
        logger.info(f"Split text into {len(chunks)} chunks,chunk_size={chunk_size}")
        for chunk in chunks:
            logger.info(f"Chunk预览...\nstart...\n{chunk.page_content[:50]}\nend...\n{chunk.page_content[-50:]}\n")
        return chunks
