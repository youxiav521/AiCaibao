import os
import base64
import asyncio
import hashlib
import tempfile
import subprocess
import zipfile
from typing import List, Dict
from pathlib import Path
from loguru import logger
import aiofiles
import fitz

from fastapi import HTTPException
from src.utils.model_factory import ModelFactory

# 常量配置
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
ARCHIVE_EXTENSIONS = {".zip"}
TMP_BASE = "/tmp/yusys-ai"
MAX_COMPARE_PAGE = 10


class MultimodalChatService:
    """统一问答服务：纯文本LLM / 多文件VLM"""

    def __init__(self):
        self.llm_model = ModelFactory.get_model(key="model_for_table_classify")
        self.vlm_model = ModelFactory.get_model(key="doc_extract")
        self.images_batch_size = 1
        os.makedirs(TMP_BASE, exist_ok=True)

    # 纯文本问答
    async def ask_without_files(self, question: str):
        resp = await self.llm_model.ainvoke(question)
        return resp.content

    # 多文件问答（VLM）
    async def ask_with_files(self, file_contents: List[Dict], question: str):
        all_images = []
        file_count = len(file_contents)
        for item in file_contents:
            tmp_path = await self._save_temp(item["content"], item["filename"])
            file_images = await self._process_file(tmp_path)
            all_images.extend(file_images)
        # 传给模型：文件总数 + 问题
        final_answer = await self._ask_vlm(
            image_paths=all_images,
            question=question,
            file_count=file_count
        )
        return final_answer.strip()

    async def _ask_vlm(
            self,
            image_paths: List[str],
            question: str,
            file_count: int = 1
    ):
        final_answer = ""
        is_compare_scene = file_count >= 2 or "对比" in question

        if is_compare_scene:
            use_images = image_paths[:MAX_COMPARE_PAGE]
            content = [{"type": "text", "text": question}]
            for img_path in use_images:
                async with aiofiles.open(img_path, "rb") as f:
                    data = await f.read()
                    b64 = base64.b64encode(data).decode()
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}
                    })

            messages = [
                {"role": "system",
                 "content": "你是专业的文档内容分析专家，基于提供的文档内容做客观分析解答。输出要求：1. 允许使用1.2.3数字序号、层级缩进、合理换行做结构化表达；2. 禁止分割线、表情图标、特殊符号、Markdown语法；3. 只输出纯文本结构化内容，不要多余开场白、客套话、多余解释；4. 内容逻辑完整、分点清晰，如实提炼文档真实信息。"},
                {"role": "user", "content": content}
            ]
            for attempt in range(3):
                try:
                    resp = await self.vlm_model.ainvoke(messages)
                    return resp.content.strip()
                except Exception as e:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2)


        # 单文件：逐页提取文字 → 最后只回答1次
        document_text = ""

        for img_path in image_paths:
            # content_msg = [
            #     {"type": "text", "text": "提取这一页的所有文字内容，不要回答任何问题，不要输出任何解释，只输出文字"},
            # ]
            content_msg = [
                {"type": "text",
                 "text": "完整描述当前页面全部内容，包含所有文字、人名、日期、落款、手写签字、红色公章、印章样式位置、表格、备注等一切可见信息，只如实记录页面全部内容，不要进行任何判断、不要回答业务问题、不要删减任何信息"},
            ]
            async with aiofiles.open(img_path, "rb") as f:
                data = await f.read()
                b64 = base64.b64encode(data).decode()
                content_msg.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                })

            messages = [
                {"role": "user", "content": content_msg}
            ]

            page_result = None
            for attempt in range(3):
                try:
                    resp = await self.vlm_model.ainvoke(messages)
                    page_result = resp.content.strip()
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2)

            if page_result:
                document_text += page_result + "\n"

        # 最终规则调用（不动）
        final_resp = await self.vlm_model.ainvoke([
            {"role": "system", "content": "严格按照要求输出，只输出最终答案，不要任何多余内容、解释、序号、重复内容"},
            {"role": "user", "content": f"文档内容：{document_text}\n\n{question}"}
        ])

        return final_resp.content.strip()
    # async def _ask_vlm(
    #         self,
    #         image_paths: List[str],
    #         question: str,
    #         file_count: int = 1
    # ):
    #     final_answer = ""
    #     is_compare_scene = file_count >= 2 or "对比" in question
    #
    #     # 多文件对比逻辑 完全保留原样
    #     if is_compare_scene:
    #         use_images = image_paths[:MAX_COMPARE_PAGE]
    #         content = [{"type": "text", "text": question}]
    #         for img_path in use_images:
    #             async with aiofiles.open(img_path, "rb") as f:
    #                 data = await f.read()
    #                 b64 = base64.b64encode(data).decode()
    #                 content.append({
    #                     "type": "image_url",
    #                     "image_url": {"url": f"data:image/png;base64,{b64}"}
    #                 })
    #
    #         messages = [
    #             {"role": "system",
    #              "content": "你是专业的文档内容分析专家，基于提供的文档内容做客观分析解答。输出要求：1. 允许使用1.2.3数字序号、层级缩进、合理换行做结构化表达；2. 禁止分割线、表情图标、特殊符号、Markdown语法；3. 只输出纯文本结构化内容，不要多余开场白、客套话、多余解释；4. 内容逻辑完整、分点清晰，如实提炼文档真实信息。"},
    #             {"role": "user", "content": content}
    #         ]
    #         for attempt in range(3):
    #             try:
    #                 resp = await self.vlm_model.ainvoke(messages)
    #                 return resp.content.strip()
    #             except Exception as e:
    #                 if attempt == 2:
    #                     raise
    #                 await asyncio.sleep(2)
    #
    #     all_content = ""
    #     for img_path in image_paths:
    #         content = [{"type": "text", "text": question}]
    #         async with aiofiles.open(img_path, "rb") as f:
    #             data = await f.read()
    #             b64 = base64.b64encode(data).decode()
    #             content.append({
    #                 "type": "image_url",
    #                 "image_url": {"url": f"data:image/png;base64,{b64}"}
    #             })
    #
    #         messages = [
    #             {"role": "system", "content": "提取本页所有有效信息，不要输出解释、原因、说明、多余文字，只保留事实内容"},
    #             {"role": "user", "content": content}
    #         ]
    #
    #         page_result = None
    #         for attempt in range(3):
    #             try:
    #                 resp = await self.vlm_model.ainvoke(messages)
    #                 page_result = resp.content.strip()
    #                 break
    #             except Exception as e:
    #                 if attempt == 2:
    #                     raise
    #                 await asyncio.sleep(2)
    #
    #         if page_result:
    #             all_content += page_result + "\n"
    #
    #     final_prompt = f"""
    #     以下是文档全部内容：
    #     {all_content}
    #
    #     用户需求：
    #     {question}
    #
    #     严格要求：只输出最终需要的答案，剔除所有无关说明、原因、解释、重复内容，只保留最终结果。
    #     """
    #
    #     final_resp = await self.vlm_model.ainvoke([
    #         {"role": "system", "content": "严格按照用户要求输出最终答案，禁止输出任何多余内容、解释、备注、序号"},
    #         {"role": "user", "content": final_prompt}
    #     ])
    #
    #     return final_resp.content.strip()
    # 多模态模型推理
    # async def _ask_vlm(
    #         self,
    #         image_paths: List[str],
    #         question: str,
    #         file_count: int = 1
    # ):
    #     final_answer = ""
    #     is_compare_scene = file_count >= 2 or "对比" in question
    #
    #     # 多文件对比场景：一次性给图，只调用一次模型
    #     if is_compare_scene:
    #         # 限制最大图片数量，防止超限爆炸
    #         use_images = image_paths[:MAX_COMPARE_PAGE]
    #
    #         content = [{"type": "text", "text": question}]
    #         for img_path in use_images:
    #             async with aiofiles.open(img_path, "rb") as f:
    #                 data = await f.read()
    #                 b64 = base64.b64encode(data).decode()
    #                 content.append({
    #                     "type": "image_url",
    #                     "image_url": {"url": f"data:image/png;base64,{b64}"}
    #                 })
    #
    #         messages = [
    #             {"role": "system",
    #              "content": "你是专业的文档内容分析专家，基于提供的文档内容做客观分析解答。输出要求：1. 允许使用1.2.3数字序号、层级缩进、合理换行做结构化表达；2. 禁止使用分割线、表情图标、特殊符号、Markdown加粗语法；3. 只输出纯文本结构化内容，不要多余开场白、客套话、多余解释；4. 内容逻辑完整、分点清晰，如实提炼文档真实信息。"},
    #             {"role": "user", "content": content}
    #         ]
    #         # 重试机制
    #         for attempt in range(3):
    #             try:
    #                 resp = await self.vlm_model.ainvoke(messages)
    #                 return resp.content.strip()
    #             except Exception as e:
    #                 if attempt == 2:
    #                     raise
    #                 await asyncio.sleep(2)
    #
    #     # 单文件场景：保留原有逐批投喂、找到就停
    #     remaining_needs = question
    #     for i in range(0, len(image_paths), self.images_batch_size):
    #         batch = image_paths[i:i + self.images_batch_size]
    #         content = [{"type": "text", "text": remaining_needs}]
    #
    #         for img_path in batch:
    #             async with aiofiles.open(img_path, "rb") as f:
    #                 data = await f.read()
    #                 b64 = base64.b64encode(data).decode()
    #                 content.append({
    #                     "type": "image_url",
    #                     "image_url": {"url": f"data:image/png;base64,{b64}"}
    #                 })
    #
    #         messages = [
    #             {"role": "system",
    #              "content": "你是专业的文档内容分析专家，基于提供的文档内容做客观分析解答。输出要求：1. 允许使用1.2.3数字序号、层级缩进、合理换行做结构化表达；2. 禁止使用分割线、表情图标、特殊符号、Markdown加粗语法；3. 只输出纯文本结构化内容，不要多余开场白、客套话、多余解释；4. 内容逻辑完整、分点清晰，如实提炼文档真实信息。"},
    #             {"role": "user", "content": content}
    #         ]
    #         page_result = None
    #         for attempt in range(3):
    #             try:
    #                 resp = await self.vlm_model.ainvoke(messages)
    #                 page_result = resp.content
    #                 break
    #             except Exception as e:
    #                 if attempt == 2:
    #                     raise
    #                 await asyncio.sleep(2)
    #         if page_result and page_result.strip():
    #             final_answer += "\n" + page_result.strip()
    #         # 单文件识别到答案就提前停止
    #         has_value = page_result is not None and len(page_result.strip()) > 0
    #         if has_value:
    #             remaining_needs = ""
    #         else:
    #             remaining_needs = question
    #         if not remaining_needs:
    #             break
    #     return final_answer.strip()
    # 文件处理逻辑
    async def _process_file(self, file_path: str) -> List[str]:
        ext = Path(file_path).suffix.lower()
        if ext in ARCHIVE_EXTENSIONS:
            return await self._extract_archive(file_path)
        if ext in IMAGE_EXTENSIONS:
            return [file_path]
        pdf_path = self._to_pdf(file_path)
        return self._pdf_to_images(pdf_path)

    def _to_pdf(self, input_path: str) -> str:
        if input_path.lower().endswith(".pdf"):
            return input_path
        out_dir = os.path.dirname(input_path)
        cmd = [
            "libreoffice", "--headless", "--norestore", "--nolockcheck",
            "--nodefault", "--nofirststartwizard",
            "--convert-to", "pdf", "--outdir", out_dir, input_path
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        return os.path.splitext(input_path)[0] + ".pdf"

    def _pdf_to_images(self, file_path: str) -> List[str]:
        doc = fitz.open(file_path)
        image_paths = []
        for i in range(len(doc)):
            page = doc.load_page(i)
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            md5 = self._get_file_md5(file_path)
            img_dir = os.path.join(TMP_BASE, f"doc/gen/pdf/images/{md5}")
            os.makedirs(img_dir, exist_ok=True)
            img_path = os.path.join(img_dir, f"{i + 1}.png")
            pix.save(img_path)
            image_paths.append(img_path)
        return image_paths

    async def _extract_archive(self, archive_path: str) -> List[str]:
        out_dir = archive_path + "_extract"
        os.makedirs(out_dir, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(out_dir)

        all_images = []
        for root, _, files in os.walk(out_dir):
            for file in files:
                path = os.path.join(root, file)
                ext = Path(path).suffix.lower()
                try:
                    if ext in IMAGE_EXTENSIONS:
                        all_images.append(path)
                    else:
                        pdf = self._to_pdf(path)
                        all_images.extend(self._pdf_to_images(pdf))
                except Exception:
                    continue
        return all_images

    async def _save_temp(self, content: bytes, filename: str):
        ext = Path(filename).suffix.lower() or ".tmp"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            temp_path = tmp.name
        async with aiofiles.open(temp_path, "wb") as f:
            await f.write(content)
        return temp_path

    def _get_file_md5(self, file_path: str):
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()