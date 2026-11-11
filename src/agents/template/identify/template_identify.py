"""
模板识别模块
实现文件上传和版面分析功能
"""
from loguru import logger
import os
import json
import aiofiles
from src.agents.template.identify.identify_agent.identify_llm import extract_sn_from_text
from src.agents.template.identify.models.meta import ParagraphMetadata, TableMetadata
from src.agents.template.identify.models.sn_items import SNItem
from src.agents.template.identify.utils import to_sn
from src.agents.template.utils import write_file, read_file
from src.agents.template.identify.layout_analysis import LayoutAnalysis
from src.agents.template.identify.docxtool import DocxQMarker

class TemplateIdentify:
    """模板识别处理器"""
    
    def __init__(self, use_meta_cache: bool = True):
        """
        初始化模板识别器
        Args:
            use_meta_cache: 是否使用元数据缓存
        """
        self.use_meta_cache = use_meta_cache
        self.meta = None
        self.file_path = None
        
    async def upload_file(self, file) -> str:
        """
        上传文件
        """
        logger.info(f"开始上传文件: {file.filename}")
        file_id = await write_file(file)
        logger.info(f"文件上传成功, file_id: {file_id}")
        return file_id
    
    async def get_file_path(self, file_id: str) -> str:
        """
        根据文件ID获取文件路径
        
        Args:
            file_id: 文件ID
            
        Returns:
            str: 文件完整路径
        """
        return await read_file(file_id)
    
    def analyze_layout(self, file_path: str):
        """
        执行文档版面分析
        """
        logger.info(f"开始版面分析: {file_path}")
        
        # 执行版面分析
        layout = LayoutAnalysis()
        self.meta = layout.base_analyze(file_path)
        self.file_path = file_path
        
        logger.info(f"版面分析完成, SN数量: {self.meta.sn_count}")
        return self.meta
    
    async def indentify_meta(self):
        """
        识别元数据，选择问题位置
        """
        if not self.meta or not self.meta.sn_metadatas:
            logger.warning("没有可用的 sn_metadatas 数据")
            return []
        
        # 遍历 sn_metadatas，提取每个 SN 的文本并调用 LLM 提取
        results = []
        for sn_meta in self.meta.sn_metadatas:
            if sn_meta.content and isinstance(sn_meta.content, TableMetadata):
                # 查看表格意图，判断是否为填充表
                intent = sn_meta.content.intent
                text = sn_meta.content.text
                # logger.info(f"表格内容 intent: {intent}")
                
                # 根据 intent 类型处理
                match intent[0] if intent and len(intent) > 0 else "DEFAULT":
                    case "STATIC_TABLE":
                        # STATIC_TABLE: 取所有 fill_in_points 的 SN 编号
                        # 获取所有唯一的行索引
                        # logger.info(f"STATIC_TABLE: 表格内容: {text}")
                        if text:
                            # 调用 LLM 提取 SN 位置
                            sn_items = await extract_sn_from_text(text)
                            results.extend(sn_items)
                    
                    case "DYNAMIC_TABLE":
                        # DYNAMIC_TABLE: 只取开始和结束的 fill_in_points 的 SN 编号
                        sn_item = self._extract_dynamic_table_sn(sn_meta)
                        results.append(sn_item)
                    
                    case "DEFAULT":
                        if text:
                            # 调用 LLM 提取 SN 位置
                            sn_items = await extract_sn_from_text(text)
                            results.extend(sn_items)
                            logger.info(f"SN {sn_meta.position} 提取完成，找到 {len(sn_items)} 个问题位置")
            
            # 提取文本内容
            elif sn_meta.content and isinstance(sn_meta.content, ParagraphMetadata):
                text = sn_meta.content.text
                if text:
                    # 调用 LLM 提取 SN 位置
                    sn_items = await extract_sn_from_text(text)
                    results.extend(sn_items)
                    logger.info(f"SN {sn_meta.position} 提取完成，找到 {len(sn_items)} 个问题位置")
        
        logger.info(f"共提取到 {len(results)} 个问题位置 SN")
        return results

    async def process_document(self, file_id: str = None, file = None):
        """
        处理文档的完整流程：上传 + 版面分析
        Args:
            file_id: 文件ID (如果已上传)
            file: 上传的文件对象 (如果需要上传)
        Returns:
            dict: 包含 file_id, marked_sn_file, sn_count 等信息
        """
        # 1. 如果提供了文件对象，先上传
        if file is not None:
            file_id = await self.upload_file(file)
        
        if not file_id:
            raise ValueError("必须提供 file_id 或 file")
        
        # 2. 获取文件路径
        file_path = await self.get_file_path(file_id)
        
        # 3. 执行版面分析
        meta = self.analyze_layout(file_path)
        ######
        # 构建保存目录
        save_dir = os.path.join(".tmp_llmops", "doc", "tmplateVars", "metas")
        os.makedirs(save_dir, exist_ok=True)

        # SN元数据转字典的函数
        def snmeta_to_dict(snmeta):
            d = {"position": snmeta.position}
            content = snmeta.content
            if isinstance(content, ParagraphMetadata):
                # 段落内容
                d["content"] = {"type": "ParagraphMetadata", "text": content.text}
            elif isinstance(content, TableMetadata):
                # 表格内容，只保存部分数据
                d["content"] = {
                    "type": "TableMetadata",
                    "text": getattr(content, "text", ""),
                    "text_overlap": getattr(content, "text_overlap", ""),
                    "intent": getattr(content, "intent", None),
                    "position": getattr(content, "position", None),
                    "df_shape": getattr(content, "df", None).shape if getattr(content, "df", None) is not None else None,
                }
            else:
                d["content"] = str(content)
            return d

        # 将 sn_metadatas 保存为 JSONL 格式（每行一个 JSON 对象）
        save_path_jsonl = os.path.join(save_dir, f"{file_id}.jsonl")
        sn_metadatas = getattr(meta, "sn_metadatas", [])
        async with aiofiles.open(save_path_jsonl, "w", encoding="utf-8") as f:
            for snm in sn_metadatas:
                snm_dict = snmeta_to_dict(snm)
                await f.write(json.dumps(snm_dict, ensure_ascii=False) + "\n")
        logger.info(f"SN元数据已保存为JSONL: {save_path_jsonl}")
        ######
        sn_items = await self.indentify_meta()
        
        # 5. 使用 DocxQMarker 生成识别后的文档
        identified_file_path = None
        if sn_items and meta.file_path:
            q_marker = DocxQMarker()
            identified_file_path = q_marker.mark(meta.file_path, sn_items)
            logger.info(f"Q标记完成，识别文件: {identified_file_path}")
        
        # 6. 返回结果
        return {
            "file_id": file_id,
            "marked_sn_file": meta.file_path,
            "identified_file": "identified_file_path",
            "sn_count": meta.sn_count,
            "sn_items": sn_items,
            "file_path_html": meta.file_path_html,
            "metadata": meta
        }
    
    def get_sn_metadata(self, sn: int):
        """
        获取指定SN编号的元数据
        
        Args:
            sn: SN编号
            
        Returns:
            SNMetadata: SN元数据对象
        """
        if not self.meta:
            raise ValueError("请先执行版面分析")
        return self.meta.get_sn_metadata(sn)
    
    def get_sn_text(self, sn: int) -> str:
        """
        获取指定SN编号对应的文本
        
        Args:
            sn: SN编号
            
        Returns:
            str: 文本内容
        """
        if not self.meta:
            raise ValueError("请先执行版面分析")
        return self.meta.get_sn_text(sn)
    
    def _extract_dynamic_table_sn(self, sn_meta) -> SNItem:
        """
        提取 DYNAMIC_TABLE 类型的 SN 编号
        只取开始和结束的 fill_in_points 的 SN 编号
        """
        start_sn = to_sn(sn_meta.position[0])
        end_sn = to_sn(sn_meta.position[1])
        sn_item = SNItem(answer_type="dynamic_table", start_sn=start_sn, end_sn=end_sn)
        logger.info(f"DYNAMIC_TABLE: 提取首尾 SN 编号 {start_sn}, {end_sn}")
        return sn_item


async def test_analyze_existing_file(file_id: str = None):
    """
    测试主函数：根据 file_id 执行版面分析
    Args:
        file_id: 已上传文件的 ID（通过 /api/template/upload 接口获取）
    """
    if not file_id:
        raise ValueError("必须提供 file_id 参数")
    
    print("=" * 60)
    print("模板识别测试")
    print("=" * 60)
    print(f"文件ID: {file_id}")
    print()
    
    # 1. 创建处理器
    processor = TemplateIdentify(use_meta_cache=False)
    
    # 2. 获取文件路径
    file_path = await processor.get_file_path(file_id)
    print(f"文件路径: {file_path}")
    print()
    
    # 3. 执行版面分析（异步调用）
    print("【步骤1】执行版面分析...")
    result = await processor.process_document(file_id=file_id)
    print(f"版面分析完成!")
    print()
    
    # 4. 输出结果详情
    print("【分析结果】")
    print(f"  SN总数: {result['sn_count']}")
    print()
    
    # 5. 输出存储位置
    print("【存储位置】")
    meta = result['metadata']
    
    # 原始上传文件
    print(f"  原始文件: {await processor.get_file_path(file_id)}")
    
    # SN标注后的文件
    if hasattr(meta, 'file_path') and meta.file_path:
        print(f"  SN标注文件: {meta.file_path}")
    
    # HTML文件
    if hasattr(meta, 'file_path_html') and meta.file_path_html:
        print(f"  HTML文件: {meta.file_path_html}")
    
    # 返回sn_items
    print(f"  SN标注结果: {result['sn_items']}")    
    return result['sn_items']


