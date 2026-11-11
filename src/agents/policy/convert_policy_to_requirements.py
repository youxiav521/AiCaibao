"""
政策文件转换为结构化需求 (V7 平衡判定版)
主要改进：重新定义判定边界，平衡不同类型的识别，移除正则预判逻辑。
"""

import json
import os
import asyncio
import re
import sys
import time
from pathlib import Path

# 添加项目根目录到 sys.path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import List, Dict, Optional
from enum import Enum
from loguru import logger
from pydantic import BaseModel, Field

from src.utils.model_factory import ModelFactory
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser


class RequirementType(Enum):
    """需求类型"""
    TITLE = "title"
    PARAGRAPH = "paragraph"
    CONTROL = "control"
    SCOPE = "scope"


class LoanPhase(Enum):
    """贷款阶段"""
    PRE = "pre"
    DURING = "during"
    POST = "post"
    ALL = "all"
    BAD = "bad"


class Requirement:
    """结构化需求模型"""
    
    def __init__(
        self,
        req_id: str,
        content: str,
        req_type: RequirementType,
        loan_phases: List[LoanPhase],
        source_file: str,
        title_level: int = 0,
        page_idx: Optional[int] = None,
        parent_title: Optional[str] = None
    ):
        self.req_id = req_id
        self.content = content
        self.req_type = req_type
        self.loan_phases = loan_phases
        self.source_file = source_file
        self.title_level = title_level
        self.page_idx = page_idx
        self.parent_title = parent_title
    
    def to_dict(self) -> Dict:
        return {
            "req_id": self.req_id,
            "content": self.content,
            "type": self.req_type.value,
            "loan_phases": [phase.value for phase in self.loan_phases],
            "source_file": self.source_file,
            "title_level": self.title_level,
            "page_idx": self.page_idx,
            "parent_title": self.parent_title
        }


class ClassificationResult(BaseModel):
    """分类单条结果"""
    index: int = Field(description="段落索引")
    type: str = Field(description="节点类型")
    phases: List[str] = Field(description="贷款阶段列表")
    confidence: Optional[str] = Field(default=None, description="置信度")


class BatchClassificationResult(BaseModel):
    """批量分类结果"""
    results: List[ClassificationResult]


class PolicyToRequirementConverter:
    """政策文件转换器"""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.req_counter = 0
        
        # 初始化 LLM
        self.llm = ModelFactory.get_model(
            key="model_bailian_80b"
        )
        self.parser = JsonOutputParser(pydantic_object=BatchClassificationResult)
        
        # 系统提示词
        self.system_prompt = """你是资深银行业信贷合规专家。你的任务是精准判定段落的【类型】和【贷款阶段】。

### 一、核心判定原则

请准确区分：
- 什么是可执行的操作义务（Control）
- 什么是规则的边界条件（Scope）
- 什么是背景说明（Paragraph）
- 什么是结构标题（Title）

### 二、类型判定标准

#### 1. control (实质性风控要求)
必须同时满足以下3个条件：
- 明确的操作动作：核实、审查、审批、记录、报告、监控等
- 明确的执行标准：谁来做、什么时候做、做到什么程度
- 可被审计：可以检查是否执行、执行质量如何

正例：
- "信贷员应在3个工作日内完成现场尽调，并形成书面报告"
- "贷款发放前必须核实担保物权属证明的真实性"

反例：
- "应及时报告尽职调查结果"（"及时"无标准，无法审计）→ paragraph
- "商业银行应创造良好的授信环境"（原则性要求，无具体动作）→ paragraph

#### 2. scope (适用边界)
界定规则的适用范围、权限限制、准入条件等。

典型场景：
- 准入条件/禁止清单："严禁向未获环保许可的企业放款"
- 适用范围："本办法适用于XX分行"、"生效日期为..."
- 权限界定："贷款人可参与借款人重大事项"
- 责任机制："对违规行为依法追究责任"
- 材料清单："需提交以下资料：..."

#### 3. paragraph (说明性描述)
提供上下文、定义、原则、建议，无强制操作义务。

典型场景：
- 术语定义："固定资产贷款是指..."
- 原则性陈述："应遵循依法合规、审慎经营的原则"
- 指导性建议："宜"、"鼓励"、"参照"

#### 4. title (层级标题)
仅起到章节索引作用，不包含实质性内容。

正例：
- "第一章 总则"
- "第二章 贷款种类、期限与利率"

### 三、阶段判定标准（多选策略）

1. pre (贷前)：准入、尽调、评级、审批、合同签订前。
2. during (贷中)：合同签署到资金发放。
3. post (贷后)：定期检查、风险预警、押品重估、档案管理。
4. bad (不良)：逾期催收、诉讼、核销、责任追究。
5. all (通用)：总则、定义、组织架构等。

注意：阶段可以多选，宁可多判不要漏判。

### 四、典型案例

案例1：责任追究
Input: "商业银行应根据评价结果，对具有以下情节的授信人员依法追究责任"
Output: {{"type": "scope", "phases": ["all"]}}
Reasoning: 界定责任机制的边界规则。

案例2：资料清单
Input: "客户需提交：营业执照、法人证书、财务报表..."
Output: {{"type": "scope", "phases": ["pre"]}}
Reasoning: 界定准入材料的边界条件。

### 五、输出要求

请使用 JSON 格式返回：
{
    "results": [
        {
            "index": int,
            "reasoning": "简短分析（30字内）",
            "type": "title/paragraph/control/scope",
            "phases": ["pre", "during", "post", "bad", "all"]
        }
    ]
}
"""

    def _is_valid_content(self, content: str) -> bool:
        """过滤无效内容"""
        if not content or len(content.strip()) < 3:
            return False
        
        if re.match(r'^[（）()、，。；：""'']+$', content.strip()):
            return False
        
        metadata_patterns = [
            r'^(中文名|外文名|英文名|别名|拼音|类型|性质|颁布时间|实施时间|颁布单位|发布机构)[:：]?$',
            r'^(发文字号|效力级别|时效性|主题分类|文号)[:：]?$',
            r'^[A-Za-z\s]+$',
            r'^\d{4}[-年]\d{1,2}[-月]\d{1,2}日?(起)?$',
            r'^www\.', r'^https?://',
        ]
        for pattern in metadata_patterns:
            if re.match(pattern, content.strip()):
                return False
        
        return True
    
    def _is_truncated(self, content: str) -> bool:
        """检查文本是否被截断"""
        content = content.strip()
        if not content:
            return False
        
        if not content[-1] in '。；！？）】"、：':
            if not re.match(r'.*[0-9%]$', content):
                return True
        
        if re.match(r'^[，、；：]', content):
            return True
        
        if re.search(r'(作，|况，|状，)$', content):
            return True
        
        return False
    
    def _should_merge(self, current: str, next_item: str) -> bool:
        """判断是否需要合并"""
        current = current.strip()
        next_item = next_item.strip()
        
        if not current or not next_item:
            return False
        
        if len(current) < 3 and current.isdigit():
            return False
        
        if re.match(r'^(\d+\.|[一二三四]、|\([0-9]+\)|（[0-9]+）|•|- )', next_item):
            if current[-1] not in '，,、;；':
                return False
        
        if self._is_truncated(current):
            if not re.match(r'^第[一二三四五六七八九十百千\d]+[章节条]', next_item):
                return True
        
        if current[-1] in '、，；':
            if not re.match(r'^[\(（]?[一二三四五六七八九十\d]+[\.、）\)]', next_item):
                return True
        
        return False
    
    def _merge_items(self, items: List[Dict]) -> List[Dict]:
        """合并文本片段"""
        if not items:
            return []
        
        merged = []
        i = 0
        merge_count = 0
        
        while i < len(items):
            current_item = items[i]
            current_content = current_item.get("content", "")
            
            merged_content = current_content
            original_idx = i
            
            while i + 1 < len(items):
                next_item = items[i + 1]
                next_content = next_item.get("content", "")
                
                if self._should_merge(merged_content, next_content):
                    merged_content += " " + next_content
                    i += 1
                    merge_count += 1
                else:
                    break
            
            merged_item = current_item.copy()
            merged_item["content"] = merged_content
            merged_item["merged_from"] = list(range(original_idx, i + 1))
            merged.append(merged_item)
            
            i += 1
        
        if merge_count > 0:
            logger.info(f"文本合并: {merge_count} 个片段 -> {len(merged)} 个完整段落")
        
        return merged

    async def _classify_batch_with_llm(self, batch_items: List[Dict], retry_count: int = 0) -> List[Dict]:
        """批量 LLM 分类"""
        if not batch_items:
            return []

        formatted_contents = []
        for item in batch_items:
            formatted_contents.append(
                f"--- Paragraph Index: {item['idx']} ---\n{item['content']}"
            )
        
        formatted_str = "\n\n".join(formatted_contents)
        max_retries = 3
        
        try:
            prompt = self.system_prompt
            messages = [SystemMessage(content=prompt), HumanMessage(content=formatted_str)]
            
            response = await asyncio.wait_for(self.llm.ainvoke(messages), timeout=90.0)
            content = response.content
            
            if "<think>" in content and "</think>" in content:
                content = content.split("</think>")[-1].strip()
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            
            try:
                parsed_data = json.loads(content)
            except json.JSONDecodeError:
                logger.warning("JSON 解析失败，尝试正则提取")
                results_match = re.search(r'"results":\s*(\[.*\])', content, re.DOTALL)
                if results_match:
                    parsed_data = {"results": json.loads(results_match.group(1))}
                else:
                    raise
            
            if isinstance(parsed_data, dict) and "results" in parsed_data:
                return parsed_data["results"]
            return []
        except asyncio.TimeoutError:
            logger.error("批量请求超时")
            return [{"index": item['idx'], "type": "paragraph", "phases": ["all"]} for item in batch_items]
        except Exception as e:
            logger.warning(f"批量请求失败: {e}")
            if retry_count < max_retries:
                wait_time = (retry_count + 1) * 2
                logger.info(f"正在重试第 {retry_count + 1} 次")
                await asyncio.sleep(wait_time)
                return await self._classify_batch_with_llm(batch_items, retry_count + 1)
            
            logger.error("处理失败")
            return [{"index": item['idx'], "type": "paragraph", "phases": ["all"]} for item in batch_items]

    async def convert_file(self, json_file: Path) -> List[Requirement]:
        """处理单个文件"""
        logger.info(f"处理文件: {json_file.name}")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = data.get("results", {})
        if not results: return []
        
        doc_data = list(results.values())[0]
        content_list = json.loads(doc_data.get("content_list", "[]"))
        
        raw_items = []
        filtered_count = 0
        
        for i, item in enumerate(content_list):
            content = item.get("text", "").strip()
            if item.get("type") == "list":
                list_items = item.get("list_items", [])
                if list_items: content = "\n".join(list_items)
            
            if not content or not self._is_valid_content(content):
                if content:
                    filtered_count += 1
                continue
            
            raw_items.append({
                "idx": i,
                "content": content,
                "item": item
            })
        
        if filtered_count > 0:
            logger.info(f"已过滤 {filtered_count} 条无效内容")
        
        processed_items = self._merge_items(raw_items)
        
        if not processed_items: 
            return []

        id_to_cls = {}
        batch_size = 2
        semaphore = asyncio.Semaphore(2)
        
        async def process_batch(batch, batch_idx):
            async with semaphore:
                if batch_idx > 0:
                    await asyncio.sleep(0.5)
                return await self._classify_batch_with_llm(batch)

        batches = [processed_items[i:i + batch_size] for i in range(0, len(processed_items), batch_size)]
        logger.info(f"LLM 分类: {len(processed_items)} 条，分 {len(batches)} 批")
        
        batch_results_list = await asyncio.gather(*[process_batch(b, idx) for idx, b in enumerate(batches)])
        
        for batch_res in batch_results_list:
            for res in batch_res:
                id_to_cls[res.get("index")] = res
        
        final_requirements = []
        current_title = None
        
        for p_item in processed_items:
            idx = p_item["idx"]
            item = p_item["item"]
            content = p_item["content"]
            cls = id_to_cls.get(idx, {"type": "paragraph", "phases": ["all"]})
            
            try:
                req_type = RequirementType(cls.get("type", "paragraph").lower())
            except ValueError:
                req_type = RequirementType.PARAGRAPH
            
            phases = []
            valid_phases = [e.value for e in LoanPhase]
            for p in cls.get("phases", ["all"]):
                if p.lower() in valid_phases:
                    phases.append(LoanPhase(p.lower()))
            if not phases: phases = [LoanPhase.ALL]
            
            self.req_counter += 1
            req = Requirement(
                req_id=f"REQ_{self.req_counter:06d}",
                content=content,
                req_type=req_type,
                loan_phases=phases,
                source_file=json_file.name,
                title_level=item.get("text_level", 0),
                page_idx=item.get("page_idx"),
                parent_title=current_title
            )
            
            if req.req_type == RequirementType.TITLE:
                current_title = req.content
            else:
                req.parent_title = current_title
                
            final_requirements.append(req)
            
        return final_requirements

    async def convert_directory(self, resume: bool = True, max_files: Optional[int] = None):
        """批量转换目录"""
        progress_file = self.output_dir / ".progress.json"
        processed_files = set()
        
        if resume and progress_file.exists():
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                    processed_files = set(progress_data.get("processed_files", []))
                    self.req_counter = progress_data.get("last_req_id", 0)
                    logger.info(f"继续处理: 已完成 {len(processed_files)} 个文件")
            except Exception as e:
                logger.warning(f"读取进度文件失败: {e}")
                processed_files = set()
        else:
            if self.output_dir.exists():
                logger.info("清理旧数据")
                for f in self.output_dir.glob("*"):
                    if f.is_file(): 
                        f.unlink()
        
        json_files = [f for f in self.input_dir.rglob("*.json") 
                      if "converted_output" not in str(f) and "requirements_output" not in str(f)]
        
        remaining_files = [f for f in json_files if f.name not in processed_files]
        
        if max_files is not None:
            remaining_files = remaining_files[:max_files]
            logger.info(f"限制处理前 {max_files} 个文件")
        
        logger.info(f"待处理 {len(remaining_files)} 个文件")
        
        all_requirements = []
        output_path = self.output_dir / "all_requirements.json"
        
        if output_path.exists():
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    all_requirements = existing_data
                    logger.info(f"已加载 {len(all_requirements)} 条历史数据")
            except Exception as e:
                logger.warning(f"加载已有结果失败: {e}")
        
        for idx, json_file in enumerate(remaining_files, 1):
            try:
                logger.info(f"[{idx}/{len(remaining_files)}] 处理中: {json_file.name}")
                file_reqs = await self.convert_file(json_file)
                
                all_requirements.extend([req.to_dict() for req in file_reqs])
                self._save_results(all_requirements)
                
                processed_files.add(json_file.name)
                self._save_progress(progress_file, list(processed_files), self.req_counter)
                
            except KeyboardInterrupt:
                logger.warning("用户中断")
                raise
            except Exception as e:
                logger.error(f"处理失败 {json_file.name}: {e}")
                continue
            
        logger.info(f"转换完成，总计 {len(all_requirements)} 条记录")
    
    def _save_progress(self, progress_file: Path, processed_files: List[str], last_req_id: int):
        """保存进度"""
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump({
                "processed_files": processed_files,
                "last_req_id": last_req_id,
                "timestamp": time.time()
            }, f, ensure_ascii=False, indent=2)

    def _save_results(self, requirements: List[Dict]):
        """保存结果"""
        output_path = self.output_dir / "all_requirements.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(requirements, f, ensure_ascii=False, indent=2)


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="政策文件需求提取工具")
    parser.add_argument("--no-resume", action="store_true", help="重新开始")
    parser.add_argument("--limit", type=int, default=None, help="限制处理数量")
    args = parser.parse_args()
    
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent
    input_dir = project_root / "storage" / "policy_parse_results"
    output_dir = project_root / "storage" / "requirements_output"
    
    converter = PolicyToRequirementConverter(str(input_dir), str(output_dir))
    resume = not args.no_resume
    
    await converter.convert_directory(resume=resume, max_files=args.limit)


if __name__ == "__main__":
    asyncio.run(main())
