"""
财报结构化提取路由
"""
import json
import os
from fastapi import APIRouter, HTTPException, Header
from typing import List, Optional
from loguru import logger

from src.applications.doc_extract.schema.fin_extract import FinExtractRequest, FinExtractResult
from src.applications.doc_extract.services.doc_extract_services import DocExtractService
from src.applications.doc_extract.utils.vlm_file_utils import download_files_from_minio
from src.agents.content_generate_agent.utils import resolve_answer_locale

router = APIRouter(prefix="/api/doc/fin_extract", tags=["财报结构化提取"])


# 固定的财报提取规则
FIN_EXTRACT_RULES = [
    {
        "extractId": "fin_extract_001",
        "extractName": "财报基础信息",
        "extractPrompt": "上传材料均属于财报内容。提取财报基础信息时严禁猜测或补全；字段没有明确证据时返回null，具体提取口径以JSON Schema字段描述为准。企业全称必须取文档中的完整法定名称原文，若只出现简称、品牌名或缩写，不得回填为企业全称。",
        "jsonSchema": '{"type":"object","properties":{"财报基础信息":{"type":"object","properties":{"基础信息":{"type":"object","properties":{"企业全称":{"type":"string","description":"必须返回文档原文中出现的公司或企业完整法定全称，通常应包含公司、有限公司、股份有限公司、集团等完整主体后缀；优先逐字提取编制单位、公司名称、企业名称、单位名称等标签后的完整文字；若名称跨行展示，需要拼接成完整全称；禁止输出简称、品牌名、股票简称、证券简称、缩写或仅包含公司名的一部分，例如看到贵州茅台酒股份有限公司时不得输出贵州茅台；不要把标签本身作为值；无法确认完整法定全称时返回null。"},"证券代码":{"type":"string","description":"仅当页面存在证券代码、股票代码、A股代码、港股代码、债券代码等明确标签及其对应值时提取；禁止从金额、日期、页码、附注编号、表格项目编号中猜测或截取；未出现明确标签时返回null。"},"证券简称":{"type":"string","description":"仅从证券简称、股票简称、简称等明确标签后的值提取；禁止根据企业名称或其他文字推断；未出现明确标签时返回null。"},"货币类型":{"type":"string","description":"仅返回人民币、美元等，如果识别为元则返回人民币"},"审计信息":{"type":"string","description":"内控审计报告意见类型"}}}}}}}',
        "namingRule": ""
    }
]

# 英文版财报提取规则
FIN_EXTRACT_RULES_EN = [
    {
        "extractId": "fin_extract_001",
        "extractName": "Financial Report Basic Info",
        "extractPrompt": "The uploaded materials are all financial report content. Do not guess or complete financial report basic information; return null when there is no explicit evidence, and follow each JSON Schema field description for extraction details. Field keys in the output objectValue must exactly match the JSON Schema property names (e.g. 'Company Full Name', 'Securities Code', 'Securities Abbreviation', 'Currency Type', 'Audit Info'); do not rephrase or translate the keys.",
        "jsonSchema": r'{"type":"object","properties":{"Financial Report Basic Info":{"type":"object","properties":{"Basic Info":{"type":"object","properties":{"Company Full Name":{"type":"string","description":"Return only the complete legal company or entity name exactly appearing in the document, usually including suffixes such as Company, Co., Ltd., Limited, Corporation, Group, Inc., or PLC; prefer the full text after labels such as Company Name, Company Full Name, Prepared by, Reporting Entity, Entity Name, Issuer, or Unit Name; if the name spans multiple lines, concatenate it into the complete legal name; do not output abbreviations, brand names, stock short names, securities abbreviations, ticker symbols, or partial company names; do not include the label itself; return null if the complete legal name cannot be confirmed."},"Securities Code":{"type":"string","description":"Return only the securities / stock / bond code itself (digits or ticker) as it appears next to an explicit label such as Securities Code, Stock Code, Security Code, Ticker, A-share Code, H-share Code, or Bond Code; do not include exchange prefix, market name (e.g. HKEX, SSE, SZSE, NASDAQ, NYSE), dot notation such as 000001.HK, or any other annotation; do not infer from amounts, dates, page numbers, note numbers, or table item numbers; return null if no explicit label is present."},"Securities Abbreviation":{"type":"string","description":"Return only the short name of the security as it appears next to an explicit label such as Securities Abbreviation, Stock Short Name, Security Short Name, Short Name, or Ticker Symbol; do not infer from the company name or other text; do not include the suffix such as Ltd, Inc, Corporation, Group, or the security code itself; return null if no explicit label is present."},"Currency Type":{"type":"string","description":"Return the currency type as it appears in the document, e.g. OMR, R.O., CNY, USD. Do not include the amount unit (One, Thousands, Millions, Billions) here. Return null if no currency is found."},"Audit Info":{"type":"string","description":"Return the type of audit opinion as stated in the audit report, e.g. Unqualified Opinion, Qualified Opinion, Adverse Opinion, Disclaimer of Opinion, Unqualified Opinion with Emphasis of Matter, or Unaudited for interim/quarterly reports; if the audit report uses a different wording, return that wording verbatim; return null if no audit information is present."}}}}}}}',
        "namingRule": ""
    }
]


def _get_rules(locale: str) -> list:
    """根据 locale 选择中文或英文提取规则。"""
    return FIN_EXTRACT_RULES_EN if locale == "en_US" else FIN_EXTRACT_RULES


def _err(locale: str, zh: str, en: str) -> str:
    """根据 locale 返回对应语言的错误消息。"""
    return en if locale == "en_US" else zh



@router.post(
    "",
    summary="财报结构化提取",
    response_model=List[FinExtractResult]
)
async def extract_fin_report(
    request: FinExtractRequest,
    accept_language: Optional[str] = Header(None, alias="Accept-Language"),
) -> List[FinExtractResult]:
    """
    使用固定规则对财报进行结构化提取，支持 LLM/VLM 两种模式。

    模式说明：
    - vlm：适用于单图片、不提取表格或表格内容简单的场景（需要minio + files）
    - llm：适用于复杂表格提取或长文档场景（需要docContent）
    """
    locale = resolve_answer_locale(accept_language)
    extract_rules = _get_rules(locale)
    try:
        logger.info(f"收到财报提取请求，模式: {request.mode}, locale: {locale}")

        if request.mode == "llm":
            # ---- LLM 模式 ----
            if not request.docContent:
                raise HTTPException(status_code=400, detail=_err(locale, "llm模式下必须提供docContent", "docContent is required in llm mode"))

            doc_content = json.loads(request.docContent)
            convert_json = doc_content.get("convert_json", [])

            if not convert_json or len(convert_json) == 0:
                raise HTTPException(status_code=400, detail=_err(locale, "docContent中的convert_json不能为空", "convert_json in docContent cannot be empty"))

            logger.info(f"LLM模式，convert_json元素数量: {len(convert_json)}")

            # 从convert_json中提取element_info
            element_info = []
            allowed_types = {"table", "excel_table", "list", "text"}

            for item in convert_json:
                if isinstance(item, dict) and "element_id" in item and "type" in item and item.get("type") in allowed_types:
                    element_type = item.get("type")
                    if element_type in {"table", "excel_table"}:
                        element_content = item.get("rows", [])
                    else:
                        element_content = item.get("md", "")

                    element_data = {
                        "id": str(item.get("element_id")),
                        "content": element_content,
                        "type": element_type
                    }

                    if element_type == "table":
                        table_caption = item.get("table_caption")
                        if table_caption:
                            if isinstance(table_caption, list):
                                element_data["table_caption"] = ", ".join(str(c) for c in table_caption)
                            else:
                                element_data["table_caption"] = str(table_caption)

                    element_info.append(element_data)

            # 调用LLM提取服务（异步版本，避免阻塞事件循环）
            service = DocExtractService()
            results = await service.extract_async(
                extract_rules=extract_rules,
                file_id="",
                element_info=element_info
            )

        elif request.mode == "vlm":
            # ---- VLM 模式 ----
            file_paths = await download_files_from_minio(request.minio, request.files)
            logger.info("VLM模式，文件数量: {}", len(file_paths))

            from src.applications.doc_extract.services.doc_extract_service_v2 import DocExtractServiceV2

            service_v2 = DocExtractServiceV2()
            try:
                results = await service_v2.extract(
                    extract_rules=extract_rules,
                    mode="vlm",
                    file_paths=file_paths
                )
            finally:
                for f in file_paths or []:
                    try:
                        if os.path.exists(f):
                            os.remove(f)
                    except Exception:
                        logger.exception("清理临时文件失败 | path={}", f)

        else:
            raise HTTPException(status_code=400, detail=_err(locale, f"不支持的模式: {request.mode}", f"Unsupported mode: {request.mode}"))

        # 后处理
        # 转换为响应模型
        response_results = [
            FinExtractResult(
                extractId=result["extractId"],
                objectName=result["objectName"],
                objectValue=result["objectValue"],
                objectSource=result.get("objectSource")
            )
            for result in results
        ]

        logger.info(f"财报提取完成，模式: {request.mode}，返回结果数量: {len(response_results)}")
        return response_results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"财报提取过程中发生错误: {e}")
        raise HTTPException(status_code=500, detail=_err(locale, f"提取失败: {str(e)}", f"Extraction failed: {str(e)}"))
