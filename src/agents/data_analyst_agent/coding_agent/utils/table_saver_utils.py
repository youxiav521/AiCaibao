import pandas as pd
import os
import time
import json
from typing import List, Dict, Any, Optional, Union, Tuple

from .table_parsers import parse_table_by_type
from ..models.document_model import Document
from loguru import logger


save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../code_executor/tmp_tables"))
MAX_FILE_PATH_LENGTH = 200


def _is_non_empty_string(value: Any) -> bool:
    """Check if value is a non-empty string after stripping whitespace."""
    return isinstance(value, str) and bool(value.strip())


def _truncate_by_bytes(s: str, max_bytes: int) -> str:
    """Truncate string by byte count to not exceed max_bytes (UTF-8 encoding)."""
    result = ""
    for char in s:
        char_bytes = len(char.encode('utf-8'))
        if len(result.encode('utf-8')) + char_bytes <= max_bytes:
            result += char
        else:
            break
    return result


def _ensure_file_name_length(file_name: str) -> str:
    """
    Truncate filename to ensure full path stays within system limits.
    Preserves timestamp suffix (last 13 digits) for uniqueness.
    """
    file_path = os.path.join(save_dir, file_name + ".json")
    current_bytes = len(file_path.encode('utf-8'))

    if current_bytes <= MAX_FILE_PATH_LENGTH:
        logger.info(f"--文件名长度检查--: 原始使用字节数 {current_bytes} 小于等于最大长度 {MAX_FILE_PATH_LENGTH}，直接返回")
        return file_name

    excess_bytes = current_bytes - MAX_FILE_PATH_LENGTH
    logger.info(f"--原始文件路径字节数超长触发截断--:  {current_bytes}>{MAX_FILE_PATH_LENGTH}")

    # Extract timestamp suffix if present (last 13 digits)
    timestamp = file_name[-13:] if len(file_name) >= 13 and file_name[-13:].isdigit() else ""
    base_name = file_name[:-13] if timestamp else file_name

    # Truncate base name to fit within byte limit
    max_base_bytes = max(0, len(base_name.encode('utf-8')) - excess_bytes)
    truncated_base = _truncate_by_bytes(base_name, max_base_bytes)

    result = truncated_base + timestamp
    logger.info(f"--文件名触发截断--: 截断结果为: {result}")
    return result


def _documents_to_dicts(documents: Union[List[Document], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Convert List[Document] or List[dict] to uniform List[dict] format."""
    if not documents:
        return []
    if isinstance(documents[0], Document):
        return [d.to_dict() for d in documents]
    return list(documents)


def _detect_table_type(table_str: str, default_type: str = "csv") -> str:
    """
    Detect table format from content heuristics.
    Returns detected type or default if unable to determine.
    """
    stripped = table_str.strip()
    if stripped.startswith(("{", "[")):
        return "json"
    if "\t" in table_str:
        return "tsv"
    if "|" in table_str and "---" in table_str:
        return "markdown"
    if "<table" in table_str.lower():
        return "html"
    # 检测 CSV：第一行包含逗号，且至少有两行数据
    lines = stripped.splitlines()
    if len(lines) >= 2 and "," in lines[0]:
        return "csv"
    return default_type


def _get_table_type(doc: Dict[str, Any], default_type: str = "csv") -> Optional[str]:
    """Get table type from doc metadata or detect from content."""
    explicit_type = doc.get("table_type")
    if explicit_type:
        return explicit_type

    table_value = doc.get("table")
    if isinstance(table_value, (dict, list)):
        return "json"
    # 注意：不检查 isinstance，只检查 strip 后非空，因为可能是其他类型转成的字符串
    if table_value is not None and str(table_value).strip():
        return _detect_table_type(str(table_value), default_type)

    return None


def _transfer_tables_from_documents(
    documents: Union[List[Document], List[Dict[str, Any]]],
    table_type: str = "csv"
) -> List[Dict[str, Any]]:
    """
    Extract tables from document list and parse to DataFrames.
    Each result dict contains: id, title, text, table (raw), dataframe (parsed)
    """
    documents = _documents_to_dicts(documents)
    results = []

    for doc in documents:
        table_value = doc.get("table", "")
        text_str = doc.get("text", "")

        # Skip if both table and text are empty/missing
        has_table_content = table_value is not None and (
            isinstance(table_value, (dict, list)) or _is_non_empty_string(table_value)
        )
        if not has_table_content and not _is_non_empty_string(text_str):
            continue

        # 调试日志：表格内容检查
        doc_title = doc.get("title", "")
        if has_table_content:
            if isinstance(table_value, (dict, list)):
                table_preview = f"dict/list, keys: {list(table_value.keys()) if isinstance(table_value, dict) else f'len={len(table_value)}'}"
            else:
                table_preview = f"str(len={len(str(table_value))}), first 100 chars: {str(table_value)[:100]}"
            logger.info(f"--发现表格内容--: 标题='{doc_title}', table类型={type(table_value).__name__}, 内容={table_preview}")
        else:
            logger.info(f"--无表格内容--: 标题='{doc_title}', table_value类型={type(table_value).__name__}, is_empty={not _is_non_empty_string(str(table_value))}")

        # Parse table if available
        df = None
        if has_table_content:
            if isinstance(table_value, (dict, list)):
                table_str = json.dumps(table_value, ensure_ascii=False)
            else:
                table_str = str(table_value)

            detected_type = _get_table_type(doc, table_type) or table_type
            df = parse_table_by_type(table_str, detected_type)

            # 调试日志：表格解析结果
            if df is None:
                logger.warning(f"--表格解析失败--: 标题='{doc.get('title')}', 检测类型={detected_type}, "
                              f"表格内容长度={len(table_str)}, 前50字符: {table_str[:50]}")
            else:
                logger.info(f"--表格解析成功--: 标题='{doc.get('title')}', 形状={df.shape}, 列名={list(df.columns)}")
        else:
            table_str = table_value

        results.append({
            "id": doc.get("id"),
            "title": doc.get("title"),
            "text": text_str,
            "table": table_str if has_table_content else table_value,
            "dataframe": df
        })

    return results


# Template definitions
_TABLE_DESCRIPT_TEMPLATE = """
- 相关表格名(可读取): {title}
表描述：{text}
表结构: {dataframe}
"""

_TABLE_NAMES_TEMPLATE_NO_COLUMNS = """
- 相关表格名(可读取): {title}
"""

_CONTEXT_TEMPLATE = """
- 相关上下文标题: {title}
相关上下文文本：{text}
{table}
"""

_CONTEXT_TEMPLATE_NO_TABLE = """
- 相关上下文标题: {title}
相关上下文文本：{text}
"""


def save_documents_to_json(
    documents: Union[List[Document], List[Dict[str, Any]]],
    save_dir: str = save_dir,
    table_type: str = "csv"
) -> List[Dict[str, Any]]:
    """
    Save tables from documents to JSON files and return metadata list.
    Supports List[Document] or List[dict] input format.
    """
    os.makedirs(save_dir, exist_ok=True)
    saved_info = []
    docs_with_dataframe = _transfer_tables_from_documents(documents, table_type=table_type)

    for table_info in docs_with_dataframe:
        df = table_info.get("dataframe")

        if df is not None:
            # Generate clean filename with uniqueness timestamp
            file_name = f"{table_info.get('id', 'unknown')}_{table_info.get('title', '')}"
            file_name = "".join(
                [c if c.isalnum() or c in ('_', '-') or '\u4e00' <= c <= '\u9fff' else '_' for c in file_name]
            )
            file_name = f"{file_name}_{int(time.time() * 1000)}"
            file_name = _ensure_file_name_length(file_name)
            file_path = os.path.join(save_dir, file_name + ".json")

            logger.info(f"---save_documents_to_json---: 临时表格保存路径：{file_path}")

            # Save using to_dict + json.dump to avoid precision issues with to_json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(df.to_dict(orient="split"), f, ensure_ascii=False, allow_nan=True)

            # Read back to verify and get preview
            new_df = pd.read_json(file_path, orient="split")
            dataframe_json = new_df.to_dict(orient="records") if not new_df.empty else None
            columns = [str(col) for col in list(new_df.columns)]
            columns_str = "[" + "，".join(columns) + "]"

            saved_info.append({
                "id": table_info.get("id"),
                "title": file_name,
                "text": table_info.get("text"),
                "table": table_info.get("table"),
                "columns": columns_str,
                "dataframe": dataframe_json,
                "file_path": file_path
            })
        else:
            saved_info.append({
                "id": table_info.get("id"),
                "title": table_info.get("title"),
                "text": table_info.get("text"),
                "table": table_info.get("table"),
                "columns": None,
                "dataframe": None,
                "file_path": None
            })

    return saved_info


def format_documents_context(saved_info: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Format saved document info into context strings for LLM consumption."""
    table_names = ""
    context_str = ""

    for doc in saved_info:
        title = doc.get("title", "")
        text = doc.get("text", "")
        if isinstance(text, list):
            text = "\n".join(str(t) for t in text)
        table = doc.get("table", "")
        dataframe = doc.get("dataframe")

        if dataframe is not None:
            table_descript = _TABLE_DESCRIPT_TEMPLATE.format(title=title, text=text, dataframe=dataframe)
            table_names += _TABLE_NAMES_TEMPLATE_NO_COLUMNS.format(title=title)
            context_str += table_descript
        elif _is_non_empty_string(text):
            context_str += _CONTEXT_TEMPLATE.format(title=title, text=text, table=table)
        else:
            context_str += _CONTEXT_TEMPLATE_NO_TABLE.format(title=title, text=text)

    return context_str, table_names


def get_text_docs(
    documents: Union[List[Document], List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Extract text content from documents, skipping empty entries."""
    documents = _documents_to_dicts(documents)
    doc_texts = []

    for doc in documents:
        text = doc.get("text")
        title = doc.get("title")
        if not _is_non_empty_string(text):
            continue
        doc_texts.append({"title": title, "text": text})

    return doc_texts


def cleanup_saved_tables(saved_info: List[Dict[str, Any]]) -> int:
    """
    Clean up temporary table files from saved_info list.

    Args:
        saved_info: List of saved document info, each containing 'file_path'

    Returns:
        Number of files successfully deleted
    """
    if not saved_info:
        return 0

    deleted_count = 0
    for doc in saved_info:
        file_path = doc.get("file_path")
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                logger.warning(f"删除临时表格文件失败 {file_path}: {str(e)}")

    if deleted_count > 0:
        logger.info(f"已清理 {deleted_count} 个临时表格文件")
    return deleted_count



def main():
    """Simple unit test for table saving functionality."""
    test_documents = [{
        "id": 1,
        "title": "__公司简介_营业执照_有限责任公司登记基本情况表_利润及利润分配表_增值税及附加税费申请表_个人信用报告_授信业务准入查询报告_其他应收款_预付账款_应收账款_实收资本_固定资产明细表格等",
        "table": "姓名,年龄,城市,职业,薪资,年发薪次数\n张三,28,北京,软件工程师,15000,10\n李四,32,上海,产品经理,18000,12\n王五,25,深圳,设计师,12000,8\n赵六,30,杭州,数据分析师,16000,15",
        "text": ""
    }]
    print(save_documents_to_json(test_documents))


if __name__ == "__main__":
    main()
