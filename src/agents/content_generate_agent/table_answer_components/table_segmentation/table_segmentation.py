"""
表格拆分核心模块

提供将复杂表格拆解为简单关系表的功能。
"""


from ...input_models import TableItem
from ...utils import (
    apply_prompt_template,
    ainvoke_llm_with_structured_output,
    ainvoke_llm_with_prompt,
    log_error_type_and_message,
)
from ..models import Tables, TablesWithHTML
from loguru import logger


async def asegment_table_into_relations(
    item: TableItem,
    model_id: str = "model_for_content_generation",
) -> Tables:
    """
    将复杂表格拆解为一组简单的关系表（仅返回行号分组）

    Args:
        item: 表格项
        model_id: 使用的模型ID，默认为 "model_for_content_generation"

    Returns:
        Tables 对象，包含拆分后的行号分组列表
    """
    if not isinstance(item, TableItem):
        raise TypeError(f"item 必须为 TableItem 类型，实际为 {type(item)}")

    # 提取 HTML
    table_html = item.html or ""
    if not table_html:
        logger.warning("表格项缺少 html 字段，返回空结果")
        return Tables(table_list=[])

    # 加载提示词模板
    try:
        prompt = apply_prompt_template(
            "table_segmentation",
            {
                "table_html": table_html,
            },
        )
    except Exception as e:
        log_error_type_and_message(
            logger=logger,
            context="加载提示词模板失败",
            e=e,
            exc_info=False,
            max_message_chars=200,
        )
        return Tables(table_list=[])

    # 调用模型
    try:
        result = await ainvoke_llm_with_structured_output(
            prompt,
            model_id,
            Tables,
        )
        logger.info(f"表格拆分成功，得到 {len(result.table_list)} 个关系表")
        return result
    except Exception as e:
        log_error_type_and_message(
            logger=logger,
            context="表格拆分失败",
            e=e,
            exc_info=False,
            max_message_chars=200,
        )
        # 返回空结果作为兜底
        return Tables(table_list=[])

# 添加方法来直接支持问题表格
async def asegment_html_table_into_relations(
    table_html: str,
    model_id: str = "model_for_content_generation",
) -> Tables:
    """
    将复杂表格拆解为一组简单的关系表（仅返回行号分组）
    Args:
        table_html: 表格 HTML 文本
        model_id: 使用的模型ID，默认为 "model_for_content_generation"
    Returns:
        Tables 对象，包含拆分后的行号分组列表
    """
    if not isinstance(table_html, str):
        raise TypeError(f"table_html 必须为 str 类型，实际为 {type(table_html)}")

    if not table_html:
        logger.warning("缺少表格 html 文本，返回空结果")
        return Tables(table_list=[])

    # 加载提示词模板
    try:
        prompt = apply_prompt_template(
            "table_segmentation",
            {
                "table_html": table_html,
            },
        )
    except Exception as e:
        log_error_type_and_message(
            logger=logger,
            context="加载提示词模板失败",
            e=e,
            exc_info=False,
            max_message_chars=200,
        )
        return Tables(table_list=[])

    # 调用模型
    try:
        result = await ainvoke_llm_with_structured_output(
            prompt,
            model_id,
            Tables,
        )
        logger.info(f"表格拆分成功，得到 {len(result.table_list)} 个关系表")
        return result
    except Exception as e:
        log_error_type_and_message(
            logger=logger,
            context="表格拆分失败",
            e=e,
            exc_info=False,
            max_message_chars=200,
        )
        # 返回空结果作为兜底
        return Tables(table_list=[])

# 废弃的方式，不使用
async def asegment_table_into_relations_with_html(
    item: TableItem,
    model_id: str = "model_for_content_generation",
) -> TablesWithHTML:
    """
    将复杂表格拆解为一组简单的关系表（返回行号分组 + 生成的HTML）

    Args:
        item: 表格项
        model_id: 使用的模型ID，默认为 "model_for_content_generation"

    Returns:
        TablesWithHTML 对象，包含拆分后的行号分组和对应的HTML表格列表
    """
    if not isinstance(item, TableItem):
        raise TypeError(f"item 必须为 TableItem 类型，实际为 {type(item)}")

    # 提取 HTML
    table_html = item.html or ""
    if not table_html:
        logger.warning("表格项缺少 html 字段，返回空结果")
        return TablesWithHTML(table_list=[])

    # 加载提示词模板
    try:
        prompt = apply_prompt_template(
            "table_segmentation_with_html",
            {
                "table_html": table_html,
            },
        )
    except Exception as e:
        log_error_type_and_message(
            logger=logger,
            context="加载提示词模板失败",
            e=e,
            exc_info=False,
            max_message_chars=200,
        )
        return TablesWithHTML(table_list=[])

    # 调用模型
    try:
        result = await ainvoke_llm_with_structured_output(
            prompt,
            model_id,
            TablesWithHTML,
        )
        logger.info(f"表格拆分成功，得到 {len(result.table_list)} 个关系表（含HTML）")
        return result
    except Exception as e:
        log_error_type_and_message(
            logger=logger,
            context="表格拆分失败",
            e=e,
            exc_info=False,
            max_message_chars=200,
        )
        # 返回空结果作为兜底
        return TablesWithHTML(table_list=[])

