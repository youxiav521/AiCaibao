"""Excel 单元格显示值格式化工具。"""
from __future__ import annotations

from datetime import date, datetime, time
from typing import Any, Optional

from openpyxl.styles.numbers import is_date_format
from openpyxl.utils.datetime import from_excel


def excel_cell_to_display_text(cell: Any) -> str:
    """尽量把 Excel 单元格转换为接近界面显示效果的文本。"""
    value = getattr(cell, "value", None)
    if value is None:
        return ""

    if isinstance(value, datetime):
        return _format_datetime_for_excel_display(value)
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, time):
        return value.strftime("%H:%M:%S")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number_format = str(getattr(cell, "number_format", "") or "")
        if getattr(cell, "is_date", False) or is_date_format(number_format):
            formatted_date = _format_excel_serial_date(value, cell)
            if formatted_date:
                return formatted_date

        if _looks_like_excel_date_header_serial(cell):
            formatted_date = _format_excel_serial_date(value, cell)
            if formatted_date:
                return formatted_date

        if "%" in number_format:
            return _format_percentage_for_excel_display(value)

        if float(value).is_integer():
            return str(int(value))

    return str(value)


def _format_datetime_for_excel_display(value: datetime) -> str:
    if value.hour == 0 and value.minute == 0 and value.second == 0 and value.microsecond == 0:
        return value.strftime("%Y-%m-%d")
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _format_excel_serial_date(value: int | float, cell: Any) -> Optional[str]:
    try:
        workbook = cell.parent.parent
        converted = from_excel(value, epoch=workbook.epoch)
    except Exception:
        return None
    return _format_datetime_for_excel_display(converted)


def _looks_like_excel_date_header_serial(cell: Any) -> bool:
    value = getattr(cell, "value", None)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    if not float(value).is_integer():
        return False

    serial = int(value)
    if serial < 30000 or serial > 60000:
        return False

    return getattr(cell, "row", 0) <= 5


def _format_percentage_for_excel_display(value: int | float) -> str:
    percentage = float(value) * 100
    text = f"{percentage:.2f}".rstrip("0").rstrip(".")
    return f"{text}%"
