from decimal import Decimal, getcontext, ROUND_HALF_UP
from typing import Union, Dict, Any
import re

# =========================
# 全局高精度配置
# =========================
getcontext().prec = 50
getcontext().rounding = ROUND_HALF_UP

try:
    import numpy as np
    _numpy_number_types = (
        getattr(np, "integer", ()),
        getattr(np, "floating", ()),
        getattr(np, "number", ()),
    )
except ImportError:
    _numpy_number_types = ()

NumberLike = Union[int, float, str, Decimal]

def _to_decimal(value: NumberLike) -> Decimal:
    """
    输入统一转换为 Decimal
    支持 '3,339,873.25' 这种字符串
    兼容 numpy 数值类型
    尽量避免 decimal.InvalidOperation，例如空字符串、None、非数值等输入
    """
    if value is None:
        raise ValueError("不能将 None 转换为 Decimal。请确保所有输入都是有效的数字。")
    if isinstance(value, Decimal):
        return value
    # numpy类型支持
    if _numpy_number_types and isinstance(value, _numpy_number_types):
        # numpy的整数类型
        if hasattr(value, "dtype") and str(value.dtype).startswith("int"):
            return Decimal(int(value))
        # numpy的浮点数类型
        elif hasattr(value, "dtype") and str(value.dtype).startswith("float"):
            return Decimal(str(value))
        # fallback: 尝试转float/转int再Decimal
        else:
            return Decimal(str(value))
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        # 把float转为str再转Decimal，避免float精度丢失
        return Decimal(str(value))
    if isinstance(value, str):
        cleaned = re.sub(r",", "", value).strip()
        # 检查空字符串或无效字符串
        if cleaned == "" or cleaned.lower() in ("none", "nan", "null"):
            raise ValueError(f"无法将空字符串或无效字符串 '{value}' 转换为 Decimal，请检查输入数据。")
        try:
            # 正则检查是否为数值
            if not re.match(r"^[-+]?\d*\.?\d+(e[-+]?\d+)?$", cleaned, flags=re.IGNORECASE):
                raise ValueError(f"字符串 '{value}' 无法转换为有效的Decimal，请检查数据。")
            return Decimal(cleaned)
        except Exception as ex:
            raise ValueError(f"无法将字符串 '{value}' 转换为 Decimal。异常原因: {ex}")

    raise TypeError(f"不支持的输入类型: {type(value)}")

def quantize(value: NumberLike, scale: int) -> Decimal:
    d_value = _to_decimal(value)
    quant = Decimal('1').scaleb(-scale)
    return d_value.quantize(quant)

def add(a: NumberLike, b: NumberLike, scale: int = None) -> Decimal:
    da, db = _to_decimal(a), _to_decimal(b)
    result = da + db
    if scale is not None:
        result = quantize(result, scale)
    return result

def sub(a: NumberLike, b: NumberLike, scale: int = None) -> Decimal:
    da, db = _to_decimal(a), _to_decimal(b)
    result = da - db
    if scale is not None:
        result = quantize(result, scale)
    return result

def mul(a: NumberLike, b: NumberLike, scale: int = None) -> Decimal:
    da, db = _to_decimal(a), _to_decimal(b)
    result = da * db
    if scale is not None:
        result = quantize(result, scale)
    return result

def div(a: NumberLike, b: NumberLike, scale: int = 10) -> Decimal:
    da, db = _to_decimal(a), _to_decimal(b)
    # 检查除数是否为0，使用is_zero()方法确保能正确识别所有形式的0（0, 0.0, "0", "0.0"等）
    if db.is_zero():
        raise ValueError(f"除法运算错误：除数为0。被除数: {a}, 除数: {b}。请检查数据，确保除数不为0。")
    raw = da / db
    result = quantize(raw, scale)
    return result

def main():
    # 多步运算示例：((a + b) * 3 - 5000) / 2，最后保留两位小数

    a = "1,234.56"
    b = "23,435,378.9"

    # 第一步：a + b
    sum_result = add(a, b)
    # 第二步：(a + b) * 3
    mul_result = mul(sum_result, 3)
    div_result = div(sub(mul_result, 5000), "0.00", scale=2)

    print("最终结果:", div_result)

if __name__ == "__main__":
    main()

