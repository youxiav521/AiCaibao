"""
文档提取模型包
"""
from .common import ObjectSourceItem, ExtractRule
from .extract import ExtractRequest as ExtractV1Request, ExtractResult as ExtractV1Result
from .extract_v2 import (
    ExtractRequest as ExtractV2Request,
    ExtractResult as ExtractV2Result
)
from .fin_extract import FinExtractRequest, FinExtractResult

__all__ = [
    # 公共模型
    "ObjectSourceItem",
    "ExtractRule",
    
    # V1 模型
    "ExtractV1Request",
    "ExtractV1Result",
    
    # V2 模型
    "ExtractV2Request",
    "ExtractV2Result",
    
    # 财报模型
    "FinExtractRequest",
    "FinExtractResult",
]
