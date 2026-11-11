"""语义匹配工具"""

from typing import Optional
from difflib import SequenceMatcher
import re
from loguru import logger

from src.utils.chinese_converter import traditional_to_simplified

# ---------------------------------------------------------------------------
# Module-level normalization utilities (reusable by subject_match_service etc.)
# ---------------------------------------------------------------------------

_TRUNCATION_REPAIRS = {
    "一年内到期的长期负": "一年内到期的长期负债",
    "补 充 资": "补充资料",
}

# ---------------------------------------------------------------------------
# 新旧会计准则同义词组：每组第一个为规范形式，其余映射到第一个
# ---------------------------------------------------------------------------
_SYNONYM_GROUPS = [
    ("应收股息", "应收股利"),
    ("应付股利", "未付利润"),
    ("应交税费", "应交税金", "未交税金"),
    ("其他应交款", "其他未交款"),
    ("递延所得税资产", "递延税款借项"),
    ("递延所得税负债", "递延税款贷项"),
    ("应付职工薪酬", "应付工资"),
    ("应付账款", "应付帐款"),
    ("应收账款", "应收帐款"),
    ("坏账准备", "坏帐准备"),
]

_ACCOUNTING_CANONICAL: dict[str, str] = {}
for _grp in _SYNONYM_GROUPS:
    _canon = _grp[0]
    for _term in _grp[1:]:
        _ACCOUNTING_CANONICAL[_term] = _canon


def normalize_accounting_terms(text: str) -> str:
    """统一会计科目中常见的等价用词（新旧准则互通、OCR常见误读）"""
    text = traditional_to_simplified(text)
    # 1. 全文同义词映射（优先级最高）
    canonical = _ACCOUNTING_CANONICAL.get(text)
    if canonical:
        return canonical
    # 2. 子串替换
    text = text.replace("债券", "债权")
    text = text.replace("一年期长期", "一年内到期的长期")
    text = text.replace("负债及所有者权益", "负债和所有者权益")
    text = text.replace("负债及股东权益", "负债和股东权益")
    text = text.replace("现金及现金等价物", "现金和现金等价物")
    text = text.replace("帐", "账")
    return text


def repair_truncation(text: str) -> str:
    """修复常见的表格列宽不足导致的尾部截断"""
    return _TRUNCATION_REPAIRS.get(text, text)


def semantic_match(text1: str, text2: str, threshold: float = 0.95) -> bool:
    """
    语义匹配两个文本（全面优化版：更严格的财务科目匹配）
    
    优化要点：
    1. 默认阈值提高到0.95，避免误匹配
    2. 增强互斥词检测，避免相反含义的科目匹配
    3. 更严格的包含匹配逻辑（长度比例提高到80%）
    4. 增加层级检测，避免父科目匹配子科目
    
    Args:
        text1: 第一个文本（通常是标准科目名称）
        text2: 第二个文本（通常是表格中的科目名称）
        threshold: 相似度阈值，默认0.95（更严格）
        
    Returns:
        bool: 是否匹配成功
    """
    if not text1 or not text2:
        return False
    
    # 清理文本：去除空格、标点符号等
    def clean_text(text: str) -> str:
        text = re.sub(r'\s+', '', text)
        text = re.sub(r"[，。、；！？\u201c\u201d\u2018\u2019（）【】()]", '', text)
        text = traditional_to_simplified(text)
        return text.strip()

    cleaned_text1 = clean_text(text1)
    cleaned_text2 = clean_text(text2)
    cleaned_text1 = repair_truncation(cleaned_text1)
    cleaned_text2 = repair_truncation(cleaned_text2)
    cleaned_text1 = normalize_accounting_terms(cleaned_text1)
    cleaned_text2 = normalize_accounting_terms(cleaned_text2)
    
    if not cleaned_text1 or not cleaned_text2:
        return False
    
    # 完全匹配（最高优先级）
    if cleaned_text1 == cleaned_text2:
        return True
    
    # 【优化1】：增强互斥关键词检测（扩展列表，避免相反含义的科目匹配）
    exclusive_pairs = [
        ('流动', '非流动'),
        ('短期', '长期'),
        ('应收', '应付'),
        ('预收', '预付'),
        ('主营业务', '其他业务'),
        ('营业', '非营业'),
        ('营业内', '营业外'),
        ('递延所得税资产', '递延所得税负债'),
        ('资产', '负债'),  # 注意：只检测完全不同的大类
        ('收入', '支出'),
        ('收入', '成本'),
        ('收入', '费用'),
    ]
    
    for word1, word2 in exclusive_pairs:
        # 如果是大类词（如"资产"、"负债"），需要更严格的判断
        if word1 in ['资产', '负债', '收入', '支出', '成本', '费用']:
            # 只有当两个词单独出现（不是"递延所得税资产"这种复合词）时才互斥
            if (cleaned_text1 == word1 and word2 in cleaned_text2) or \
               (cleaned_text2 == word1 and word2 in cleaned_text1) or \
               (cleaned_text1 == word2 and word1 in cleaned_text2) or \
               (cleaned_text2 == word2 and word1 in cleaned_text1):
                return False
        else:
            # 普通互斥词，只要出现就互斥
            if (word1 in cleaned_text1 and word2 in cleaned_text2) or \
               (word2 in cleaned_text1 and word1 in cleaned_text2):
                return False
    
    # 【优化2】：包含匹配（长度比例 >= 80% 才通过，否则继续后续检查而非短路返回 False）
    if cleaned_text1 in cleaned_text2:
        if len(cleaned_text1) >= len(cleaned_text2) * 0.8:
            return True
        # 长度差距大但不直接 return False，让后续检查（相似度、后缀剥离）继续
    elif cleaned_text2 in cleaned_text1:
        if len(cleaned_text2) >= len(cleaned_text1) * 0.8:
            return True
    
    # 【优化3】：使用 SequenceMatcher 计算相似度（阈值已提高到默认0.95）
    similarity = SequenceMatcher(None, cleaned_text1, cleaned_text2).ratio()
    
    if similarity >= threshold:
        return True
    
    # 【优化4】：后缀剥离匹配（支持双边和单边后缀剥离）
    suffixes = ['合计', '总计', '小计', '合计金额', '总计金额', '项目', '科目', '总额', '净额', '净值']
    relaxed_threshold = max(0.90, threshold - 0.05)
    
    for suffix in suffixes:
        both_have = cleaned_text1.endswith(suffix) and cleaned_text2.endswith(suffix)
        only_t1 = cleaned_text1.endswith(suffix) and not cleaned_text2.endswith(suffix)
        only_t2 = cleaned_text2.endswith(suffix) and not cleaned_text1.endswith(suffix)
        
        if both_have:
            # 双边都有后缀：剥离后比较
            base1 = cleaned_text1[:-len(suffix)]
            base2 = cleaned_text2[:-len(suffix)]
            if base1 and base2:
                if base1 == base2 or SequenceMatcher(None, base1, base2).ratio() >= relaxed_threshold:
                    return True
        elif only_t1:
            # 只有 text1 有后缀：剥离 text1 的后缀，与 text2 比较
            base1 = cleaned_text1[:-len(suffix)]
            if base1 and (base1 == cleaned_text2 or SequenceMatcher(None, base1, cleaned_text2).ratio() >= relaxed_threshold):
                return True
        elif only_t2:
            # 只有 text2 有后缀：剥离 text2 的后缀，与 text1 比较
            base2 = cleaned_text2[:-len(suffix)]
            if base2 and (cleaned_text1 == base2 or SequenceMatcher(None, cleaned_text1, base2).ratio() >= relaxed_threshold):
                return True
    
    return False


def _strip_report_qualifier_prefixes(name: str) -> str:
    """
    剥离报表名称中的限定性前缀，返回核心报表名称。

    例如：
      "新会计准则资产负债表"  → "资产负债表"
      "旧会计准则合并利润表"  → "合并利润表"
      "企业会计准则现金流量表" → "现金流量表"
      "合并资产负债表"         → "合并资产负债表"（无限定前缀，原样返回）
    """
    QUALIFIER_PREFIXES = [
        '新会计准则', '旧会计准则', '企业会计准则',
        '新准则', '旧准则', '会计准则',
    ]
    for prefix in sorted(QUALIFIER_PREFIXES, key=len, reverse=True):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


_CALIBER_KW_RE = re.compile(r'合并|单体|母公司')

_REPORT_ALIASES: dict[str, list[str]] = {
    "资产负债表": ["资产负债", "财务状况表"],
    "利润表": ["损益表", "收益表"],
    "现金流量表": ["现金流量", "资金流量表"],
}


def _extract_report_core(name: str) -> str:
    """剥离限定前缀和口径关键词，提取核心报表类型名。

    例：
      "新会计准则合并资产负债表" → "资产负债表"
      "合并利润表"               → "利润表"
      "资产负债表"               → "资产负债表"
    """
    n = _strip_report_qualifier_prefixes(name)
    n = _CALIBER_KW_RE.sub('', n)
    return n


def _caliber_compatible(table_name: str, report_caliber: int) -> bool:
    """检查表名中的口径关键词是否与 report_caliber 冲突。

    策略：
    - 如果表名显式包含口径关键词，必须与 report_caliber 一致。
    - 如果表名不含任何口径关键词，视为兼容（Java 可能未在表名中标注口径）。
    """
    has_merged = "合并" in table_name
    has_parent = "单体" in table_name or "母公司" in table_name
    if not has_merged and not has_parent:
        return True
    if report_caliber == 2:
        return has_merged and not has_parent
    else:
        return has_parent or not has_merged


def match_report_name(report_name: str, report_caliber: int, table_report_name: str) -> bool:
    """
    匹配报表名称（考虑报表口径）。

    核心思路：
    1. 剥离双方的限定前缀和口径关键词，比较核心报表类型名。
    2. 口径单独校验：表名中显式出现的口径关键词不能与 report_caliber 矛盾；
       若表名无口径关键词，视为兼容。
    3. 支持别名（损益表↔利润表 等）。

    Args:
        report_name: 报表名称（来自 reportList）
        report_caliber: 报表口径（1=单体，2=合并）
        table_report_name: 表格中的报表名称

    Returns:
        bool: 是否匹配成功
    """
    if not report_name or not table_report_name:
        return False

    def _clean(name: str) -> str:
        name = re.sub(r"[\s，。、；！？\u201c\u201d\u2018\u2019（）()\[\]【】]", '', name)
        return traditional_to_simplified(name)

    clean_r = _clean(report_name)
    clean_t = _clean(table_report_name)

    # 快速完全匹配
    if clean_r == clean_t:
        return True

    # 提取核心报表类型名（去掉"合并/单体/母公司"和限定前缀）
    core_r = _extract_report_core(clean_r)
    core_t = _extract_report_core(clean_t)

    if not core_r or not core_t:
        return False

    # 口径冲突检测
    if not _caliber_compatible(clean_t, report_caliber):
        logger.debug("口径冲突: report={} caliber={}, table={}", report_name, report_caliber, table_report_name)
        return False

    # 核心名称精确匹配
    if core_r == core_t:
        return True

    # 核心名称语义匹配（处理细微差异，如 OCR 漏字）
    if semantic_match(core_r, core_t, threshold=0.85):
        return True

    # 核心名称包含匹配（如 core_t="资产负债" 包含在 core_r="资产负债表" 中）
    if core_r in core_t and len(core_r) >= len(core_t) * 0.6:
        return True
    if core_t in core_r and len(core_t) >= len(core_r) * 0.6:
        return True

    # 别名匹配
    all_aliases: list[str] = []
    if core_r in _REPORT_ALIASES:
        all_aliases.extend(_REPORT_ALIASES[core_r])
    for canonical, aliases in _REPORT_ALIASES.items():
        if core_r in aliases:
            all_aliases.append(canonical)
            all_aliases.extend(a for a in aliases if a != core_r)
            break

    for alias in all_aliases:
        if alias == core_t:
            return True
        if semantic_match(alias, core_t, threshold=0.85):
            return True

    return False
