"""
表格意图识别模块
独立实现表格意图识别功能
"""
import re
import pandas as pd
from loguru import logger

def is_empty(cell):
    """判断单元格是否为空"""
    if pd.isna(cell):
        return True
    if isinstance(cell, str):
        cell = cell.strip()
        return cell == '' or cell == '/'
    return False

def is_serial_number(text):
    """判断是否为序号（如 1, 2, 3, ①, 一, 等）"""
    if is_empty(text):
        return False
    text = str(text).strip()
    # 纯数字序号
    if text.isdigit():
        return True
    # 常见序号模式
    sn_patterns = [
        r'^\d+[.、]',      # 1.  1、  1,
        r'^[①②③④⑤⑥⑦⑧⑨⑩]$',  # ① ② ③
        r'^[一二三四五六七八九十]+[、.]',  # 一、  二.
        r'^[abcABC][.、]',  # a.  A.
    ]
    return any(re.match(p, text) for p in sn_patterns)

class TableIntentRecognition:
    """
    表格意图识别
    """

    def fill_in_info(self, df: pd.DataFrame):
        """
        填充表结构识别：填充位置，非填充行，重复行
        """
        logger.debug(f'TableIntentRecognition: {df.shape}')

        self.fill_in_points = []
        self.duplicated_rows = []
        self.checkboxs_point = []
        self.empty_points = []
        self.fill_in_rows=[]
        self.fill_in_ratio = 0

        for idx_r in range(len(df)):
            row = df.iloc[idx_r]
            last_v = ''
            vc = row.value_counts().to_dict()

            # 如果整行只有一个值，标记为重复行
            if len(vc) == 1:
                first_val = row.iloc[0] if len(row) > 0 else ''
                self.duplicated_rows.append((idx_r, first_val))
                continue

            # 遍历每一行的每列，也就是逐单元格遍历
            for idx_c in range(len(df.columns)):
                cell = row.iloc[idx_c]
                # 统计所有空白单元格
                if is_empty(cell):
                    self.empty_points.append([idx_r, idx_c])
                if is_empty(cell) and not is_empty(last_v):
                    # 空白单元格，且前相邻格不为空白，则需要填充
                    self.fill_in_points.append([idx_r, idx_c, last_v])
                else:
                    # 检查复选框
                    if isinstance(cell, str):
                        cbs = re.findall(r'□([^□：:]+(?:[：:][^□]*)?)', cell)
                        if len(cbs) > 0 and cell != last_v:
                            self.fill_in_points.append([idx_r, idx_c, cell])
                            self.checkboxs_point.append([idx_r, idx_c, cbs])

                last_v = cell

        # 计算填充比例
        if len(self.fill_in_points) > 0:
            self.fill_in_rows = list(set([item[0] for item in self.fill_in_points]))
            total_rows = df.shape[0]
            if total_rows > 0:
                self.fill_in_ratio = round(
                    (len(self.fill_in_rows) + len(self.duplicated_rows)) / total_rows, 2
                )
            logger.debug(f'[table-fill-in] ratio: {self.fill_in_ratio}')

    def analysis_table_intent(self, df: pd.DataFrame):
        """
        表格意图识别

        Returns:
            tuple: (意图类型, 信息字典)
                  - DYNAMIC_TABLE: 动态信息表格
                  - DEFAULT: 默认类型
                  - STATIC_TABLE: 静态信息表格
        """
        self.fill_in_info(df)
        
        total_rows = df.shape[0]
        # 判断是否为填充表：填充比例 > 0.5 且行数 > 3
        if len(self.empty_points) <= len(self.fill_in_points) or len(self.fill_in_points)>=total_rows:
            return ("STATIC_TABLE", {
                "empty_points": self.empty_points,
                "fill_points": self.fill_in_points,
                "fill_rows": self.fill_in_rows,
                "1value_rows": self.duplicated_rows
            })
        elif self.fill_in_ratio > 0.5 and df.shape[0] > 3:
            result = ("DYNAMIC_TABLE", {
                "empty_points": self.empty_points,
                "fill_points": self.fill_in_points,
                "fill_rows": self.fill_in_rows,
                "1value_rows": self.duplicated_rows
            })
        else:
            result = ("DEFAULT", {
                "empty_points": self.empty_points,
                "fill_points": self.fill_in_points,
                "fill_rows": self.fill_in_rows,
                "1value_rows": self.duplicated_rows
            })

        return result
