import os
import pandas as pd

# 默认表格存放基础目录 - 相对于 code_executor 工作目录
DEFAULT_BASE_DIR = "./tmp_tables"

def _ensure_base_dir(base_dir: str = DEFAULT_BASE_DIR):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

def _get_table_path(table_name: str, ext: str = "json") -> str:
    """
    获取表格的完整路径，默认扩展名为json。
    :param table_name: 表格名称（不带扩展名）
    :param ext: 文件扩展名，默认为json
    :return: 完整路径
    """
    _ensure_base_dir(DEFAULT_BASE_DIR)
    return os.path.join(DEFAULT_BASE_DIR, f"{table_name}.{ext}")

def read_csv(table_name: str, **kwargs) -> pd.DataFrame:
    """
    读取指定名称的csv表格为DataFrame。
    :param table_name: 表格名称（不带扩展名）
    :param kwargs: 传递给pd.read_csv的其他参数
    :return: DataFrame
    """
    path = _get_table_path(table_name, "csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到表格文件: {path}")
    return pd.read_csv(path, **kwargs)

def read_excel(table_name: str, sheet_name=0, **kwargs) -> pd.DataFrame:
    """
    读取指定名称的excel表格为DataFrame。
    :param table_name: 表格名称（不带扩展名）
    :param sheet_name: 工作表名称或索引，默认为第一个
    :param kwargs: 传递给pd.read_excel的其他参数
    :return: DataFrame
    """
    path = _get_table_path(table_name, "xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到表格文件: {path}")
    return pd.read_excel(path, sheet_name=sheet_name, **kwargs)

def read_json(table_name: str, orient: str = "split", **kwargs) -> pd.DataFrame:
    """
    读取指定名称的json表格为DataFrame。
    :param table_name: 表格名称（不带扩展名）
    :param orient: 指定json的结构，默认为'split'
    :param kwargs: 传递给pd.read_json的其他参数
    :return: DataFrame
    """
    path = _get_table_path(table_name, "json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到表格文件: {path}")
    return pd.read_json(path, orient=orient, **kwargs)

def read_table(table_name: str, ext: str = "json", **kwargs) -> pd.DataFrame:
    """
    通用读取方法，支持json(csv/xlsx/xls)等格式, 默认json。
    :param table_name: 表格名称（不带扩展名）
    :param ext: 文件扩展名，默认为json
    :param kwargs: 传递给pandas读取函数的其他参数
    :return: DataFrame
    """
    path = _get_table_path(table_name, ext)
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到表格文件: {path}")
    ext_lower = ext.lower()
    # TODO 读取也应该暂时用split
    if ext_lower == "json":
        orient = kwargs.pop("orient", "split")
        return pd.read_json(path, orient=orient, **kwargs)
    elif ext_lower == "csv":
        return pd.read_csv(path, **kwargs)
    elif ext_lower in ["xlsx", "xls"]:
        return pd.read_excel(path, **kwargs)
    else:
        raise ValueError(f"暂不支持的表格格式: {ext}")

def save_dataframe(df: pd.DataFrame, table_name: str, ext: str = "json", **kwargs):
    """
    保存DataFrame到指定名称的表格文件，默认json，可选csv/xlsx。
    :param df: 要保存的DataFrame
    :param table_name: 表格名称（不带扩展名）
    :param ext: 文件扩展名（json/csv/xlsx）
    :param kwargs: 传递给pandas保存函数的其他参数
    """
    path = _get_table_path(table_name, ext)
    ext_lower = ext.lower()
    # 检查路径是否已存在，如果存在则自动在文件名末尾添加(x)
    orig_path = path
    if os.path.exists(orig_path):
        base, extname = os.path.splitext(orig_path)
        i = 1
        path = f"{base}({i}){extname}"
        while os.path.exists(path):
            i += 1
            path = f"{base}({i}){extname}"

    if ext_lower == "json":
        # TODO 这里暂时按split来保存表格格式，需要思考那种格式的保存最合适且适合提示给大模型和进行后续转换
        orient = kwargs.pop("orient", "split")
        df.to_json(path, orient=orient, force_ascii=False, **kwargs)
    elif ext_lower == "csv":
        df.to_csv(path, index=False, **kwargs)
    elif ext_lower in ["xlsx", "xls"]:
        df.to_excel(path, index=False, **kwargs)
    else:
        raise ValueError(f"暂不支持的表格格式: {ext}")
    return os.path.basename(path)

def save_table(df: pd.DataFrame, table_name: str, ext: str = "json", **kwargs):
    try: 
        if not isinstance(df, pd.DataFrame):
            err_msg = f"save_table:参数df类型错误，应为pd.DataFrame，实际为{type(df)}"
            # print(err_msg)
            return err_msg
        table_file_name = save_dataframe(df, table_name, ext, **kwargs)
        df_json = df.to_json(orient="split", force_ascii=False) if not df.empty else None
        print(f"save_table:保存表格 {table_file_name} 成功")
        print(f"内容示例：{df_json}")
        return f"save_table:保存表格 {table_file_name} 成功"
    except Exception as e:
        print(f"表格保存错误{e}")
        return f"save_table:保存表格{table_name}失败: {e}"

def list_tables(ext: str = None):
    """
    列出基础目录下所有表格文件。
    :param ext: 只列出指定扩展名的文件（如'json'/'csv'/'xlsx'），不指定则全部列出
    :return: 文件名列表
    """
    _ensure_base_dir(DEFAULT_BASE_DIR)
    files = os.listdir(DEFAULT_BASE_DIR)
    if ext:
        ext = ext.lower()
        return [f for f in files if f.lower().endswith(f".{ext}")]
    return files
