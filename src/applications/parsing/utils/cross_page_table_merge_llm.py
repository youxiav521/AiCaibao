from openai import OpenAI
from dotenv import load_dotenv 
import json
from pydantic import BaseModel
from openai import BadRequestError
import os
import csv
from typing import Dict, Any, List, Tuple, Optional


from copy import deepcopy

client = OpenAI(base_url="http://103.154.31.78:20001/compatible-mode/v1",
       api_key="IfWZqv1F55hsRWlWB2fqCca91VK9IMET")


def calling_merge(table_json_str1 : str,table_json_str2 : str):
    """
    正常跨页表格合并：将表格2按行拼接到表格1之后，并尽量移除表格2的表头行。

    输入为两个表格对象的 JSON 字符串；返回包含更新后的两个表格对象的字典。
    """
    try:
        A: Dict[str, Any] = json.loads(table_json_str1) if table_json_str1 else {}
    except Exception:
        A = {}
    try:
        B: Dict[str, Any] = json.loads(table_json_str2) if table_json_str2 else {}
    except Exception:
        B = {}

    def _get_num_rows(table_obj: Dict[str, Any]) -> int:
        try:
            data = table_obj.get("data") or {}
            num_rows = data.get("num_rows")
            return int(num_rows) if num_rows is not None else 0
        except Exception:
            return 0

    def _set_num_rows(table_obj: Dict[str, Any], value: int) -> None:
        data = table_obj.setdefault("data", {})
        data["num_rows"] = int(value)

    def _remove_header_rows_from_table_cells(table_obj: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
        data = table_obj.get("data") or {}
        cells: List[Dict[str, Any]] = list(data.get("table_cells") or [])
        header_rows = set()
        for c in cells:
            if c.get("column_header") is True:
                try:
                    header_rows.add(int(c.get("start_row_offset_idx") or 0))
                except Exception:
                    continue
        if not header_rows:
            return cells, 0
        filtered = [c for c in cells if int(c.get("start_row_offset_idx") or 0) not in header_rows]
        return filtered, len(header_rows)

    # 优先使用 table_cells 进行结构化合并；若不存在再回退到 grid 级别拼接
    data_A = A.setdefault("data", {})
    data_B = B.setdefault("data", {})

    if data_A.get("table_cells") or data_B.get("table_cells"):
        cells_A: List[Dict[str, Any]] = list(data_A.get("table_cells") or [])
        filtered_B, removed_header_cnt = _remove_header_rows_from_table_cells(B)
        a_rows = _get_num_rows(A)
        # 位移 B 的行索引后追加
        for c in filtered_B:
            try:
                c = deepcopy(c)
                c["start_row_offset_idx"] = int(c.get("start_row_offset_idx") or 0) + a_rows
                c["end_row_offset_idx"] = int(c.get("end_row_offset_idx") or 0) + a_rows
                cells_A.append(c)
            except Exception:
                continue
        data_A["table_cells"] = cells_A
        # 更新行数
        b_rows_total = _get_num_rows(B)
        added_rows = max(0, b_rows_total - removed_header_cnt)
        _set_num_rows(A, a_rows + added_rows)

        # 同步合并 Docling grid（若存在）
        grid_A: List[List[Dict[str, Any]]] = list(data_A.get("grid") or [])
        grid_B: List[List[Dict[str, Any]]] = list(data_B.get("grid") or [])

        if grid_B:
            if not grid_A:
                # 若 A 无 grid，则直接采用 B 的 grid（无法判断并去重表头）
                data_A["grid"] = grid_B
            else:
                def _row_texts(row: List[Dict[str, Any]]) -> List[str]:
                    return [str((cell or {}).get("text") or "") for cell in row]

                drop_first = False
                b0 = _row_texts(grid_B[0]) if len(grid_B) > 0 else []
                a0 = _row_texts(grid_A[0]) if len(grid_A) > 0 else []
                al = _row_texts(grid_A[-1]) if len(grid_A) > 0 else []
                if b0 == a0 or b0 == al:
                    drop_first = True
                if drop_first:
                    grid_A.extend(grid_B[1:])
                else:
                    grid_A.extend(grid_B)
                data_A["grid"] = grid_A
    else:
        # 使用 grid 合并（尽可能去重表头）
        grid_A: List[List[Dict[str, Any]]] = list(data_A.get("grid") or [])
        grid_B: List[List[Dict[str, Any]]] = list(data_B.get("grid") or [])

        def _row_texts(row: List[Dict[str, Any]]) -> List[str]:
            return [str((cell or {}).get("text") or "") for cell in row]

        if grid_A and grid_B:
            # 简单判断首行是否为重复表头：与 A 的首行或最后一个表头行等值时去掉 B 首行
            drop_first = False
            if len(grid_B) > 0:
                b0 = _row_texts(grid_B[0])
                if grid_A:
                    a0 = _row_texts(grid_A[0])
                    al = _row_texts(grid_A[-1])
                    if b0 == a0 or b0 == al:
                        drop_first = True
            if drop_first:
                grid_A.extend(grid_B[1:])
            else:
                grid_A.extend(grid_B)
            data_A["grid"] = grid_A
        else:
            # 无法合并，保留原样
            pass

    # 标记 B 被合并
    B["label"] = "table_fragment_merged"
    data_B = B.setdefault("data", {})
    if data_B.get("table_cells") is not None:
        data_B["table_cells"] = []
    if data_B.get("grid") is not None:
        data_B["grid"] = []
    _set_num_rows(B, 0)
    data_B["merged_into"] = A.get("self_ref") or ""

    return {
        "action": "calling_merge",
        "merged_A": A,
        "merged_B": B,
    }

def calling_merge_cell(table_json_str1 : str,table_json_str2 : str):
    """
    行内切分合并：将表格2的第一行与表格1的最后一行按列拼接，随后把表格2剩余行附加到表格1。
    优先按 table_cells 结构进行合并；若不可用，则回退到 grid 层级。
    """
    try:
        A: Dict[str, Any] = json.loads(table_json_str1) if table_json_str1 else {}
    except Exception:
        A = {}
    try:
        B: Dict[str, Any] = json.loads(table_json_str2) if table_json_str2 else {}
    except Exception:
        B = {}

    data_A = A.setdefault("data", {})
    data_B = B.setdefault("data", {})

    def _get_num_rows(table_obj: Dict[str, Any]) -> int:
        try:
            data = table_obj.get("data") or {}
            num_rows = data.get("num_rows")
            return int(num_rows) if num_rows is not None else 0
        except Exception:
            return 0

    def _set_num_rows(table_obj: Dict[str, Any], value: int) -> None:
        data = table_obj.setdefault("data", {})
        data["num_rows"] = int(value)

    def _remove_header_rows_from_table_cells(table_obj: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
        data = table_obj.get("data") or {}
        cells: List[Dict[str, Any]] = list(data.get("table_cells") or [])
        header_rows = set()
        for c in cells:
            if c.get("column_header") is True:
                try:
                    header_rows.add(int(c.get("start_row_offset_idx") or 0))
                except Exception:
                    continue
        if not header_rows:
            return cells, 0
        filtered = [c for c in cells if int(c.get("start_row_offset_idx") or 0) not in header_rows]
        return filtered, len(header_rows)

    if data_A.get("table_cells") or data_B.get("table_cells"):
        cells_A: List[Dict[str, Any]] = list(data_A.get("table_cells") or [])
        cells_B_all: List[Dict[str, Any]] = list(data_B.get("table_cells") or [])

        a_rows = _get_num_rows(A)
        last_row_idx = max(0, a_rows - 1)

        # 先移除 B 的表头行
        filtered_B, _ = _remove_header_rows_from_table_cells(B)

        # 拆分 B 的第一行与后续行
        b_first_row_cells: List[Dict[str, Any]] = [c for c in filtered_B if int(c.get("start_row_offset_idx") or 0) == 0]
        b_rest_cells: List[Dict[str, Any]] = [c for c in filtered_B if int(c.get("start_row_offset_idx") or 0) > 0]

        # 将 B 第一行逐列合并到 A 的最后一行（按列跨度重叠判定）
        a_last_row_cells_idx: List[int] = [i for i, c in enumerate(cells_A) if int(c.get("start_row_offset_idx") or 0) == last_row_idx]
        for b_cell in b_first_row_cells:
            b_start_col = int(b_cell.get("start_col_offset_idx") or 0)
            b_end_col = int(b_cell.get("end_col_offset_idx") or b_start_col)
            b_text = str(b_cell.get("text") or "")

            # 找到与之列区间重叠的 A 最后一行单元格
            overlapped_a_indices: List[int] = []
            for i in a_last_row_cells_idx:
                a_cell = cells_A[i]
                a_start_col = int(a_cell.get("start_col_offset_idx") or 0)
                a_end_col = int(a_cell.get("end_col_offset_idx") or a_start_col)
                if not (b_end_col <= a_start_col or b_start_col >= a_end_col):
                    overlapped_a_indices.append(i)

            if overlapped_a_indices:
                # 简单地将文本拼接到第一个重叠单元格
                a_idx = overlapped_a_indices[0]
                a_cell = cells_A[a_idx]
                a_cell_text = str(a_cell.get("text") or "")
                joiner = " " if (a_cell_text and b_text) else ""
                a_cell["text"] = a_cell_text + joiner + b_text
            else:
                # 若没有重叠单元格，则在 A 最后一行创建一个新单元格
                new_cell = {
                    "start_row_offset_idx": last_row_idx,
                    "end_row_offset_idx": last_row_idx,
                    "start_col_offset_idx": b_start_col,
                    "end_col_offset_idx": b_end_col,
                    "text": b_text,
                }
                cells_A.append(new_cell)

        # 将 B 剩余行向后偏移并追加到 A
        for c in b_rest_cells:
            try:
                c = deepcopy(c)
                c["start_row_offset_idx"] = int(c.get("start_row_offset_idx") or 0) - 1 + a_rows
                c["end_row_offset_idx"] = int(c.get("end_row_offset_idx") or 0) - 1 + a_rows
                cells_A.append(c)
            except Exception:
                continue

        data_A["table_cells"] = cells_A
        # 更新 A 的行数：A 原有行 + (B 总行 - 1)
        b_rows_total = _get_num_rows(B)
        _set_num_rows(A, last_row_idx + 1 + max(0, b_rows_total - 1))

        # 同步合并 Docling grid（若存在）：合并 A 最后一行与 B 第一行，并追加 B 其余行
        grid_A: List[List[Dict[str, Any]]] = list(data_A.get("grid") or [])
        grid_B: List[List[Dict[str, Any]]] = list(data_B.get("grid") or [])
        if grid_A and grid_B:
            la = grid_A[-1]
            fb = grid_B[0]
            max_len = max(len(la), len(fb))
            merged_row: List[Dict[str, Any]] = []
            for i in range(max_len):
                a_cell = la[i] if i < len(la) else {"text": ""}
                b_cell = fb[i] if i < len(fb) else {"text": ""}
                a_text = str(a_cell.get("text") or "")
                b_text = str(b_cell.get("text") or "")
                joiner = " " if (a_text and b_text) else ""
                merged_cell = deepcopy(a_cell)
                merged_cell["text"] = a_text + joiner + b_text
                merged_row.append(merged_cell)
            grid_A[-1] = merged_row
            grid_A.extend(grid_B[1:])
            data_A["grid"] = grid_A
    else:
        # 回退到 grid 逻辑
        grid_A: List[List[Dict[str, Any]]] = list(data_A.get("grid") or [])
        grid_B: List[List[Dict[str, Any]]] = list(data_B.get("grid") or [])
        if grid_A and grid_B:
            # 合并最后一行与第一行
            if grid_A and grid_B:
                la = grid_A[-1]
                fb = grid_B[0]
                max_len = max(len(la), len(fb))
                merged_row: List[Dict[str, Any]] = []
                for i in range(max_len):
                    a_cell = la[i] if i < len(la) else {"text": ""}
                    b_cell = fb[i] if i < len(fb) else {"text": ""}
                    a_text = str(a_cell.get("text") or "")
                    b_text = str(b_cell.get("text") or "")
                    joiner = " " if (a_text and b_text) else ""
                    merged_cell = deepcopy(a_cell)
                    merged_cell["text"] = a_text + joiner + b_text
                    merged_row.append(merged_cell)
                grid_A[-1] = merged_row
            # 追加 B 的其余行
            grid_A.extend(grid_B[1:])
            data_A["grid"] = grid_A

    # 标记 B 被合并
    B["label"] = "table_fragment_merged"
    data_B = B.setdefault("data", {})
    if data_B.get("table_cells") is not None:
        data_B["table_cells"] = []
    if data_B.get("grid") is not None:
        data_B["grid"] = []
    _set_num_rows(B, 0)
    data_B["merged_into"] = A.get("self_ref") or ""

    return {
        "action": "calling_merge_cell",
        "merged_A": A,
        "merged_B": B,
    }

tools = [
    {
        "type": "function",
        "function": {
            "name": "calling_merge",
            "description": "当表格在两个完整的、独立的行之间被跨页切分时调用。适用于第一个表格的最后一行和第二个表格的第一行都是语义完整的、独立的项目的情况。",
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calling_merge_cell",
            "description": "当一个表格行被页面截断，导致其内容分散在两个表格中时调用。适用于第一个表格的最后一行是句子的前半部分，而第二个表格的第一行是其后半部分的情况。",
        }
    }
]

def merge_agent(table_json_str1,table_json_str2, pred_csv_basename: Optional[str] = None):
    try:
        table_json_1 = json.loads(table_json_str1)
    except Exception:
        table_json_1 = {}
    try:
        table_json_2 = json.loads(table_json_str2)
    except Exception:
        table_json_2 = {}

    def extract_grid(table_json_obj):
        if isinstance(table_json_obj, dict):
            if "grid" in table_json_obj:
                return table_json_obj.get("grid")
            data_obj = table_json_obj.get("data", {})
            return data_obj.get("grid")
        return None

    grid_1 = extract_grid(table_json_1) or []
    grid_2 = extract_grid(table_json_2) or []

    last_row_table_1 = grid_1[-1] if isinstance(grid_1, list) and len(grid_1) > 0 else None
    first_row_table_2 = grid_2[0] if isinstance(grid_2, list) and len(grid_2) > 0 else None

    json_data = {
        "表格1最后一行": last_row_table_1,
        "表格2第一行": first_row_table_2
    }
    
    prompt = f"""

# Role: 财务表格结构解析专家

## Profile
- language: 中文
- description: 专注于财务表格跨页切分类型识别与结构化处理，依据上下页行数据的语义关联性、格式连续性及业务规则，精准判断并调用对应工具完成表格拼接或分割操作。
- background: 财务报表处理系统需要自动化识别表格跨页逻辑，确保数据完整性与分析准确性。
- personality: 严谨细致，遵循财务术语规范，注重格式逻辑与业务场景适配性
- expertise: 财务表格结构分析、跨页逻辑识别、文本语义拼接、财务数据标准化处理
- target_audience: 财务报表解析系统、企业数据处理平台、财务智能分析工具

## Skills

1. 跨页切分识别
   - 行间语义匹配: 通过上下文关联判断行与行之间的逻辑关系
   - 格式连续性分析: 检测表格行的格式是否保持连贯
   - 术语完整性评估: 识别财务术语是否因分页而断裂
   - 标点闭合检测: 识别未闭合的括号/书名号等标点符号

2. 文本结构化处理
   - 单元格对齐技术: 实现跨页行的列对齐
   - 拼接算法优化: 根据业务规则进行文本拼接
   - 异常数据校验: 识别格式异常或逻辑错误
   - 工具调用管理: 合理选择并执行对应处理工具

## Rules

1. 基本原则：
   - 严格遵循决策框架执行判断流程
   - 优先保障财务术语完整性
   - 保持表格格式结构的连贯性
   - 保证数据逻辑的合理性

2. 行为准则：
   - 只处理输入的两行数据
   - 不进行超出提供的信息范围的推测
   - 必须遵循指定的工具调用规范
   - 禁止组合不相关数据

3. 限制条件：
   - 仅处理两行数据的对比分析
   - 不使用外部知识或上下文信息
   - 不允许进行人工干预判断
   - 严格限制工具调用为指定2种

## Workflows

- 目标: 精准判断表格跨页切分类型并调用对应处理工具
- 步骤 1: 执行列对齐操作，按位置优先、索引回退、语义校正规则处理
- 步骤 2: 依据对齐结果进行语义分析，检测H2强信号
- 步骤 3: 不存在H2强信号时默认选择H1处理
- 预期结果: 正确调用calling_merge或calling_merge_cell工具

# 输入数据
- `last_row_table_1`: {last_row_table_1}
- `first_row_table_2`: {first_row_table_2}

## Initialization
作为财务表格结构解析专家，你必须遵守上述Rules，按照Workflows执行任务。"""

    response = client.chat.completions.create(
        model="qwen3-32b",
        messages=[{"role": "user", "content": prompt}],
        tools=tools,
        extra_body={"enable_thinking": False},
        tool_choice="required",
        temperature=0.01,
    )

    # 日志落盘：将合并数据展示与模型选择的函数名增量写入 logs/cross_page_table_merge.jsonl
    try:
        response_json = json.loads(response.model_dump_json())
        function_name = ""
        if response_json.get('choices') and response_json['choices'][0]['message'].get('tool_calls'):
            tool_call = response_json['choices'][0]["message"]["tool_calls"][0]
            function_name = tool_call.get("function", {}).get("name", "")
        log_record = {"合并数据展示": json_data, "调用函数": function_name}
        os.makedirs("logs", exist_ok=True)
        log_path = os.path.join("logs", "cross_page_table_merge.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_record, ensure_ascii=False) + "\n")
    except Exception:
        pass

    # 追加预测到 CSV：logs/<pred_csv_basename>_with_pred.csv（不污染 GT 文件）
    try:
        def _safe_get_page_no(obj: Dict[str, Any]) -> Optional[int]:
            try:
                prov_list = obj.get("prov") or []
                if isinstance(prov_list, list) and prov_list:
                    first = prov_list[0]
                    if isinstance(first, dict):
                        page_no = first.get("page_no")
                        return int(page_no) if page_no is not None else None
            except Exception:
                return None
            return None

        pred_type = None
        if 'function_name' in locals():
            if function_name == "calling_merge":
                pred_type = "normal"
            elif function_name == "calling_merge_cell":
                pred_type = "row_split"

        start_page = _safe_get_page_no(table_json_1)
        end_page = _safe_get_page_no(table_json_2)

        if pred_type and start_page is not None and end_page is not None:
            csv_dir = "logs"
            os.makedirs(csv_dir, exist_ok=True)
            base_name = (pred_csv_basename or "合并测试数据集").strip()
            csv_path = os.path.join(csv_dir, f"{base_name}_with_pred.csv")
            file_exists = os.path.exists(csv_path)
            with open(csv_path, "a", encoding="utf-8", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["start_page", "end_page", "cross_page_type"])
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    "start_page": int(start_page),
                    "end_page": int(end_page),
                    "cross_page_type": pred_type,
                })
    except Exception:
        pass

    return response


def do_merge(response,table_json_str1,table_json_str2):
    response_json = json.loads(response.model_dump_json())
    if response_json['choices'] and response_json['choices'][0]['message'].get('tool_calls'):
        tool_call = response_json['choices'][0]["message"]["tool_calls"][0]
        function_name = tool_call["function"]["name"]
        
        if function_name == "calling_merge":
            return calling_merge(table_json_str1,table_json_str2)
        elif function_name == "calling_merge_cell":
            return calling_merge_cell(table_json_str1,table_json_str2)
        else:
            print(f"Unknown function: {function_name}")
    else:
        print("No tool calls found in the response.")
    return None