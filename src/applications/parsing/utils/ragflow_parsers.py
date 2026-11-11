# -*- coding: utf-8 -*-
import re
from typing import List, Dict, Any, Tuple
from loguru import logger

from markdown import markdown
from src.utils import json_utils
from src.utils.tokenizer import Tokenizer

_tokenizer = Tokenizer()


def num_tokens_from_string(string: str) -> int:
    """计算字符串中的 token 数量（cl100k_base 编码）。"""
    return _tokenizer(string)


def get_text(fnm: str, binary=None) -> str:
    """从文件或二进制内容读取文本"""
    txt = ""
    if binary:
        try:
            txt = binary.decode("utf-8")
        except UnicodeDecodeError:
            try:
                txt = binary.decode("gb18030", errors="ignore")
            except Exception:
                txt = str(binary)
    else:
        with open(fnm, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
    return txt

# ================= 来自 deepdoc/parser/markdown_parser.py =================
class RAGFlowMarkdownParser:
    def __init__(self, chunk_token_num=128):
        self.chunk_token_num = int(chunk_token_num)

    def extract_tables_and_remainder(self, markdown_text, separate_tables=True):
        """提取表格和剩余文本内容"""
        tables = []
        working_text = markdown_text

        def replace_tables_with_rendered_html(pattern, table_list, render=True):
            """用占位符或渲染后的 HTML 替换表格"""
            new_text = ""
            last_end = 0
            for match in pattern.finditer(working_text):
                raw_table = match.group()
                
                # 计算起止行号
                start_line = working_text.count('\n', 0, match.start())
                cnt = raw_table.count('\n')
                end_line = start_line + cnt - 1
                
                if raw_table.startswith('\n'):
                    start_line += 1
                
                table_list.append({
                    "content": raw_table,
                    "start_line": start_line,
                    "end_line": end_line
                })
                
                if separate_tables:
                    # 用占位符替换表格，保留换行数量
                    if raw_table.startswith('\n'):
                        repl = "\n# RAGFLOW_TABLE_PLACEHOLDER" + ("\n" * (cnt - 1))
                    else:
                        repl = "# RAGFLOW_TABLE_PLACEHOLDER" + ("\n" * cnt)
                        
                    new_text += working_text[last_end : match.start()] + repl
                else:
                    # 替换为渲染后的 HTML
                    if render and markdown:
                        html_table = markdown(raw_table, extensions=["markdown.extensions.tables"]) 
                    else:
                        html_table = raw_table
                    new_text += working_text[last_end : match.start()] + html_table + "\n\n"
                last_end = match.end()
            new_text += working_text[last_end:]
            return new_text

        if "|" in markdown_text:  # 性能优化
            # 标准 Markdown 表格
            border_table_pattern = re.compile(
                r"""
                (?:\n|^)
                (?:\|.*?\|.*?\|.*?\n)
                (?:\|(?:\s*[:-]+[-| :]*\s*)\|.*?\n)
                (?:\|.*?\|.*?\|.*?\n)+
            """,
                re.VERBOSE,
            )
            working_text = replace_tables_with_rendered_html(border_table_pattern, tables)

            # 无边框 Markdown 表格
            no_border_table_pattern = re.compile(
                r"""
                (?:\n|^)
                (?:\S.*?\|.*?\n)
                (?:(?:\s*[:-]+[-| :]*\s*).*?\n)
                (?:\S.*?\|.*?\n)+
                """,
                re.VERBOSE,
            )
            working_text = replace_tables_with_rendered_html(no_border_table_pattern, tables)

        # 替换带属性的 HTML 标签为简单标签
        TAGS = ["table", "td", "tr", "th", "tbody", "thead", "div"]
        table_with_attributes_pattern = re.compile(rf"<(?:{'|'.join(TAGS)})[^>]*>", re.IGNORECASE)

        def replace_tag(m):
            tag_name = re.match(r"<(\w+)", m.group()).group(1)
            return "<{}>".format(tag_name)

        working_text = re.sub(table_with_attributes_pattern, replace_tag, working_text)

        if "<table>" in working_text.lower():  # 性能优化
            # HTML 表格提取
            html_table_pattern = re.compile(
                r"""
            (?:\n|^)
            \s*
            (?:
                (?:<html[^>]*>\s*<body[^>]*>\s*<table[^>]*>.*?</table>\s*</body>\s*</html>)
                |
                (?:<body[^>]*>\s*<table[^>]*>.*?</table>\s*</body>)
                |
                (?:<table[^>]*>.*?</table>)
            )
            \s*
            (?=\n|$)
            """,
                re.VERBOSE | re.DOTALL | re.IGNORECASE,
            )

            def replace_html_tables():
                nonlocal working_text
                new_text = ""
                last_end = 0
                for match in html_table_pattern.finditer(working_text):
                    raw_table = match.group()
                    
                    start_line = working_text.count('\n', 0, match.start())
                    cnt = raw_table.count('\n')
                    end_line = start_line + cnt - 1
                    
                    if raw_table.startswith('\n'):
                        start_line += 1
                        
                    tables.append({
                        "content": raw_table,
                        "start_line": start_line,
                        "end_line": end_line
                    })
                    
                    if separate_tables:
                        if raw_table.startswith('\n'):
                            repl = "\n# RAGFLOW_TABLE_PLACEHOLDER" + ("\n" * (cnt - 1))
                        else:
                            repl = "# RAGFLOW_TABLE_PLACEHOLDER" + ("\n" * cnt)
                        new_text += working_text[last_end : match.start()] + repl
                    else:
                        new_text += working_text[last_end : match.start()] + raw_table + "\n\n"
                    last_end = match.end()
                new_text += working_text[last_end:]
                working_text = new_text

            replace_html_tables()

        return working_text, tables


class MarkdownElementExtractor:
    """Markdown 元素提取器"""
    
    def __init__(self, markdown_content):
        self.markdown_content = markdown_content
        self.lines = markdown_content.split("\n")

    def get_delimiters(self, delimiters):
        """解析分隔符"""
        toks = re.findall(r"`([^`]+)`", delimiters)
        toks = sorted(set(toks), key=lambda x: -len(x))
        return "|".join(re.escape(t) for t in toks if t)

    def extract_elements(self, delimiter=None, include_meta=False):
        """提取各类元素（标题、代码块、列表等）"""
        sections = []

        i = 0
        dels = ""
        if delimiter:
            dels = self.get_delimiters(delimiter)
        if len(dels) > 0:
            text = "\n".join(self.lines)
            if include_meta:
                pattern = re.compile(dels)
                last_end = 0
                for m in pattern.finditer(text):
                    part = text[last_end : m.start()]
                    if part and part.strip():
                        sections.append(
                            {
                                "content": part.strip(),
                                "start_line": text.count("\n", 0, last_end),
                                "end_line": text.count("\n", 0, m.start()),
                            }
                        )
                    last_end = m.end()

                part = text[last_end:]
                if part and part.strip():
                    sections.append(
                        {
                            "content": part.strip(),
                            "start_line": text.count("\n", 0, last_end),
                            "end_line": text.count("\n", 0, len(text)),
                        }
                    )
            else:
                parts = re.split(dels, text)
                sections = [p.strip() for p in parts if p and p.strip()]
            return sections
            
        while i < len(self.lines):
            line = self.lines[i]

            if re.match(r"^#{1,6}\s+.*$", line):
                # 标题
                element = self._extract_header(i)
                sections.append(element if include_meta else element["content"])
                i = element["end_line"] + 1
            elif line.strip().startswith("```"):
                # 代码块
                element = self._extract_code_block(i)
                sections.append(element if include_meta else element["content"])
                i = element["end_line"] + 1
            elif re.match(r"^\s*[-*+]\s+.*$", line) or re.match(r"^\s*\d+\.\s+.*$", line):
                # 列表块
                element = self._extract_list_block(i)
                sections.append(element if include_meta else element["content"])
                i = element["end_line"] + 1
            elif line.strip().startswith(">"):
                # 引用块
                element = self._extract_blockquote(i)
                sections.append(element if include_meta else element["content"])
                i = element["end_line"] + 1
            elif line.strip():
                # 文本块
                element = self._extract_text_block(i)
                sections.append(element if include_meta else element["content"])
                i = element["end_line"] + 1
            else:
                i += 1

        if include_meta:
            sections = [section for section in sections if section["content"].strip()]
        else:
            sections = [section for section in sections if section.strip()]
        return sections

    def _extract_header(self, start_pos):
        """提取标题"""
        return {
            "type": "header",
            "content": self.lines[start_pos],
            "start_line": start_pos,
            "end_line": start_pos,
        }

    def _extract_code_block(self, start_pos):
        """提取代码块"""
        end_pos = start_pos
        content_lines = [self.lines[start_pos]]

        for i in range(start_pos + 1, len(self.lines)):
            content_lines.append(self.lines[i])
            end_pos = i
            if self.lines[i].strip().startswith("```"):
                break

        return {
            "type": "code_block",
            "content": "\n".join(content_lines),
            "start_line": start_pos,
            "end_line": end_pos,
        }

    def _extract_list_block(self, start_pos):
        """提取列表块"""
        end_pos = start_pos
        content_lines = []

        i = start_pos
        while i < len(self.lines):
            line = self.lines[i]
            # 检查是否为列表项或列表的延续
            if (
                re.match(r"^\s*[-*+]\s+.*$", line)
                or re.match(r"^\s*\d+\.\s+.*$", line)
                or (i > start_pos and not line.strip())
                or (i > start_pos and re.match(r"^\s{2,}[-*+]\s+.*$", line))
                or (i > start_pos and re.match(r"^\s{2,}\d+\.\s+.*$", line))
                or (i > start_pos and re.match(r"^\s+\w+.*$", line))
            ):
                content_lines.append(line)
                end_pos = i
                i += 1
            else:
                break

        return {
            "type": "list_block",
            "content": "\n".join(content_lines),
            "start_line": start_pos,
            "end_line": end_pos,
        }

    def _extract_blockquote(self, start_pos):
        """提取引用块"""
        end_pos = start_pos
        content_lines = []

        i = start_pos
        while i < len(self.lines):
            line = self.lines[i]
            if line.strip().startswith(">") or (i > start_pos and not line.strip()):
                content_lines.append(line)
                end_pos = i
                i += 1
            else:
                break

        return {
            "type": "blockquote",
            "content": "\n".join(content_lines),
            "start_line": start_pos,
            "end_line": end_pos,
        }

    def _extract_text_block(self, start_pos):
        """提取文本块（段落及内联元素）"""
        end_pos = start_pos
        content_lines = [self.lines[start_pos]]

        i = start_pos + 1
        while i < len(self.lines):
            line = self.lines[i]
            # 遇到块级元素时停止
            if re.match(r"^#{1,6}\s+.*$", line) or line.strip().startswith("```") or re.match(r"^\s*[-*+]\s+.*$", line) or re.match(r"^\s*\d+\.\s+.*$", line) or line.strip().startswith(">"):
                break
            elif not line.strip():
                # 检查下一行是否为块级元素
                if i + 1 < len(self.lines) and (
                    re.match(r"^#{1,6}\s+.*$", self.lines[i + 1])
                    or self.lines[i + 1].strip().startswith("```")
                    or re.match(r"^\s*[-*+]\s+.*$", self.lines[i + 1])
                    or re.match(r"^\s*\d+\.\s+.*$", self.lines[i + 1])
                    or self.lines[i + 1].strip().startswith(">")
                ):
                    break
                else:
                    content_lines.append(line)
                    end_pos = i
                    i += 1
            else:
                content_lines.append(line)
                end_pos = i
                i += 1

        return {
            "type": "text_block",
            "content": "\n".join(content_lines),
            "start_line": start_pos,
            "end_line": end_pos,
        }

# ================= 来自 deepdoc/parser/txt_parser.py =================
class RAGFlowTxtParser:
    """TXT 文件解析器"""
    
    def __call__(self, fnm, binary=None, chunk_token_num=128, delimiter="\n!?;。；！？"):
        txt = get_text(fnm, binary)
        return self.parser_txt(txt, chunk_token_num, delimiter)

    @classmethod
    def parser_txt(cls, txt, chunk_token_num=128, delimiter="\n!?;。；！？"):
        """解析文本内容为分块"""
        if not isinstance(txt, str):
            raise TypeError("txt 类型应为 str！")
        cks = [""]
        tk_nums = [0]
            
        def add_chunk(t):
            nonlocal cks, tk_nums, delimiter
            tnum = num_tokens_from_string(t)
            if tk_nums[-1] > chunk_token_num:
                cks.append(t)
                tk_nums.append(tnum)
            else:
                cks[-1] += t
                tk_nums[-1] += tnum

        dels = []
        s = 0
        for m in re.finditer(r"`([^`]+)`", delimiter, re.I):
            f, t = m.span()
            dels.append(m.group(1))
            dels.extend(list(delimiter[s: f]))
            s = t
        if s < len(delimiter):
            dels.extend(list(delimiter[s:]))
        dels = [re.escape(d) for d in dels if d]
        dels = [d for d in dels if d]
        dels = "|".join(dels)
        
        if dels:
            secs = re.split(r"(%s)" % dels, txt)
            for sec in secs:
                if re.match(f"^{dels}$", sec):
                    continue
                add_chunk(sec)
        else:
            add_chunk(txt)

        return [[c, ""] for c in cks]


# ================= 来自 deepdoc/parser/json_parser.py =================
class RAGFlowJsonParser:
    """JSON/JSONL 解析与切片（基于 LangChain RecursiveJsonSplitter）。

    说明：
    - 旧实现为 deepdoc/json_parser 的拷贝版本，存在对 max_chunk_size 的 *2 放大等行为，
      容易造成切片粒度不可控；且与项目其他切片策略（按目标 chunk size）不一致。
    - 当前实现直接复用 `langchain_text_splitters.json.RecursiveJsonSplitter`，保持行为一致。
    - 返回值为 `List[str]`：每个元素是一个 JSON 字符串（通常为 JSON object），可直接作为 element。
    """

    def __init__(
        self,
        max_chunk_size: int = 2000,
        min_chunk_size: int | None = None,
        *,
        convert_lists: bool = True,
        ensure_ascii: bool = False,
    ):
        super().__init__()
        try:
            from langchain_text_splitters.json import RecursiveJsonSplitter
        except Exception as e:  # pragma: no cover - optional dependency fallback
            raise RuntimeError("langchain_text_splitters is required for JSON parsing") from e

        self.max_chunk_size = int(max_chunk_size)
        self.min_chunk_size = int(min_chunk_size) if min_chunk_size is not None else None
        self.convert_lists = bool(convert_lists)
        self.ensure_ascii = bool(ensure_ascii)
        self._splitter = RecursiveJsonSplitter(
            max_chunk_size=self.max_chunk_size,
            min_chunk_size=self.min_chunk_size,
        )

    def __call__(self, fnm, binary=None):
        txt = get_text(fnm, binary)
        if self.is_jsonl_format(txt):
            sections = self._parse_jsonl(txt)
        else:
            sections = self._parse_json(txt)
        return sections

    @staticmethod
    def _wrap_root(data: Any) -> dict[str, Any]:
        if isinstance(data, dict):
            return data
        return {"root": data}

    def _parse_json(self, content: str) -> list[str]:
        sections: list[str] = []
        try:
            json_data = json_utils.loads(content)
            wrapped = self._wrap_root(json_data)
            sections = self._splitter.split_text(
                json_data=wrapped,
                convert_lists=self.convert_lists,
                ensure_ascii=self.ensure_ascii,
            )
        except ValueError:
            pass
        return sections

    def _parse_jsonl(self, content: str) -> list[str]:
        lines = content.strip().splitlines()
        all_chunks: list[str] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                data = json_utils.loads(line)
                wrapped = self._wrap_root(data)
                chunks = self._splitter.split_text(
                    json_data=wrapped,
                    convert_lists=self.convert_lists,
                    ensure_ascii=self.ensure_ascii,
                )
                all_chunks.extend(chunk for chunk in chunks if chunk)
            except ValueError:
                continue
        return all_chunks

    def is_jsonl_format(self, txt: str, sample_limit: int = 10, threshold: float = 0.8) -> bool:
        lines = [line.strip() for line in txt.strip().splitlines() if line.strip()]
        if not lines:
            return False

        try:
            json_utils.loads(txt)
            return False
        except ValueError:
            pass

        sample_limit = min(len(lines), sample_limit)
        sample_lines = lines[:sample_limit]
        valid_lines = sum(1 for line in sample_lines if self._is_valid_json(line))

        if not valid_lines:
            return False

        return (valid_lines / len(sample_lines)) >= threshold

    def _is_valid_json(self, line: str) -> bool:
        try:
            json_utils.loads(line)
            return True
        except ValueError:
            return False
