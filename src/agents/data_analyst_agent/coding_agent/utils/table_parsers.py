import pandas as pd
import json
from io import StringIO
from typing import Optional, Callable, Dict


def _is_valid_table_string(table_str: str) -> bool:
    """Validate that table string is non-empty after stripping whitespace."""
    if table_str is None:
        return False
    if not isinstance(table_str, str):
        table_str = str(table_str)
    return bool(table_str.strip())


def parse_csv_tsv(table_str: str, sep: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Parse CSV or TSV format table string to DataFrame.
    sep: None for auto-detection, or explicitly specify ','/';'/'\t'
    """
    if not _is_valid_table_string(table_str):
        return None
    try:
        if sep is None:
            sep = "\t" if "\t" in table_str else ","
        return pd.read_csv(StringIO(table_str), sep=sep)
    except Exception:
        return None


def parse_json(table_str: str) -> Optional[pd.DataFrame]:
    """
    Parse JSON or JSONL format table string to DataFrame.
    Supports standard JSON array and JSONL (one JSON object per line).
    """
    if not _is_valid_table_string(table_str):
        return None
    try:
        data = json.loads(table_str)
        return pd.DataFrame(data)
    except Exception:
        try:
            lines = [json.loads(line) for line in table_str.strip().splitlines() if line.strip()]
            return pd.DataFrame(lines)
        except Exception:
            return None


def parse_markdown_table(table_str: str) -> Optional[pd.DataFrame]:
    """Parse Markdown format table string to DataFrame."""
    if not _is_valid_table_string(table_str):
        return None
    try:
        lines = [line for line in table_str.splitlines() if "|" in line]
        if not lines:
            return None

        clean_lines = []
        for line in lines:
            if set(line.replace("|", "").replace(" ", "")) <= set("-:"):
                continue
            clean_lines.append(line)

        if len(clean_lines) < 2:
            return None

        md_str = "\n".join([l.strip().strip("|") for l in clean_lines])
        return pd.read_csv(StringIO(md_str), sep="|")
    except Exception:
        return None


def parse_html_table(table_str: str) -> Optional[pd.DataFrame]:
    """Parse HTML format table string to DataFrame."""
    if not _is_valid_table_string(table_str):
        return None
    try:
        dfs = pd.read_html(StringIO(table_str))
        return dfs[0] if dfs else None
    except Exception:
        return None


# Parser dispatch table mapping type names to handler functions
_PARSER_DISPATCH: Dict[str, Callable[[str], Optional[pd.DataFrame]]] = {
    "csv": lambda s: parse_csv_tsv(s, sep=","),
    "tsv": lambda s: parse_csv_tsv(s, sep="\t"),
    "json": parse_json,
    "jsonl": parse_json,
    "markdown": parse_markdown_table,
    "html": parse_html_table,
}


def parse_table_by_type(table_str: str, table_type: str) -> Optional[pd.DataFrame]:
    """
    Parse table string to DataFrame based on specified type.
    table_type: 'csv', 'tsv', 'json', 'jsonl', 'markdown', 'html'
    """
    table_type = table_type.lower()
    parser = _PARSER_DISPATCH.get(table_type)
    if parser:
        return parser(table_str)
    return parse_csv_tsv(table_str)
