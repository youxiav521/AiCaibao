import sys
sys.stdout.reconfigure(encoding='utf-8')
import re
import asyncio

_EN_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "to", "for", "at",
    "on", "by", "with", "from", "as", "is", "are", "was", "were",
})

_PUNCT_RE = re.compile(r"[,;\-:()\[\]{}./]")


def _tokenize(text):
    if not text:
        return set()
    cleaned = _PUNCT_RE.sub(" ", text.lower())
    tokens = cleaned.split()
    return {t for t in tokens if len(t) > 1 and t not in _EN_STOP_WORDS}


class FakeSubject:
    def __init__(self, code, en_name, cn_name="", synonyms=None):
        self.code = code
        self.en_name = en_name
        self.cn_name = cn_name or "-"
        self.synonyms = synonyms or []


def _subject_tokens(subject):
    tokens = set()
    for text in (subject.en_name, subject.cn_name):
        tokens.update(_tokenize(text))
    for alias in subject.synonyms:
        tokens.update(_tokenize(alias))
    return tokens


def test_tokenize():
    assert _tokenize("Cash and bank balances") == {"cash", "bank", "balances"}
    assert _tokenize("non-current assets") == {"non", "current", "assets"}
    assert _tokenize("Property, plant and equipment") == {"property", "plant", "equipment"}
    assert _tokenize("") == set()
    assert _tokenize("a an the") == set()
    print("tokenize 测试通过!")


def test_subject_tokens_includes_synonyms():
    s = FakeSubject("AST01", "Property plant and equipment", synonyms=["PPE", "fixed assets"])
    tokens = _subject_tokens(s)
    assert "ppe" in tokens
    assert "fixed" in tokens
    assert "assets" in tokens
    assert "property" in tokens
    print("subject_tokens 测试通过!")


def test_token_recall():
    subjects = [
        FakeSubject("AST01", "Cash and bank balances"),
        FakeSubject("AST02", "Property plant and equipment", synonyms=["PPE"]),
        FakeSubject("AST03", "Trade receivables"),
        FakeSubject("LIA01", "Liabilities", synonyms=["debt"]),
        FakeSubject("LIA02", "Bank overdrafts"),
        FakeSubject("REV01", "Revenue from contracts"),
    ]

    token_index = {}
    for s in subjects:
        for token in _subject_tokens(s):
            token_index.setdefault(token, set()).add(s.code)

    lines = [
        {"rawName": "Bank and cash balances", "cleanedName": ""},
        {"rawName": "PPE", "cleanedName": ""},
        {"rawName": "Debt", "cleanedName": ""},
    ]

    line_tokens_list = []
    for line in lines:
        tokens = set()
        for key in ("rawName", "cleanedName"):
            tokens.update(_tokenize(line.get(key, "")))
        line_tokens_list.append(tokens)

    recalled_codes = set()
    for tokens in line_tokens_list:
        for token in tokens:
            recalled_codes.update(token_index.get(token, set()))

    set_b_codes = {s.code for s in subjects} - recalled_codes

    print(f"原始科目数: {len(subjects)}")
    print(f"倒排命中: {sorted(recalled_codes)}")
    print(f"无共同token: {sorted(set_b_codes)}")

    assert "AST01" in recalled_codes, "Cash and bank balances 应通过倒排命中"
    assert "AST02" in recalled_codes, "PPE 应通过同义词倒排命中"
    assert "LIA01" in recalled_codes, "Liabilities/debt 应通过同义词倒排命中"
    assert "LIA02" in recalled_codes, "Bank overdrafts 应通过共同 token 'bank' 命中"
    assert "REV01" in set_b_codes, "Revenue from contracts 没有共同 token，应在集合 B"

    print("倒排索引测试通过!")


if __name__ == "__main__":
    test_tokenize()
    test_subject_tokens_includes_synonyms()
    test_token_recall()
