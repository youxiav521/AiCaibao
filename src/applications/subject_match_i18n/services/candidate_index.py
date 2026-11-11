"""候选科目初筛：倒排索引 + Embedding 救回"""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
from loguru import logger
from openai import AsyncOpenAI

from src.applications.subject_match_i18n.services.template_parser import TemplateSubject
from src.applications.subject_match_i18n.utils.match_utils import _EN_STOP_WORDS

_EMBEDDING_RESCUE_THRESHOLD = 0.45
_EMBEDDING_BATCH_SIZE = 128
_PUNCT_RE = re.compile(r"[,;\-:()\[\]{}./]")


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    cleaned = _PUNCT_RE.sub(" ", text.lower())
    tokens = cleaned.split()
    compact = cleaned.replace(" ", "")
    if compact and len(compact) > 1:
        tokens.append(compact)
    return {t for t in tokens if len(t) > 1 and t not in _EN_STOP_WORDS}


def _subject_tokens(subject: TemplateSubject) -> set[str]:
    tokens: set[str] = set()
    for text in (subject.en_name,):
        tokens.update(_tokenize(text))
    for alias in subject.synonyms:
        tokens.update(_tokenize(alias))
    return tokens


def _subject_embedding_texts(subject: TemplateSubject) -> list[str]:
    texts: list[str] = []
    for text in (subject.en_name,):
        if text and text != "-":
            texts.append(text)
    for alias in subject.synonyms:
        if alias:
            texts.append(alias)
    return texts


def _line_text(line: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("rawName", "cleanedName"):
        val = line.get(key, "")
        if val:
            parts.append(val)
    return " ".join(parts)


class CandidateIndex:
    def __init__(self, subjects: list[TemplateSubject]):
        self._subjects = subjects
        self._code_to_subject: dict[str, TemplateSubject] = {s.code: s for s in subjects}
        self._token_index: dict[str, set[str]] = {}
        self._build_token_index()

    def _build_token_index(self) -> None:
        for s in self._subjects:
            for token in _subject_tokens(s):
                self._token_index.setdefault(token, set()).add(s.code)

    def _token_recall(self, line_tokens: set[str]) -> set[str]:
        recalled: set[str] = set()
        for token in line_tokens:
            recalled.update(self._token_index.get(token, set()))
        return recalled

    def token_recall_codes(self, lines: list[dict[str, Any]]) -> set[str]:
        if not self._subjects or not lines:
            return {s.code for s in self._subjects}

        token_recalled_codes: set[str] = set()
        for line in lines:
            tokens: set[str] = set()
            for key in ("rawName", "cleanedName"):
                tokens.update(_tokenize(line.get(key, "")))
            token_recalled_codes.update(self._token_recall(tokens))
        return token_recalled_codes

    @property
    def code_to_subject(self) -> dict[str, TemplateSubject]:
        return self._code_to_subject


async def embedding_rescue(
    lines: list[dict[str, Any]],
    candidate_subjects: list[TemplateSubject],
) -> tuple[set[str], dict[int, dict[str, float]]]:
    """
    Embedding 语义召回 + per-line cos 分数表。
    返回:
      - rescued: 跨所有 line 最大 cos >= _EMBEDDING_RESCUE_THRESHOLD 的 subject code 集合
      - per_line_sim: {row_idx -> {subject_code -> cos}}，含所有 cos > 0 的候选（按需补充给 LLM）
    """
    api_key = os.environ.get("EMBEDDING_API_KEY_") or os.environ.get("EMBEDDING_API_KEY")
    api_base = os.environ.get("EMBEDDING_API_BASE")
    model = os.environ.get("EMBEDDING_MODEL")
    if not api_key or not api_base or not model:
        logger.warning("Embedding 环境变量未配置，跳过 Embedding 救回")
        return set(), {}

    subject_texts: list[str] = []
    text_to_code: list[str] = []
    for s in candidate_subjects:
        for text in _subject_embedding_texts(s):
            subject_texts.append(text)
            text_to_code.append(s.code)
    line_texts = [_line_text(line) for line in lines]
    line_row_idx = [line.get("rowIndex") for line in lines]

    if not subject_texts or not line_texts:
        logger.warning("Embedding救回跳过 | subject_texts={}, line_texts={}", len(subject_texts), len(line_texts))
        return set(), {}

    all_texts = subject_texts + line_texts
    try:
        vectors = await _embed_texts(all_texts, api_key, api_base, model)
    except Exception:
        logger.exception("Embedding 调用失败，跳过救回")
        return set(), {}

    if not vectors or len(vectors) != len(all_texts):
        logger.warning("Embedding 返回数量不匹配，跳过救回 | expected={}, got={}", len(all_texts), len(vectors))
        return set(), {}

    n_subject_texts = len(subject_texts)
    subject_vecs = _l2_normalize(np.array(vectors[:n_subject_texts], dtype=np.float32))
    line_vecs = _l2_normalize(np.array(vectors[n_subject_texts:], dtype=np.float32))

    # similarity_matrix[i][j] = cos(subject_text_i, line_text_j)
    similarity_matrix = subject_vecs @ line_vecs.T

    # rescued: 跨所有 line 取最大 cos >= 阈值
    max_sim_per_text = similarity_matrix.max(axis=1)
    code_to_max_sim: dict[str, float] = {}
    for i, code in enumerate(text_to_code):
        sim = float(max_sim_per_text[i])
        if code not in code_to_max_sim or sim > code_to_max_sim[code]:
            code_to_max_sim[code] = sim

    rescued: set[str] = set()
    for code, sim in code_to_max_sim.items():
        if sim >= _EMBEDDING_RESCUE_THRESHOLD:
            rescued.add(code)

    # per_line_sim: 每行 × subject code → cos（取多文本聚合的最大值）
    per_line_sim: dict[int, dict[str, float]] = {}
    for j, row_idx in enumerate(line_row_idx):
        if row_idx is None:
            continue
        sims_for_line: dict[str, float] = {}
        for i, code in enumerate(text_to_code):
            sim = float(similarity_matrix[i][j])
            if sim <= 0:
                continue
            if code not in sims_for_line or sim > sims_for_line[code]:
                sims_for_line[code] = sim
        if sims_for_line:
            per_line_sim[row_idx] = sims_for_line

    logger.info(
        "Embedding救回 | 候选科目={}, 救回={}, 阈值={}, per_line候选行={}",
        len(candidate_subjects),
        len(rescued),
        _EMBEDDING_RESCUE_THRESHOLD,
        len(per_line_sim),
    )
    return rescued, per_line_sim


async def embedding_verify(
    line_texts: list[str],
    candidate_texts: list[str],
) -> list[list[float]]:
    """
    批量 embedding 验证：对每条 line_text 与每条 candidate_text 计算余弦相似度。
    返回 similarity_matrix[i][j] = cos(line_texts[i], candidate_texts[j])。
    如 embedding 环境变量未配置，返回空列表。
    """
    api_key = os.environ.get("EMBEDDING_API_KEY_") or os.environ.get("EMBEDDING_API_KEY")
    api_base = os.environ.get("EMBEDDING_API_BASE")
    model = os.environ.get("EMBEDDING_MODEL")
    if not api_key or not api_base or not model:
        logger.warning("Embedding 环境变量未配置，跳过 embedding 验证")
        return []

    if not line_texts or not candidate_texts:
        return []

    all_texts = line_texts + candidate_texts
    try:
        vectors = await _embed_texts(all_texts, api_key, api_base, model)
    except Exception:
        logger.exception("Embedding 验证调用失败")
        return []

    if not vectors or len(vectors) != len(all_texts):
        return []

    n_lines = len(line_texts)
    line_vecs = _l2_normalize(np.array(vectors[:n_lines], dtype=np.float32))
    cand_vecs = _l2_normalize(np.array(vectors[n_lines:], dtype=np.float32))

    sim_matrix = (line_vecs @ cand_vecs.T).tolist()
    return sim_matrix


async def _embed_texts(
    texts: list[str],
    api_key: str,
    api_base: str,
    model: str,
) -> list[list[float]]:
    if not texts:
        return []
    client = AsyncOpenAI(api_key=api_key, base_url=api_base)
    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), _EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + _EMBEDDING_BATCH_SIZE]
        resp = await client.embeddings.create(model=model, input=batch)
        for d in resp.data:
            all_vectors.append(d.embedding)
    return all_vectors


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms
