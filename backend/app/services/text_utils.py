import re
from typing import List

import jieba

_PUNCT_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]+")
_SPACE_RE = re.compile(r"\s+")

# 轻量停用词，够你这个项目用了
STOPWORDS = {
    "什么", "如何", "怎么", "请问", "一个", "一种", "以及", "是否", "哪些",
    "那个", "这个", "这些", "那些", "为什么", "怎样", "多少", "有没有",
    "是", "的", "了", "在", "与", "和", "及", "有", "就", "对", "吗", "呢", "啊",
    "把", "被", "其", "这", "那", "中", "里", "上", "下", "是否", "可以",
}


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


def tokenize_zh(text: str) -> List[str]:
    """
    中文分词 + 清洗
    """
    if not text:
        return []

    text = normalize_text(text)

    tokens = []
    for tok in jieba.lcut(text):
        tok = tok.strip()
        if not tok:
            continue
        if tok in STOPWORDS:
            continue
        # 过滤太短且没有信息量的符号
        if len(tok) == 1 and not tok.isalnum():
            continue
        tokens.append(tok)

    return tokens


def extract_keywords(query: str) -> List[str]:
    """
    从 query 中提取去重后的关键词
    """
    tokens = tokenize_zh(query)
    seen = set()
    keywords = []

    for tok in tokens:
        if tok in seen:
            continue
        seen.add(tok)
        keywords.append(tok)

    return keywords


def calc_lexical_score(query: str, text: str) -> float:
    """
    词面命中分，范围 [0, 1]

    计算方式：
    - query 的关键词命中率
    - 再加一点“整句/短语精确匹配”的 bonus
    """
    q_keywords = extract_keywords(query)
    if not q_keywords:
        return 0.0

    text_tokens = set(tokenize_zh(text))
    hit_terms = [tok for tok in q_keywords if tok in text_tokens]

    coverage = len(hit_terms) / len(q_keywords)

    nq = normalize_text(query).replace(" ", "")
    nt = normalize_text(text).replace(" ", "")
    exact = 1.0 if nq and nq in nt else 0.0

    # coverage 为主，exact 为辅
    score = 0.85 * coverage + 0.15 * exact
    return min(1.0, float(score))