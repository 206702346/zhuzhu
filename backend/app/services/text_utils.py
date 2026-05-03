import re
import jieba

_PUNCT_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]+")


def tokenize_zh(text: str):
    """
    简单中文分词 + 清洗
    """
    if not text:
        return []

    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)

    tokens = []
    for tok in jieba.lcut(text):
        tok = tok.strip()
        if tok:
            tokens.append(tok)

    return tokens