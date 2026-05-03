from typing import List, Dict, Any

import numpy as np
from rank_bm25 import BM25Okapi

from app.services.vector_store import vector_store
from app.services.text_utils import tokenize_zh


def bm25_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    基于当前 vector_store.metadata 动态构建 BM25 索引并检索。
    适合当前项目阶段，数据量不大时完全够用。
    """
    if not vector_store.metadata:
        return []

    corpus_tokens = [tokenize_zh(item.get("text", "")) for item in vector_store.metadata]
    query_tokens = tokenize_zh(query)

    if not query_tokens:
        return []

    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            continue

        item = vector_store.metadata[idx].copy()
        item["bm25_score"] = score
        item["bm25_idx"] = int(idx)
        results.append(item)

    return results