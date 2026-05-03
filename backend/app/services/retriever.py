from typing import List, Dict, Any

from app.core.config import settings
from app.services.embedding import get_embedding_model
from app.services.vector_store import vector_store
from app.services.bm25_retriever import bm25_search


def vector_search(query: str, top_k: int):
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.encode([query])
    return vector_store.search(query_embedding, top_k=top_k)


def hybrid_search(
    query: str,
    top_k: int = None,
    vector_k: int = None,
    bm25_k: int = None,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """
    采用 Reciprocal Rank Fusion (RRF) 融合 BM25 和向量检索结果。

    RRF 公式：
    score(d) = Σ 1 / (k + rank_i(d))
    其中 k 一般取 60 左右。
    """
    if top_k is None:
        top_k = settings.top_k

    if vector_k is None:
        vector_k = max(top_k * 3, top_k)

    if bm25_k is None:
        bm25_k = max(top_k * 3, top_k)

    vector_results = vector_search(query, vector_k)
    bm25_results = bm25_search(query, bm25_k)

    fused = {}

    def add_candidate(item, rank: int, source_type: str):
        idx = item.get("vector_idx") if source_type == "vector" else item.get("bm25_idx")
        if idx is None:
            return

        if idx not in fused:
            fused[idx] = {
                "doc_id": item.get("doc_id"),
                "source": item.get("source"),
                "chunk_id": item.get("chunk_id"),
                "text": item.get("text"),
                "score": 0.0,          # 最终融合分
                "hybrid_score": 0.0,
                "vector_score": None,
                "bm25_score": None,
                "vector_rank": None,
                "bm25_rank": None,
                "match_sources": [],
            }

        cur = fused[idx]
        cur["hybrid_score"] += 1.0 / (rrf_k + rank)
        cur["score"] = cur["hybrid_score"]

        if source_type == "vector":
            cur["vector_score"] = item.get("score")
            cur["vector_rank"] = rank
        else:
            cur["bm25_score"] = item.get("bm25_score")
            cur["bm25_rank"] = rank

        cur["match_sources"] = sorted(set(cur["match_sources"] + [source_type]))

        for key in ["doc_id", "source", "chunk_id", "text"]:
            if cur.get(key) is None and item.get(key) is not None:
                cur[key] = item.get(key)

    for rank, item in enumerate(vector_results, start=1):
        add_candidate(item, rank, "vector")

    for rank, item in enumerate(bm25_results, start=1):
        add_candidate(item, rank, "bm25")

    ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)

    return ranked[:top_k]


def retrieve(query: str, top_k: int = None):
    return hybrid_search(query, top_k=top_k)