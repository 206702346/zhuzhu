from typing import List, Dict, Any

from app.core.config import settings
from app.services.embedding import get_embedding_model
from app.services.vector_store import vector_store
from app.services.bm25_retriever import bm25_search
from app.services.reranker import get_reranker


def vector_search(query: str, top_k: int) -> List[Dict[str, Any]]:
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.encode([query])
    return vector_store.search(query_embedding, top_k=top_k)


def hybrid_search(
    query: str,
    candidate_k: int = None,
    vector_k: int = None,
    bm25_k: int = None,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """
    RRF 融合 BM25 和向量检索结果。
    """
    if candidate_k is None:
        candidate_k = max(settings.top_k * 5, 10)

    if vector_k is None:
        vector_k = max(candidate_k, 10)

    if bm25_k is None:
        bm25_k = max(candidate_k, 10)

    vector_results = vector_search(query, vector_k)
    bm25_results = bm25_search(query, bm25_k)

    fused = {}

    def add_candidate(item: Dict[str, Any], rank: int, source_type: str):
        idx = item.get("vector_idx") if source_type == "vector" else item.get("bm25_idx")
        if idx is None:
            return

        if idx not in fused:
            fused[idx] = {
                "doc_id": item.get("doc_id"),
                "source": item.get("source"),
                "chunk_id": item.get("chunk_id"),
                "text": item.get("text"),
                "hybrid_score": 0.0,
                "score": 0.0,  # 先暂存为融合分，后面 rerank 会覆盖成最终分
                "vector_score": None,
                "bm25_score": None,
                "vector_rank": None,
                "bm25_rank": None,
                "hybrid_rank": None,
                "rerank_score": None,
                "rerank_rank": None,
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

    ranked = sorted(fused.values(), key=lambda x: x["hybrid_score"], reverse=True)

    for rank, item in enumerate(ranked, start=1):
        item["hybrid_rank"] = rank
        item["score"] = item["hybrid_score"]

    return ranked[:candidate_k]


def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    使用 CrossEncoder 对候选片段进行重排序。
    """
    if not candidates:
        return []

    if not settings.rerank_enabled:
        for rank, item in enumerate(candidates, start=1):
            item["rerank_rank"] = rank
            item["rerank_score"] = None
            item["score"] = item.get("hybrid_score", item.get("score", 0.0))
        return candidates[:top_k]

    try:
        reranker = get_reranker()
        passages = [item["text"] for item in candidates]
        scores = reranker.score(query, passages)

        scored = []
        for item, rerank_score in zip(candidates, scores):
            new_item = item.copy()
            new_item["rerank_score"] = float(rerank_score)
            new_item["score"] = float(rerank_score)  # 最终分，用于兼容前端
            scored.append(new_item)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)

        for rank, item in enumerate(scored, start=1):
            item["rerank_rank"] = rank

        return scored[:top_k]

    except Exception as e:
        print(f"[WARN] rerank 失败，回退到混合检索结果: {e}")

        for rank, item in enumerate(candidates, start=1):
            item["rerank_rank"] = None
            item["rerank_score"] = None
            item["score"] = item.get("hybrid_score", item.get("score", 0.0))

        return candidates[:top_k]


def retrieve(query: str, top_k: int = None) -> List[Dict[str, Any]]:
    if top_k is None:
        top_k = settings.top_k

    candidate_k = max(top_k * 5, 10)
    candidates = hybrid_search(query, candidate_k=candidate_k)
    final_results = rerank_candidates(query, candidates, top_k=top_k)

    return final_results