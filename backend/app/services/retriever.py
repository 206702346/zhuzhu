from typing import List, Dict, Any

from app.core.config import settings
from app.services.embedding import get_embedding_model
from app.services.vector_store import vector_store
from app.services.bm25_retriever import bm25_search
from app.services.reranker import get_reranker
from app.services.text_utils import calc_lexical_score


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
    RRF 融合 BM25 和向量检索结果
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

                # 融合分
                "hybrid_score": 0.0,
                "score": 0.0,

                # 原始检索分
                "vector_score": None,
                "bm25_score": None,

                # 排名
                "vector_rank": None,
                "bm25_rank": None,
                "hybrid_rank": None,
                "rerank_rank": None,

                # rerank 分
                "rerank_score": None,

                # 词面分
                "lexical_score": None,

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
    CrossEncoder rerank + 词面加权
    """
    if not candidates:
        return []

    lexical_weight = max(0.0, float(settings.lexical_boost_weight))

    # 先计算词面分
    candidates_with_lexical = []
    for item in candidates:
        new_item = item.copy()
        lexical_score = calc_lexical_score(query, new_item.get("text", ""))
        new_item["lexical_score"] = float(lexical_score)
        candidates_with_lexical.append(new_item)

    # 如果关闭 rerank，就只做 hybrid + lexical boost
    if not settings.rerank_enabled:
        for rank, item in enumerate(candidates_with_lexical, start=1):
            item["rerank_rank"] = rank
            item["rerank_score"] = None

            base_score = float(item.get("hybrid_score", item.get("score", 0.0)))
            final_score = base_score * (1.0 + lexical_weight * float(item.get("lexical_score", 0.0)))
            item["score"] = float(final_score)

        candidates_with_lexical.sort(key=lambda x: x["score"], reverse=True)

        for rank, item in enumerate(candidates_with_lexical, start=1):
            item["rerank_rank"] = rank

        return candidates_with_lexical[:top_k]

    try:
        reranker = get_reranker()
        passages = [item["text"] for item in candidates_with_lexical]
        raw_scores = reranker.score(query, passages)

        scored = []
        for item, rerank_score in zip(candidates_with_lexical, raw_scores):
            new_item = item.copy()
            new_item["rerank_score"] = float(rerank_score)

            lexical_score = float(new_item.get("lexical_score", 0.0))
            final_score = float(rerank_score) * (1.0 + lexical_weight * lexical_score)

            new_item["score"] = float(final_score)
            scored.append(new_item)

        scored.sort(key=lambda x: x["score"], reverse=True)

        for rank, item in enumerate(scored, start=1):
            item["rerank_rank"] = rank

        return scored[:top_k]

    except Exception as e:
        print(f"[WARN] rerank 失败，回退到 hybrid + lexical：{e}")

        for rank, item in enumerate(candidates_with_lexical, start=1):
            item["rerank_rank"] = None
            item["rerank_score"] = None

            base_score = float(item.get("hybrid_score", item.get("score", 0.0)))
            final_score = base_score * (1.0 + lexical_weight * float(item.get("lexical_score", 0.0)))
            item["score"] = float(final_score)

        candidates_with_lexical.sort(key=lambda x: x["score"], reverse=True)
        return candidates_with_lexical[:top_k]


def retrieve(query: str, top_k: int = None) -> List[Dict[str, Any]]:
    if top_k is None:
        top_k = settings.top_k

    candidate_k = max(top_k * 5, 10)
    candidates = hybrid_search(query, candidate_k=candidate_k)
    final_results = rerank_candidates(query, candidates, top_k=top_k)

    return final_results