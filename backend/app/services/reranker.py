from typing import List, Optional

from sentence_transformers import CrossEncoder

from app.core.config import settings


class RerankerModel:
    def __init__(self):
        print(f"正在加载 Reranker 模型: {settings.rerank_model_name}")
        self.model = CrossEncoder(
            settings.rerank_model_name,
            max_length=512,
        )
        print("Reranker 模型加载完成")

    def score(self, query: str, passages: List[str]) -> List[float]:
        pairs = [[query, passage] for passage in passages]
        scores = self.model.predict(
            pairs,
            batch_size=8,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]


_reranker: Optional[RerankerModel] = None


def get_reranker() -> RerankerModel:
    global _reranker

    if _reranker is None:
        _reranker = RerankerModel()

    return _reranker