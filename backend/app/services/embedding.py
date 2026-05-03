from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.config import settings


class EmbeddingModel:
    def __init__(self):
        print(f"正在加载 Embedding 模型: {settings.embedding_model_name}")
        self.model = SentenceTransformer(settings.embedding_model_name)
        print("Embedding 模型加载完成")

    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")


_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model() -> EmbeddingModel:
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = EmbeddingModel()

    return _embedding_model