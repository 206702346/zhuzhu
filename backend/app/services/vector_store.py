import os
import json
import uuid
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np

from app.core.config import settings


class VectorStore:
    def __init__(self):
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.dim = None

        os.makedirs(settings.index_dir, exist_ok=True)
        os.makedirs(settings.metadata_dir, exist_ok=True)

        self.index_path = os.path.join(settings.index_dir, "faiss.index")
        self.metadata_path = os.path.join(settings.metadata_dir, "metadata.json")

    def build_empty_index(self, dim: int):
        self.dim = dim
        # 因为 embedding 已归一化，所以 IndexFlatIP 等价于 cosine similarity 排序
        self.index = faiss.IndexFlatIP(dim)

    def add(
        self,
        embeddings: np.ndarray,
        chunks: List[str],
        source: str,
    ) -> int:
        if embeddings.ndim != 2:
            raise ValueError("embeddings 必须是二维数组")

        n, dim = embeddings.shape

        if self.index is None:
            self.build_empty_index(dim)

        if dim != self.index.d:
            raise ValueError(f"向量维度不一致: index={self.index.d}, input={dim}")

        self.index.add(embeddings)

        doc_id = str(uuid.uuid4())

        for i, chunk in enumerate(chunks):
            self.metadata.append(
                {
                    "doc_id": doc_id,
                    "source": source,
                    "chunk_id": i,
                    "text": chunk,
                }
            )

        self.save()

        return n

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            item = self.metadata[idx].copy()
            item["score"] = float(score)
            item["vector_idx"] = int(idx)
            results.append(item)

        return results

    def save(self):
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            self.dim = self.index.d

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def stats(self) -> Dict[str, Any]:
        sources = set()

        for item in self.metadata:
            sources.add(item.get("source", "unknown"))

        vector_count = 0
        if self.index is not None:
            vector_count = self.index.ntotal

        return {
            "document_count": len(sources),
            "chunk_count": len(self.metadata),
            "vector_count": vector_count,
            "documents": sorted(list(sources)),
        }

    def clear(self):
        self.index = None
        self.metadata = []
        self.dim = None

        if os.path.exists(self.index_path):
            os.remove(self.index_path)

        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)


vector_store = VectorStore()
vector_store.load()