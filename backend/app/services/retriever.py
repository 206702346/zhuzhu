from typing import List, Dict, Any

from app.core.config import settings
from app.services.embedding import get_embedding_model
from app.services.vector_store import vector_store


def retrieve(query: str, top_k: int = None) -> List[Dict[str, Any]]:
    if top_k is None:
        top_k = settings.top_k

    embedding_model = get_embedding_model()
    query_embedding = embedding_model.encode([query])
    results = vector_store.search(query_embedding, top_k=top_k)

    return results