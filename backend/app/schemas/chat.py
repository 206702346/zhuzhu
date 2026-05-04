from typing import List, Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class SourceItem(BaseModel):
    source: str
    chunk_id: int
    score: float
    text: str

    hybrid_score: Optional[float] = None
    rerank_score: Optional[float] = None

    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None

    hybrid_rank: Optional[int] = None
    rerank_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    bm25_rank: Optional[int] = None

    match_sources: Optional[List[str]] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]