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


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]