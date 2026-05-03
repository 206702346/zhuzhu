from fastapi import APIRouter

from app.schemas.chat import ChatRequest, ChatResponse, SourceItem
from app.services.retriever import retrieve
from app.services.llm import build_prompt, call_llm

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest):
    contexts = retrieve(request.question, top_k=request.top_k)

    prompt = build_prompt(request.question, contexts)
    answer = call_llm(prompt)

    sources = [
        SourceItem(
            source=item["source"],
            chunk_id=item["chunk_id"],
            score=item["score"],
            text=item["text"],
        )
        for item in contexts
    ]

    return ChatResponse(
        answer=answer,
        sources=sources,
    )