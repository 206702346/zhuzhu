from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.document import router as document_router
from app.api.chat import router as chat_router

app = FastAPI(
    title="RAG Course QA System",
    description="基于 RAG 的课程知识库问答系统",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(document_router)
app.include_router(chat_router)


@app.get("/")
async def root():
    return {
        "message": "RAG Course QA System is running."
    }