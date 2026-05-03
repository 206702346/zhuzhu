import os
import shutil

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.core.config import settings
from app.services.document_loader import load_document
from app.services.text_splitter import split_text
from app.services.embedding import get_embedding_model
from app.services.vector_store import vector_store

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    os.makedirs(settings.upload_dir, exist_ok=True)

    file_path = os.path.join(settings.upload_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        text = load_document(file_path)

        chunks = split_text(
            text,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        if not chunks:
            raise HTTPException(status_code=400, detail="文档内容为空，无法建立索引")

        embedding_model = get_embedding_model()
        embeddings = embedding_model.encode(chunks)

        added_count = vector_store.add(
            embeddings=embeddings,
            chunks=chunks,
            source=file.filename,
        )

        return {
            "message": "文档上传并索引成功",
            "filename": file.filename,
            "chunk_count": len(chunks),
            "vector_count": added_count,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_document_stats():
    return vector_store.stats()


@router.delete("/clear")
async def clear_documents():
    vector_store.clear()

    # 同时清空上传目录
    if os.path.exists(settings.upload_dir):
        for filename in os.listdir(settings.upload_dir):
            file_path = os.path.join(settings.upload_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    return {
        "message": "知识库已清空"
    }