"""
Веб-приложение RAG-бота — диалоговый интерфейс.
"""

import logging
import uuid
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from config import (
    DOCS_PATH,
    LOG_LEVEL,
    LOG_FORMAT,
    LLM_PROVIDER,
    MAX_HISTORY_LENGTH,
    HOST,
    PORT,
)
from rag.pipeline import RAGPipeline

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Bot", description="Диалоговый интерфейс RAG-бота")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

rag_pipeline = RAGPipeline()
conversation_history: Dict[str, List[dict]] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class SessionRequest(BaseModel):
    session_id: str


def chunk_text(text: str, max_chars: int = 6000) -> List[str]:
    if len(text) <= max_chars:
        return [text]

    chunks = []
    current_chunk = []
    current_length = 0

    for paragraph in text.split("\n\n"):
        paragraph_length = len(paragraph) + 2
        if paragraph_length > max_chars:
            for sentence in paragraph.split(". "):
                sentence_length = len(sentence) + 2
                if current_length + sentence_length > max_chars and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length
        elif current_length + paragraph_length > max_chars and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [paragraph]
            current_length = paragraph_length
        else:
            current_chunk.append(paragraph)
            current_length += paragraph_length

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks


def load_documents_from_directory(directory: Path):
    documents = []
    sources = []

    if not directory.exists():
        return documents, sources

    for file_path in directory.glob("*.txt"):
        try:
            text = file_path.read_text(encoding="utf-8")
            chunks = chunk_text(text)
            if len(chunks) > 1:
                for i, chunk in enumerate(chunks, 1):
                    documents.append(chunk)
                    sources.append(f"{file_path.name} (часть {i}/{len(chunks)})")
            else:
                documents.append(text)
                sources.append(file_path.name)
        except Exception as e:
            logger.error(f"Ошибка чтения {file_path}: {e}")

    return documents, sources


def get_or_create_session(session_id: str | None) -> str:
    sid = session_id or str(uuid.uuid4())
    if sid not in conversation_history:
        conversation_history[sid] = []
    return sid


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    stats = rag_pipeline.get_stats()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "provider": LLM_PROVIDER,
            "is_loaded": stats["is_loaded"],
            "total_documents": stats["total_documents"],
        },
    )


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        return JSONResponse({"error": "Сообщение не может быть пустым"}, status_code=400)

    session_id = get_or_create_session(req.session_id)
    history = conversation_history[session_id]

    result = rag_pipeline.query_with_history(req.message.strip(), history)

    history.append({"role": "user", "content": req.message.strip()})
    history.append({"role": "assistant", "content": result["answer"]})

    max_messages = MAX_HISTORY_LENGTH * 2
    if len(history) > max_messages:
        conversation_history[session_id] = history[-max_messages:]

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "session_id": session_id,
        "model": result["model"],
    }


@app.post("/api/clear")
async def clear_session(req: SessionRequest):
    if req.session_id in conversation_history:
        count = len(conversation_history[req.session_id]) // 2
        conversation_history[req.session_id] = []
        return {"cleared": True, "messages_removed": count}
    return {"cleared": False, "messages_removed": 0}


@app.post("/api/ingest")
async def ingest():
    documents, sources = load_documents_from_directory(DOCS_PATH)

    if not documents:
        return JSONResponse(
            {"error": f"Документы не найдены в {DOCS_PATH}. Добавьте .txt файлы."},
            status_code=400,
        )

    success = rag_pipeline.index_documents(documents, sources)
    if not success:
        return JSONResponse({"error": "Ошибка при индексации"}, status_code=500)

    stats = rag_pipeline.get_stats()
    return {
        "success": True,
        "documents": stats["total_documents"],
        "vectors": stats["total_vectors"],
        "dimension": stats["dimension"],
    }


@app.get("/api/stats")
async def stats():
    return rag_pipeline.get_stats()


@app.get("/api/test")
async def test_connection():
    ok = rag_pipeline.test_connection()
    return {"ok": ok, "provider": LLM_PROVIDER}


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Запуск RAG Web App (провайдер: {LLM_PROVIDER})")
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
