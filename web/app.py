"""
Веб-приложение RAG-бота — диалоговый интерфейс.
"""

import time
import logging
import uuid
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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
    LOGS_DB_PATH,
    LOGS_CSV_PATH,
)
from rag.pipeline import RAGPipeline
from db_logger import DatabaseLogger

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="RAG Bot", description="Диалоговый интерфейс RAG-бота")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

rag_pipeline = RAGPipeline()
logger_db = DatabaseLogger(db_path=str(LOGS_DB_PATH))
conversation_history: Dict[str, List[dict]] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class SessionRequest(BaseModel):
    session_id: str


def _split_sentences(paragraph: str) -> list[str]:
    """Разбивает абзац на предложения по '. '."""
    parts = paragraph.split(". ")
    sentences = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        if i < len(parts) - 1 and not part.endswith((".", "!", "?")):
            part += "."
        sentences.append(part)
    return sentences


def _overlap_sentences(sentences: list[str], overlap_ratio: float) -> list[str]:
    """Берёт хвостовые предложения, покрывающие ~overlap_ratio длины чанка."""
    if not sentences or overlap_ratio <= 0:
        return []
    chunk_text = " ".join(sentences)
    target = int(len(chunk_text) * overlap_ratio)
    if target <= 0:
        return []

    overlap: list[str] = []
    overlap_len = 0
    for sentence in reversed(sentences):
        add = len(sentence) + (1 if overlap else 0)
        if overlap and overlap_len + add > target:
            break
        overlap.insert(0, sentence)
        overlap_len += add
        if overlap_len >= target:
            break
    return overlap


def smart_chunk_text(text: str, chunk_size: int = 500, overlap_ratio: float = 0.2) -> list[str]:
    """
    Умное разбиение на чанки:
    1. Сначала по абзацам (\\n\\n)
    2. Если абзац слишком длинный — по предложениям
    3. Перекрытие (overlap) ~20% между соседними чанками длинного абзаца
    """
    chunks: list[str] = []

    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue

        if len(para) <= chunk_size:
            chunks.append(para)
            continue

        sentences = _split_sentences(para)
        if not sentences:
            chunks.append(para[:chunk_size])
            continue

        current: list[str] = []
        for sentence in sentences:
            prospective = " ".join(current + [sentence]) if current else sentence
            if current and len(prospective) > chunk_size:
                chunks.append(" ".join(current))
                current = _overlap_sentences(current, overlap_ratio) + [sentence]
            else:
                current.append(sentence)

        if current:
            chunks.append(" ".join(current))

    return chunks


def load_documents_from_directory(directory: Path):
    documents = []
    sources = []

    if not directory.exists():
        return documents, sources

    for file_path in directory.glob("*.txt"):
        try:
            text = file_path.read_text(encoding="utf-8")
            chunks = smart_chunk_text(text)
            if not chunks:
                continue
            for i, chunk in enumerate(chunks, 1):
                documents.append(chunk)
                if len(chunks) == 1:
                    sources.append(file_path.name)
                else:
                    sources.append(f"{file_path.name} (часть {i}/{len(chunks)})")
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
    query = req.message.strip()

    start_time = time.time()
    try:
        result = rag_pipeline.query_with_history(query, history)
    except Exception as e:
        logger.exception("Ошибка обработки запроса")
        result = {
            "answer": f"Ошибка при обработке запроса: {e}",
            "sources": [],
            "model": "error",
            "from_cache": False,
        }
    response_time_ms = int((time.time() - start_time) * 1000)

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": result["answer"]})

    max_messages = MAX_HISTORY_LENGTH * 2
    if len(history) > max_messages:
        conversation_history[session_id] = history[-max_messages:]

    sources = result.get("sources", [])
    source_str = ", ".join(sources) if isinstance(sources, list) else str(sources or "Unknown")

    logger_db.log_interaction(
        query=query,
        response=result["answer"],
        source=source_str,
        session_id=session_id,
        from_cache=result.get("from_cache", False),
        response_time_ms=response_time_ms,
    )

    return {
        "answer": result["answer"],
        "sources": sources,
        "session_id": session_id,
        "model": result.get("model"),
        "from_cache": result.get("from_cache", False),
        "response_time_ms": response_time_ms,
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


@app.get("/api/logs/stats")
async def log_stats():
    """Возвращает метрики логирования для мониторинга."""
    try:
        return logger_db.get_stats()
    except Exception as e:
        logger.error(f"Ошибка чтения статистики логов: {e}")
        return JSONResponse({"error": "Не удалось получить статистику логов"}, status_code=500)


@app.get("/api/logs/export")
async def export_logs():
    """Скачивает CSV-файл с логами для анализа."""
    try:
        logger_db.export_to_csv(str(LOGS_CSV_PATH))
    except Exception as e:
        logger.error(f"Ошибка экспорта логов: {e}")
        return JSONResponse({"error": "Не удалось выгрузить логи"}, status_code=500)
    return FileResponse(str(LOGS_CSV_PATH), media_type="text/csv", filename="logs.csv")


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Запуск RAG Web App (провайдер: {LLM_PROVIDER})")
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
