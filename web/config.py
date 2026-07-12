"""
Конфигурация веб-приложения RAG-бота.

Поддерживаются два провайдера LLM:
- proxy (основной) — ProxyAPI, OpenAI-совместимый endpoint
- gigachat — GigaChat от Сбера
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent

load_dotenv(BASE_DIR / ".env")
load_dotenv(PROJECT_ROOT / ".env")

# ========== ПРОВАЙДЕР LLM ==========
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "proxy").lower()
if LLM_PROVIDER not in ("proxy", "gigachat"):
    raise ValueError("LLM_PROVIDER должен быть 'proxy' или 'gigachat'")

# ========== PROXY API ==========
PROXY_API_URL = os.getenv("PROXY_API_URL", "https://api.proxyapi.ru/openai/v1")
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "")

PROXY_EMBED_MODEL = os.getenv("PROXY_EMBED_MODEL", "text-embedding-3-small")
PROXY_CHAT_MODEL = os.getenv("PROXY_CHAT_MODEL", "gpt-4o-mini")
PROXY_VISION_MODEL = os.getenv("PROXY_VISION_MODEL", "gpt-4o-mini")

# ========== GIGACHAT ==========
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS", "")
GIGACHAT_VERIFY_SSL = os.getenv("GIGACHAT_VERIFY_SSL_CERTS", "false").lower() == "true"
GIGACHAT_CHAT_MODEL = os.getenv("GIGACHAT_CHAT_MODEL", "GigaChat")
GIGACHAT_EMBED_MODEL = os.getenv("GIGACHAT_EMBED_MODEL", "Embeddings")

# ========== ВАЛИДАЦИЯ КЛЮЧЕЙ ==========
if LLM_PROVIDER == "proxy" and not PROXY_API_KEY:
    raise ValueError("PROXY_API_KEY не установлен! Укажите ключ в .env")
if LLM_PROVIDER == "gigachat" and not GIGACHAT_CREDENTIALS:
    raise ValueError("GIGACHAT_CREDENTIALS не установлен! Укажите ключ в .env")

# ========== FAISS ==========
FAISS_INDEX_PATH = BASE_DIR / f"index_{LLM_PROVIDER}.faiss"
FAISS_METADATA_PATH = BASE_DIR / f"metadata_{LLM_PROVIDER}.json"
DOCS_PATH = PROJECT_ROOT / "data" / "docs"

# ========== RAG ==========
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))
MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "3000"))
MAX_HISTORY_LENGTH = int(os.getenv("MAX_HISTORY_LENGTH", "10"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))

# ========== ВЕБ-СЕРВЕР ==========
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

# ========== ПРОМПТЫ ==========
SYSTEM_PROMPT = """Ты — интеллектуальный ассистент с доступом к базе знаний.
Твоя задача — отвечать на вопросы пользователей, опираясь на предоставленный контекст из базы знаний и историю разговора.

Правила работы:
1. Используй информацию из контекста базы знаний для формирования ответа
2. Учитывай историю предыдущих сообщений для понимания контекста разговора
3. Если пользователь спрашивает про "это", "то", "предыдущий вопрос" — смотри в историю сообщений выше
4. Если в контексте нет информации для ответа, честно скажи об этом
5. Отвечай на русском языке четко и структурированно
6. Если уместно, используй списки и пункты для лучшей читаемости
7. Будь вежливым и профессиональным
"""

RAG_PROMPT_TEMPLATE = """Контекст из базы знаний:
{context}

Вопрос пользователя: {query}

Ответ:"""

# ========== ЛОГИРОВАНИЕ ==========
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
