"""
RAG-пайплайн с поддержкой нескольких LLM-провайдеров.
"""

import logging
from typing import Dict, List

from config import (
    LLM_PROVIDER,
    SYSTEM_PROMPT,
    RAG_PROMPT_TEMPLATE,
    TOP_K_RESULTS,
    MAX_CONTEXT_LENGTH,
)
from rag.providers import get_provider
from rag.vectorstore import FAISSVectorStore
from rag.retriever import DocumentRetriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Координирует поиск документов и генерацию ответов."""

    def __init__(self):
        self.provider = get_provider(LLM_PROVIDER)
        self.vectorstore = FAISSVectorStore()
        self.retriever = DocumentRetriever(self.provider, self.vectorstore)
        self.is_loaded = self.vectorstore.load()

        if self.is_loaded:
            logger.info("RAG Pipeline: индекс загружен")
        else:
            logger.warning("RAG Pipeline: индекс не найден, выполните индексацию")

    def query(self, user_query: str, top_k: int = TOP_K_RESULTS) -> Dict:
        return self.query_with_history(user_query, [], top_k)

    def query_with_history(
        self,
        user_query: str,
        history: list = None,
        top_k: int = TOP_K_RESULTS,
    ) -> Dict:
        history = history or []
        logger.info(f"Запрос: '{user_query[:50]}...' (история: {len(history)})")

        if not self.is_loaded:
            return {
                "answer": "База знаний не загружена. Нажмите «Индексация» для загрузки документов.",
                "context": "",
                "sources": [],
                "model": self.provider.chat_model,
            }

        context = self.retriever.retrieve_context(
            user_query, top_k=top_k, max_length=MAX_CONTEXT_LENGTH
        )
        sources = self.retriever.get_relevant_sources(user_query, top_k)

        prompt_with_context = RAG_PROMPT_TEMPLATE.format(context=context, query=user_query)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            recent = history[-20:] if len(history) > 20 else history
            messages.extend(recent)
        messages.append({"role": "user", "content": prompt_with_context})

        try:
            answer = self.provider.chat_completion(messages)
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            answer = f"Ошибка при обращении к API: {e}"

        return {
            "answer": answer,
            "context": context,
            "sources": sources,
            "model": self.provider.chat_model,
        }

    def index_documents(self, documents: List[str], sources: List[str]) -> bool:
        logger.info(f"Индексация {len(documents)} документов")
        try:
            embeddings = self.provider.embed_texts(documents)
            dimension = len(embeddings[0])
            self.vectorstore.create_index(dimension)
            self.vectorstore.add_documents(documents, embeddings, sources)
            self.vectorstore.save()
            self.is_loaded = True
            return True
        except Exception as e:
            logger.error(f"Ошибка индексации: {e}")
            return False

    def get_stats(self) -> Dict:
        stats = self.vectorstore.get_stats()
        stats.update({
            "is_loaded": self.is_loaded,
            "provider": self.provider.name,
            **self.provider.get_info(),
        })
        if hasattr(self.provider, "api_url_display"):
            stats["api_url"] = self.provider.api_url_display
        return stats

    def test_connection(self) -> bool:
        return self.provider.test_connection()
