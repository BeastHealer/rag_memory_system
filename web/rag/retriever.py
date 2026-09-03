"""
Модуль для извлечения релевантных документов из векторного хранилища.
"""

import logging
from typing import List, Tuple

from config import TOP_K_RESULTS, MAX_CONTEXT_LENGTH
from rag.providers.base import LLMProvider
from rag.vectorstore import FAISSVectorStore

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """Извлекает релевантные документы по запросу пользователя."""

    def __init__(self, provider: LLMProvider, vectorstore: FAISSVectorStore):
        self.provider = provider
        self.vectorstore = vectorstore

    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Tuple[str, str, float]]:
        query_embedding = self.provider.embed_text(query)
        return self.vectorstore.search(query_embedding, k=top_k)

    def retrieve_context(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        max_length: int = MAX_CONTEXT_LENGTH,
    ) -> str:
        results = self.retrieve(query, top_k)
        return self.build_context(results, max_length)

    def build_context(
        self,
        results: List[Tuple[str, str, float]],
        max_length: int = MAX_CONTEXT_LENGTH,
    ) -> str:
        if not results:
            return "Релевантная информация не найдена в базе знаний."

        context_parts = []
        total_length = 0

        for i, (text, source, _distance) in enumerate(results, 1):
            doc_text = f"[Документ {i} из {source}]\n{text}\n"
            if total_length + len(doc_text) > max_length:
                remaining = max_length - total_length
                if remaining > 100:
                    context_parts.append(doc_text[:remaining] + "...\n")
                break
            context_parts.append(doc_text)
            total_length += len(doc_text)

        return "\n".join(context_parts)

    def get_relevant_sources(self, query: str, top_k: int = TOP_K_RESULTS) -> List[str]:
        results = self.retrieve(query, top_k)
        return self.sources_from_results(results)

    @staticmethod
    def sources_from_results(results: List[Tuple[str, str, float]]) -> List[str]:
        return list(dict.fromkeys(source for _, source, _ in results))
