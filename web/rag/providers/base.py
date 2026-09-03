"""
Абстрактный базовый класс для LLM-провайдеров.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class LLMProvider(ABC):
    """Интерфейс провайдера: эмбеддинги, чат, проверка подключения."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def embed_model(self) -> str:
        pass

    @property
    @abstractmethod
    def chat_model(self) -> str:
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = 1000) -> str:
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        pass

    def get_info(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "embed_model": self.embed_model,
            "chat_model": self.chat_model,
        }
