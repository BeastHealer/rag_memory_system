"""
Провайдер GigaChat — LLM от Сбера.
"""

import logging
from typing import List, Dict

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

from config import (
    GIGACHAT_CREDENTIALS,
    GIGACHAT_VERIFY_SSL,
    GIGACHAT_CHAT_MODEL,
    GIGACHAT_EMBED_MODEL,
)
from rag.providers.base import LLMProvider

logger = logging.getLogger(__name__)

ROLE_MAP = {
    "system": MessagesRole.SYSTEM,
    "user": MessagesRole.USER,
    "assistant": MessagesRole.ASSISTANT,
}


class GigaChatProvider(LLMProvider):
    """Работа с LLM через GigaChat API."""

    def __init__(self):
        self._embed_model = GIGACHAT_EMBED_MODEL
        self._chat_model = GIGACHAT_CHAT_MODEL
        self._client_kwargs = {
            "credentials": GIGACHAT_CREDENTIALS,
            "verify_ssl_certs": GIGACHAT_VERIFY_SSL,
            "model": self._chat_model,
        }
        logger.info("GigaChatProvider initialized")

    def _get_client(self) -> GigaChat:
        return GigaChat(**self._client_kwargs)

    @property
    def name(self) -> str:
        return "gigachat"

    @property
    def embed_model(self) -> str:
        return self._embed_model

    @property
    def chat_model(self) -> str:
        return self._chat_model

    def embed_text(self, text: str) -> List[float]:
        with self._get_client() as client:
            result = client.embeddings([text], model=self._embed_model)
            return result.data[0].embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        with self._get_client() as client:
            for text in texts:
                result = client.embeddings([text], model=self._embed_model)
                embeddings.append(result.data[0].embedding)
        return embeddings

    def chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = 1000) -> str:
        giga_messages = [
            Messages(role=ROLE_MAP.get(m["role"], MessagesRole.USER), content=m["content"])
            for m in messages
        ]
        chat = Chat(messages=giga_messages, temperature=0.7, max_tokens=max_tokens)
        with self._get_client() as client:
            response = client.chat(chat)
            return response.choices[0].message.content

    def test_connection(self) -> bool:
        try:
            self.embed_text("test")
            self.chat_completion([{"role": "user", "content": "test"}], max_tokens=5)
            return True
        except Exception as e:
            logger.error(f"GigaChat connection test failed: {e}")
            return False
