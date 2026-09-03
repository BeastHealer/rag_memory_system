"""
Провайдер ProxyAPI — OpenAI-совместимый HTTP endpoint.
"""

import logging
from typing import List, Dict

import requests

from config import (
    PROXY_API_URL,
    PROXY_API_KEY,
    PROXY_EMBED_MODEL,
    PROXY_CHAT_MODEL,
    REQUEST_TIMEOUT,
)
from rag.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class ProxyAPIProvider(LLMProvider):
    """Работа с LLM через ProxyAPI (OpenAI-совместимый API)."""

    def __init__(self):
        self.api_url = PROXY_API_URL.rstrip("/")
        self.api_key = PROXY_API_KEY
        self._embed_model = PROXY_EMBED_MODEL
        self._chat_model = PROXY_CHAT_MODEL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.embeddings_endpoint = f"{self.api_url}/embeddings"
        self.chat_endpoint = f"{self.api_url}/chat/completions"
        logger.info(f"ProxyAPIProvider: {self.api_url}")

    @property
    def name(self) -> str:
        return "proxy"

    @property
    def embed_model(self) -> str:
        return self._embed_model

    @property
    def chat_model(self) -> str:
        return self._chat_model

    @property
    def api_url_display(self) -> str:
        return self.api_url

    def embed_text(self, text: str) -> List[float]:
        payload = {
            "model": self._embed_model,
            "input": text,
            "encoding_format": "float",
        }
        response = requests.post(
            self.embeddings_endpoint,
            headers=self.headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_text(text))
        return embeddings

    def chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = 1000) -> str:
        payload = {
            "model": self._chat_model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": max_tokens,
        }
        response = requests.post(
            self.chat_endpoint,
            headers=self.headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def test_connection(self) -> bool:
        try:
            self.embed_text("test")
            self.chat_completion([{"role": "user", "content": "test"}], max_tokens=5)
            return True
        except Exception as e:
            logger.error(f"ProxyAPI connection test failed: {e}")
            return False
