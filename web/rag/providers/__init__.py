from rag.providers.base import LLMProvider
from rag.providers.proxy_api import ProxyAPIProvider
from rag.providers.gigachat_provider import GigaChatProvider

__all__ = ["LLMProvider", "ProxyAPIProvider", "GigaChatProvider", "get_provider"]


def get_provider(name: str) -> LLMProvider:
    providers = {
        "proxy": ProxyAPIProvider,
        "gigachat": GigaChatProvider,
    }
    if name not in providers:
        raise ValueError(f"Неизвестный провайдер: {name}")
    return providers[name]()
