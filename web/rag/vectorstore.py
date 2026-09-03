"""
Модуль для работы с векторным хранилищем FAISS.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from config import FAISS_INDEX_PATH, FAISS_METADATA_PATH

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """Векторное хранилище на основе FAISS."""

    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = FAISS_METADATA_PATH,
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index = None
        self.metadata = []

    def create_index(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []
        logger.info(f"Создан FAISS индекс, размерность: {dimension}")

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        sources: List[str] = None,
    ):
        if self.index is None:
            raise ValueError("Индекс не инициализирован")

        embeddings_array = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings_array)

        for i, text in enumerate(texts):
            self.metadata.append({
                "text": text,
                "source": sources[i] if sources else f"doc_{i}",
                "index": len(self.metadata),
            })

        logger.info(f"Добавлено {len(texts)} документов, всего: {len(self.metadata)}")

    def search(self, query_embedding: List[float], k: int = 3) -> List[Tuple[str, str, float]]:
        if self.index is None or self.index.ntotal == 0:
            return []

        query_array = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_array, min(k, self.index.ntotal))

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                results.append((meta["text"], meta["source"], float(distances[0][i])))
        return results

    def save(self):
        if self.index is None:
            return

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path.absolute()))

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"Индекс сохранён: {self.index_path}")

    def load(self) -> bool:
        if not self.index_path.exists() or not self.metadata_path.exists():
            return False

        try:
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            logger.info(f"Индекс загружен: {self.index.ntotal} векторов")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки индекса: {e}")
            return False

    def get_stats(self) -> dict:
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "total_documents": len(self.metadata),
            "dimension": self.index.d if self.index else 0,
            "index_exists": self.index_path.exists(),
            "metadata_exists": self.metadata_path.exists(),
        }
