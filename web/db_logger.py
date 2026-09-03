"""
Логирование взаимодействий пользователя с RAG-ботом в SQLite.
Отдельно от кэша ответов: хранит запросы, ответы, источники, сессии и время ответа.
"""

import csv
import logging
import sqlite3
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


class DatabaseLogger:
    """
    Класс для логирования взаимодействий в SQLite базу данных.
    Хранит: вопросы, ответы, метаданные (время, источник, session_id), статус кэша.
    """

    def __init__(self, db_path: str = "logs.db"):
        self.db_path = Path(db_path)
        self._init_database()

    def _init_database(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT,
                    query TEXT,
                    response TEXT,
                    source TEXT,
                    from_cache BOOLEAN,
                    response_time_ms INTEGER
                )
            """)
            conn.commit()

    def log_interaction(
        self,
        query: str,
        response: str,
        source: str,
        session_id: str,
        from_cache: bool,
        response_time_ms: int
    ) -> None:
        """Записывает одно взаимодействие пользователя с ассистентом."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO interactions (session_id, query, response, source, from_cache, response_time_ms)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, query, response, source, from_cache, response_time_ms),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Не удалось записать лог взаимодействия: {e}")

    def get_stats(self) -> Dict:
        """Возвращает агрегированную статистику по логам."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM interactions")
            total = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM interactions WHERE from_cache = 1")
            cached = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT session_id) FROM interactions")
            unique_sessions = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(response_time_ms) FROM interactions")
            avg_time = cursor.fetchone()[0]

            return {
                "total_interactions": total,
                "cached_interactions": cached,
                "unique_sessions": unique_sessions,
                "cache_hit_rate": round(cached / total, 4) if total else 0.0,
                "avg_response_time_ms": int(avg_time) if avg_time is not None else 0,
            }

    def export_to_csv(self, csv_path: str = "logs.csv") -> None:
        """Экспортирует логи в CSV для последующего анализа (например, в Excel или для дообучения)."""
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, timestamp, session_id, query, response, source, from_cache, response_time_ms
                FROM interactions
                """
            )
            rows = cursor.fetchall()

        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["id", "timestamp", "session_id", "query", "response", "source", "from_cache", "response_time_ms"]
            )
            writer.writerows(rows)
 