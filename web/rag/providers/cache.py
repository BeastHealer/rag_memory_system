import sqlite3
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SQLiteCache:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    query_hash TEXT PRIMARY KEY,
                    query TEXT,
                    answer TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def get(self, query: str) -> str | None:
        query_hash = hashlib.sha256(query.encode('utf-8')).hexdigest()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT answer FROM cache WHERE query_hash = ?", (query_hash,))
            result = cursor.fetchone()
            if result:
                logger.info("✅ Ответ мгновенно получен из кэша (SQLite)")
                return result[0]
        return None

    def set(self, query: str, answer: str):
        query_hash = hashlib.sha256(query.encode('utf-8')).hexdigest()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cache (query_hash, query, answer)
                VALUES (?, ?, ?)
            """, (query_hash, query, answer))
            conn.commit()
        logger.info("💾 Новый ответ сохранен в кэш (SQLite)")

    def get_stats(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM cache")
            count = cursor.fetchone()[0]
        return {"cached_queries": count}