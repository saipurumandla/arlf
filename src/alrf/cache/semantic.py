import hashlib
import json
import time
from pathlib import Path

import aiosqlite
import numpy as np

from alrf.cache.embedding import Embedder, HashingEmbedder, cosine
from alrf.models.response import RouterResult

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS cache_entries (
    query_hash  TEXT PRIMARY KEY,
    embedding   BLOB NOT NULL,
    result_json TEXT NOT NULL,
    route       TEXT NOT NULL,
    created_at  REAL NOT NULL,
    last_used   REAL NOT NULL
);
"""


class SemanticCache:
    def __init__(
        self,
        db_path: str = ".alrf/routing.db",
        threshold: float = 0.95,
        embedder: Embedder | None = None,
    ) -> None:
        self._path = Path(db_path)
        self._threshold = threshold
        self._embedder = embedder or HashingEmbedder()

    async def _ensure_db(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._path) as db:
            await db.execute(_CREATE_SQL)
            await db.commit()

    async def lookup(self, query: str) -> RouterResult | None:
        await self._ensure_db()
        vec = self._embedder.embed(query)

        async with aiosqlite.connect(self._path) as db:
            async with db.execute(
                "SELECT query_hash, embedding, result_json FROM cache_entries"
            ) as cur:
                rows = await cur.fetchall()

            best: tuple[str, str] | None = None
            best_score = 0.0
            for query_hash, blob, result_json in rows:
                score = cosine(vec, np.frombuffer(blob, dtype=np.float32))
                if score > best_score:
                    best, best_score = (query_hash, result_json), score

            if best is None or best_score < self._threshold:
                return None

            await db.execute(
                "UPDATE cache_entries SET last_used = ? WHERE query_hash = ?",
                (time.time(), best[0]),
            )
            await db.commit()

        result = RouterResult.model_validate(json.loads(best[1]))
        return result.model_copy(update={"cached": True, "cost_usd": 0.0})

    async def store(self, query: str, result: RouterResult) -> None:
        await self._ensure_db()
        now = time.time()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO cache_entries VALUES (?,?,?,?,?,?)",
                (
                    hashlib.sha256(query.encode()).hexdigest(),
                    self._embedder.embed(query).tobytes(),
                    result.model_dump_json(),
                    result.route,
                    now,
                    now,
                ),
            )
            await db.commit()
