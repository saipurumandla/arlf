import hashlib
import uuid
from pathlib import Path

import aiosqlite
import structlog

from alrf.models.response import RouterResult

logger = structlog.get_logger()

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS routing_events (
    id           TEXT PRIMARY KEY,
    query_hash   TEXT NOT NULL,
    route        TEXT NOT NULL,
    provider     TEXT NOT NULL,
    model        TEXT NOT NULL,
    policy       TEXT NOT NULL,
    latency_ms   INTEGER NOT NULL,
    cost_usd     REAL,
    confidence   REAL NOT NULL,
    escalated    INTEGER NOT NULL DEFAULT 0,
    rag_used     INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class ObservabilityStore:
    def __init__(self, db_path: str = ".alrf/routing.db") -> None:
        self._path = Path(db_path)

    async def _ensure_db(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._path) as db:
            await db.execute(_CREATE_SQL)
            await db.commit()

    async def record(self, query: str, result: RouterResult, policy: str) -> None:
        try:
            await self._ensure_db()
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            async with aiosqlite.connect(self._path) as db:
                await db.execute(
                    "INSERT INTO routing_events "
                    "(id,query_hash,route,provider,model,policy,latency_ms,"
                    "cost_usd,confidence,escalated,rag_used) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        str(uuid.uuid4()), query_hash, result.route,
                        result.provider, result.model, policy,
                        result.latency_ms, result.cost_usd, result.confidence,
                        int(result.escalated), int(result.rag_used),
                    ),
                )
                await db.commit()
        except Exception as exc:
            await logger.awarning("observability_write_failed", error=str(exc))
