from pathlib import Path

import aiosqlite

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS cache_stats (
    route  TEXT PRIMARY KEY,
    hits   INTEGER NOT NULL DEFAULT 0,
    misses INTEGER NOT NULL DEFAULT 0
);
"""


class CacheStats:
    def __init__(self, db_path: str = ".alrf/routing.db") -> None:
        self._path = Path(db_path)

    async def _ensure_db(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._path) as db:
            await db.execute(_CREATE_SQL)
            await db.commit()

    async def record(self, route: str, hit: bool) -> None:
        bump = (
            "UPDATE cache_stats SET hits = hits + 1 WHERE route = ?"
            if hit
            else "UPDATE cache_stats SET misses = misses + 1 WHERE route = ?"
        )
        await self._ensure_db()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                "INSERT INTO cache_stats (route, hits, misses) VALUES (?,0,0) "
                "ON CONFLICT(route) DO NOTHING",
                (route,),
            )
            await db.execute(bump, (route,))
            await db.commit()

    async def hit_rate_by_route(self) -> dict[str, float]:
        await self._ensure_db()
        async with aiosqlite.connect(self._path) as db:
            async with db.execute("SELECT route, hits, misses FROM cache_stats") as cur:
                rows = await cur.fetchall()
        return {
            route: round(hits / (hits + misses), 4)
            for route, hits, misses in rows
            if hits + misses > 0
        }
