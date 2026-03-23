from pathlib import Path

import aiosqlite


class QueryHistory:
    def __init__(self, db_path: str = ".alrf/routing.db") -> None:
        self._path = Path(db_path)

    async def cost_by_route(self, last_days: int = 7) -> dict[str, float]:
        async with aiosqlite.connect(self._path) as db:
            async with db.execute(
                "SELECT route, COALESCE(SUM(cost_usd), 0.0) "
                "FROM routing_events "
                "WHERE cost_usd IS NOT NULL "
                "  AND created_at >= datetime('now', ?) "
                "GROUP BY route",
                (f"-{last_days} days",),
            ) as cur:
                return {row[0]: row[1] async for row in cur}

    async def latency_percentiles(self) -> dict[str, dict[str, float]]:
        async with aiosqlite.connect(self._path) as db:
            async with db.execute(
                "SELECT provider, latency_ms FROM routing_events ORDER BY provider, latency_ms"
            ) as cur:
                rows = await cur.fetchall()

        by_provider: dict[str, list[int]] = {}
        for provider, latency in rows:
            by_provider.setdefault(provider, []).append(latency)

        result: dict[str, dict[str, float]] = {}
        for provider, values in by_provider.items():
            n = len(values)
            result[provider] = {
                "p50": float(values[n // 2]),
                "p95": float(values[min(int(n * 0.95), n - 1)]),
            }
        return result

    async def escalation_rate(self) -> float:
        async with aiosqlite.connect(self._path) as db:
            async with db.execute(
                "SELECT CAST(SUM(escalated) AS REAL) / COUNT(*) FROM routing_events"
            ) as cur:
                row = await cur.fetchone()
        return round(row[0] or 0.0, 4) if row else 0.0

    async def rag_hit_rate(self) -> float:
        async with aiosqlite.connect(self._path) as db:
            async with db.execute(
                "SELECT CAST(SUM(rag_used) AS REAL) / COUNT(*) FROM routing_events"
            ) as cur:
                row = await cur.fetchone()
        return round(row[0] or 0.0, 4) if row else 0.0
