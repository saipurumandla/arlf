import tempfile

import aiosqlite
import pytest

from alrf.observability.history import QueryHistory


async def _seed(db_path: str, rows: list[dict]) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "CREATE TABLE IF NOT EXISTS routing_events ("
            "id TEXT PRIMARY KEY, query_hash TEXT, route TEXT, provider TEXT, "
            "model TEXT, policy TEXT, latency_ms INTEGER, cost_usd REAL, "
            "confidence REAL, escalated INTEGER DEFAULT 0, "
            "rag_used INTEGER DEFAULT 0, "
            "created_at TEXT DEFAULT (datetime('now')))"
        )
        for i, r in enumerate(rows):
            await db.execute(
                "INSERT INTO routing_events "
                "(id,query_hash,route,provider,model,policy,latency_ms,cost_usd,confidence,escalated,rag_used) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (str(i), f"hash{i}", r["route"], r["provider"], r.get("model", "m"),
                 r.get("policy", "cost_aware"), r["latency_ms"], r.get("cost_usd"),
                 r.get("confidence", 0.8), r.get("escalated", 0), r.get("rag_used", 0)),
            )
        await db.commit()


async def test_cost_by_route_sums_correctly() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = f"{tmp}/r.db"
        await _seed(db, [
            {"route": "fast",      "provider": "openai",     "latency_ms": 300, "cost_usd": 0.001},
            {"route": "fast",      "provider": "openai",     "latency_ms": 250, "cost_usd": 0.002},
            {"route": "reasoning", "provider": "anthropic",  "latency_ms": 800, "cost_usd": 0.010},
        ])
        totals = await QueryHistory(db).cost_by_route(last_days=30)
    assert abs(totals.get("fast", 0) - 0.003) < 0.0001
    assert abs(totals.get("reasoning", 0) - 0.010) < 0.0001


async def test_latency_percentiles_per_provider() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = f"{tmp}/r.db"
        await _seed(db, [
            {"route": "fast", "provider": "openai", "latency_ms": 100},
            {"route": "fast", "provider": "openai", "latency_ms": 200},
            {"route": "fast", "provider": "openai", "latency_ms": 900},
            {"route": "local", "provider": "ollama", "latency_ms": 50},
        ])
        stats = await QueryHistory(db).latency_percentiles()
    assert "openai" in stats
    assert stats["openai"]["p50"] == 200.0


async def test_escalation_rate() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = f"{tmp}/r.db"
        await _seed(db, [
            {"route": "fast", "provider": "openai", "latency_ms": 100, "escalated": 1},
            {"route": "fast", "provider": "openai", "latency_ms": 100, "escalated": 0},
            {"route": "fast", "provider": "openai", "latency_ms": 100, "escalated": 0},
            {"route": "fast", "provider": "openai", "latency_ms": 100, "escalated": 0},
        ])
        rate = await QueryHistory(db).escalation_rate()
    assert abs(rate - 0.25) < 0.001


async def test_rag_hit_rate() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = f"{tmp}/r.db"
        await _seed(db, [
            {"route": "rag+fast", "provider": "openai", "latency_ms": 400, "rag_used": 1},
            {"route": "fast",     "provider": "openai", "latency_ms": 200, "rag_used": 0},
        ])
        rate = await QueryHistory(db).rag_hit_rate()
    assert abs(rate - 0.5) < 0.001
