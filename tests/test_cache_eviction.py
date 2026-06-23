import time

import aiosqlite

from alrf.cache.semantic import SemanticCache
from alrf.models.response import RouterResult


def _result(route: str = "fast") -> RouterResult:
    return RouterResult(
        answer="Redis is an in-memory key-value store.",
        route=route,
        provider="openai",
        model="gpt-4o-mini",
        cost_usd=0.0004,
        latency_ms=180,
        confidence=0.82,
        escalated=False,
        rag_used=False,
        decision_trace=[],
    )


async def _age_entries(db_path: str, seconds: float) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE cache_entries SET created_at = ?", (time.time() - seconds,)
        )
        await db.commit()


async def _count(db_path: str) -> int:
    async with aiosqlite.connect(db_path) as db:
        async with db.execute("SELECT COUNT(*) FROM cache_entries") as cur:
            row = await cur.fetchone()
    return int(row[0]) if row else 0


async def test_expired_entry_is_not_served(tmp_path: object) -> None:
    db = f"{tmp_path}/cache.db"
    cache = SemanticCache(db_path=db, ttl_seconds=60)
    await cache.store("What is Redis?", _result())
    await _age_entries(db, 120)

    assert await cache.lookup("What is Redis?") is None


async def test_entry_within_ttl_is_served(tmp_path: object) -> None:
    db = f"{tmp_path}/cache.db"
    cache = SemanticCache(db_path=db, ttl_seconds=3600)
    await cache.store("What is Redis?", _result())
    await _age_entries(db, 120)

    assert await cache.lookup("What is Redis?") is not None


async def test_expired_entry_is_deleted_on_lookup(tmp_path: object) -> None:
    db = f"{tmp_path}/cache.db"
    cache = SemanticCache(db_path=db, ttl_seconds=60)
    await cache.store("What is Redis?", _result())
    await _age_entries(db, 120)
    await cache.lookup("anything")

    assert await _count(db) == 0


async def test_lru_eviction_at_max_entries(tmp_path: object) -> None:
    db = f"{tmp_path}/cache.db"
    cache = SemanticCache(db_path=db, max_entries=2)
    await cache.store("What is Redis?", _result())
    await cache.store("How do I tune the connection pool?", _result())
    await cache.store("Implement a retry decorator with jitter", _result())

    assert await _count(db) == 2
    assert await cache.lookup("what is redis") is None


async def test_recently_used_entry_survives_eviction(tmp_path: object) -> None:
    db = f"{tmp_path}/cache.db"
    cache = SemanticCache(db_path=db, max_entries=2)
    await cache.store("What is Redis?", _result())
    await cache.store("How do I tune the connection pool?", _result())
    await cache.lookup("what is redis")
    await cache.store("Implement a retry decorator with jitter", _result())

    assert await cache.lookup("what is redis") is not None


async def test_hit_rate_recorded_per_route(tmp_path: object) -> None:
    db = f"{tmp_path}/cache.db"
    cache = SemanticCache(db_path=db)
    await cache.store("What is Redis?", _result(route="fast"))
    await cache.store("Design a sharding strategy for postgres", _result(route="reasoning"))
    await cache.lookup("what is redis")

    rates = await cache.hit_rate_by_route()
    assert rates["fast"] == 0.5
    assert rates["reasoning"] == 0.0
