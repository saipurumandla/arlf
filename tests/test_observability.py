import tempfile
from pathlib import Path

import aiosqlite
import pytest

from alrf import Router, RouterConfig
from alrf.observability.store import ObservabilityStore
from alrf.models.response import RouterResult


def _result(**kwargs: object) -> RouterResult:
    defaults: dict = dict(
        answer="Redis is a cache.",
        route="fast",
        provider="openai",
        model="gpt-4o-mini",
        cost_usd=None,
        latency_ms=200,
        confidence=0.8,
        escalated=False,
        rag_used=False,
        decision_trace=[],
    )
    defaults.update(kwargs)
    return RouterResult(**defaults)  # type: ignore[arg-type]


async def test_store_writes_row_after_record() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ObservabilityStore(db_path=f"{tmp}/test.db")
        await store.record("What is Redis?", _result(), "cost_aware")
        async with aiosqlite.connect(f"{tmp}/test.db") as db:
            async with db.execute("SELECT COUNT(*) FROM routing_events") as cur:
                row = await cur.fetchone()
        assert row is not None and row[0] == 1


async def test_store_writes_correct_fields() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ObservabilityStore(db_path=f"{tmp}/test.db")
        r = _result(route="reasoning", provider="anthropic", confidence=0.91, escalated=True)
        await store.record("some query", r, "quality_first")
        async with aiosqlite.connect(f"{tmp}/test.db") as db:
            async with db.execute("SELECT route, provider, policy, escalated, confidence FROM routing_events") as cur:
                row = await cur.fetchone()
    assert row is not None
    assert row[0] == "reasoning"
    assert row[1] == "anthropic"
    assert row[2] == "quality_first"
    assert row[3] == 1     # escalated stored as 1
    assert abs(row[4] - 0.91) < 0.001


async def test_store_creates_parent_directory() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = f"{tmp}/nested/dir/routing.db"
        store = ObservabilityStore(db_path=db_path)
        await store.record("query", _result(), "cost_aware")
        assert Path(db_path).exists()


async def test_router_writes_to_store_after_run(openai_mock: object) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = f"{tmp}/routing.db"
        config = RouterConfig(policy="quality_first", observability_db_path=db_path, escalation_threshold=0.0)
        await Router(config=config).run("What is Redis?")
        async with aiosqlite.connect(db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM routing_events") as cur:
                row = await cur.fetchone()
    assert row is not None and row[0] == 1
