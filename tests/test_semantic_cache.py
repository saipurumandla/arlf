import respx

from alrf import Router, RouterConfig
from alrf.cache.semantic import SemanticCache
from alrf.models.response import RouterResult


def _result(**kwargs: object) -> RouterResult:
    defaults: dict = dict(
        answer="Redis is an in-memory key-value store.",
        route="fast",
        provider="openai",
        model="gpt-4o-mini",
        cost_usd=0.0004,
        latency_ms=180,
        confidence=0.82,
        escalated=False,
        rag_used=False,
        decision_trace=[],
    )
    defaults.update(kwargs)
    return RouterResult(**defaults)  # type: ignore[arg-type]


async def test_near_identical_query_hits_cache(tmp_path: object) -> None:
    cache = SemanticCache(db_path=f"{tmp_path}/cache.db")
    await cache.store("What is Redis?", _result())

    hit = await cache.lookup("what is redis")
    assert hit is not None
    assert hit.answer == "Redis is an in-memory key-value store."
    assert hit.cached is True
    assert hit.cost_usd == 0.0


async def test_different_query_misses(tmp_path: object) -> None:
    cache = SemanticCache(db_path=f"{tmp_path}/cache.db")
    await cache.store("What is Redis?", _result())

    assert await cache.lookup("Implement a retry decorator with jitter") is None


async def test_partial_overlap_below_threshold_misses(tmp_path: object) -> None:
    cache = SemanticCache(db_path=f"{tmp_path}/cache.db")
    await cache.store("What is Redis?", _result())

    assert await cache.lookup("What is Redis used for in a web stack?") is None


async def test_empty_cache_returns_none(tmp_path: object) -> None:
    cache = SemanticCache(db_path=f"{tmp_path}/cache.db")
    assert await cache.lookup("anything") is None


async def test_lower_threshold_allows_looser_match(tmp_path: object) -> None:
    cache = SemanticCache(db_path=f"{tmp_path}/cache.db", threshold=0.7)
    await cache.store("What is Redis?", _result())

    assert await cache.lookup("What is Redis used for?") is not None


async def test_router_serves_second_run_from_cache(openai_mock: respx.MockRouter) -> None:
    config = RouterConfig(policy="quality_first", escalation_threshold=0.0)
    router = Router(config=config)

    first = await router.run("What is Redis?")
    second = await router.run("What is Redis?")

    assert first.cached is False
    assert second.cached is True
    assert second.answer == first.answer
    assert openai_mock.calls.call_count == 1


async def test_router_skips_cache_when_disabled(openai_mock: respx.MockRouter) -> None:
    config = RouterConfig(policy="quality_first", escalation_threshold=0.0, cache_enabled=False)
    router = Router(config=config)

    await router.run("What is Redis?")
    second = await router.run("What is Redis?")

    assert second.cached is False
    assert openai_mock.calls.call_count == 2
