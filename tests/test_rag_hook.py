import pytest
from unittest.mock import AsyncMock

from alrf import RouterConfig
from alrf.rag.hook import RAGHook
from alrf.router import Router


@pytest.fixture
def rag_hook() -> AsyncMock:
    hook = AsyncMock()
    hook.retrieve.return_value = [
        "Redis is an in-memory key-value store.",
        "It supports strings, hashes, lists, sets, and sorted sets.",
    ]
    return hook


async def test_rag_hook_called_when_retrieval_needed(rag_hook: AsyncMock, openai_mock: object) -> None:
    # Long causal query → needs_retrieval=True (causal signal + token_est > 30)
    # escalation_threshold=0.0 keeps test focused on RAG behaviour, not escalation logic
    query = (
        "how does the garbage collector work in CPython and why does it sometimes cause "
        "unexpected latency spikes in long-running async services that hold lots of live object references in memory"
    )
    config = RouterConfig(escalation_threshold=0.0)
    router = Router(config=config, rag_hook=rag_hook)
    await router.run(query)
    rag_hook.retrieve.assert_called_once_with(query)


async def test_rag_hook_not_called_for_simple_query(rag_hook: AsyncMock, openai_mock: object) -> None:
    # quality_first routes simple → fast (OpenAI), keeps test away from Ollama
    config = RouterConfig(policy="quality_first")
    router = Router(config=config, rag_hook=rag_hook)
    await router.run("What is Redis?")
    rag_hook.retrieve.assert_not_called()


async def test_router_works_without_rag_hook(openai_mock: object) -> None:
    config = RouterConfig(policy="quality_first")
    result = await Router(config=config).run("What is Redis?")
    assert result.answer


async def test_rag_hook_error_does_not_crash_router(openai_mock: object) -> None:
    broken_hook = AsyncMock()
    broken_hook.retrieve.side_effect = RuntimeError("retriever is down")
    config = RouterConfig(policy="quality_first")
    result = await Router(config=config, rag_hook=broken_hook).run("What is Redis?")
    assert result.answer


def test_rag_hook_protocol_satisfied_by_duck_typing() -> None:
    class MyRetriever:
        async def retrieve(self, query: str) -> list[str]:
            return ["doc1"]

    assert isinstance(MyRetriever(), RAGHook)
