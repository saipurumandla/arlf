from collections.abc import AsyncIterator
from unittest.mock import patch

import pytest

from alrf import Router, RouterConfig
from alrf.providers.base import ProviderResponse
from alrf.providers.gemini import GeminiProvider


async def _chunks(prompt: str, model: str) -> AsyncIterator[str]:
    for part in ["Redis ", "is an ", "in-memory ", "key-value store."]:
        yield part


async def _hedging_chunks(prompt: str, model: str) -> AsyncIterator[str]:
    for part in ["I'm not sure. ", "It depends. ", "I don't know."]:
        yield part


@pytest.fixture
def config() -> RouterConfig:
    return RouterConfig(policy="quality_first", escalation_threshold=0.7)


async def test_stream_yields_chunks_in_order(config: RouterConfig) -> None:
    with patch("alrf.router.OpenAIProvider") as MockOpenAI:
        MockOpenAI.return_value.stream = _chunks
        received = [chunk async for chunk in Router(config=config).stream("What is Redis?")]

    assert received == ["Redis ", "is an ", "in-memory ", "key-value store."]


async def test_confidence_scored_on_joined_buffer(config: RouterConfig) -> None:
    router = Router(config=config)
    with patch("alrf.router.OpenAIProvider") as MockOpenAI:
        MockOpenAI.return_value.stream = _chunks
        async for _ in router.stream("What is Redis?"):
            pass

    result = router.last_result
    assert result is not None
    assert result.answer == "Redis is an in-memory key-value store."
    assert result.confidence > 0.0
    assert result.escalated is False


async def test_low_confidence_stream_is_not_retried(config: RouterConfig) -> None:
    router = Router(config=config)
    with patch("alrf.router.OpenAIProvider") as MockOpenAI:
        MockOpenAI.return_value.stream = _hedging_chunks
        chunks = [chunk async for chunk in router.stream("What is Redis?")]

    result = router.last_result
    assert result is not None
    assert result.confidence < config.escalation_threshold
    assert result.escalated is False
    assert len(chunks) == 3


async def test_last_result_is_none_before_streaming(config: RouterConfig) -> None:
    assert Router(config=config).last_result is None


async def test_gemini_falls_back_to_a_single_chunk() -> None:
    response = ProviderResponse(
        text="Redis is an in-memory key-value store.",
        model="gemini-1.5-flash",
        provider="gemini",
        stop_reason="stop",
        input_tokens=8,
        output_tokens=9,
    )
    provider = GeminiProvider()
    with patch.object(GeminiProvider, "complete", return_value=response):
        chunks = [chunk async for chunk in provider.stream("What is Redis?", "gemini-1.5-flash")]

    assert chunks == ["Redis is an in-memory key-value store."]
