import httpx
import pytest
from unittest.mock import AsyncMock

from alrf.fallback.chain import FallbackChain
from alrf.providers.base import ProviderResponse


def _mock_response(text: str = "ok") -> ProviderResponse:
    return ProviderResponse(
        text=text, model="gpt-4o-mini", provider="openai",
        stop_reason="stop", input_tokens=5, output_tokens=3,
    )


@pytest.fixture
def ok_provider() -> AsyncMock:
    p = AsyncMock()
    p.complete.return_value = _mock_response("success")
    return p


@pytest.fixture
def timeout_provider() -> AsyncMock:
    p = AsyncMock()
    p.complete.side_effect = httpx.TimeoutException("timed out")
    return p


@pytest.fixture
def http_error_provider() -> AsyncMock:
    p = AsyncMock()
    p.complete.side_effect = httpx.HTTPStatusError(
        "503",
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        response=httpx.Response(503),
    )
    return p


async def test_single_provider_success(ok_provider: AsyncMock) -> None:
    chain = FallbackChain([(ok_provider, "gpt-4o-mini")])
    resp = await chain.run("hello")
    assert resp.text == "success"


async def test_falls_back_on_timeout(timeout_provider: AsyncMock, ok_provider: AsyncMock) -> None:
    chain = FallbackChain([
        (timeout_provider, "gpt-4o-mini"),
        (ok_provider, "claude-haiku-4-5-20251001"),
    ])
    resp = await chain.run("hello")
    assert resp.text == "success"
    timeout_provider.complete.assert_called_once()


async def test_falls_back_on_http_error(http_error_provider: AsyncMock, ok_provider: AsyncMock) -> None:
    chain = FallbackChain([
        (http_error_provider, "gpt-4o-mini"),
        (ok_provider, "claude-haiku-4-5-20251001"),
    ])
    resp = await chain.run("hello")
    assert resp.text == "success"


async def test_raises_when_all_fail(timeout_provider: AsyncMock, http_error_provider: AsyncMock) -> None:
    chain = FallbackChain([
        (timeout_provider, "gpt-4o-mini"),
        (http_error_provider, "claude-haiku-4-5-20251001"),
    ])
    with pytest.raises((httpx.TimeoutException, httpx.HTTPStatusError)):
        await chain.run("hello")


async def test_second_slot_not_called_on_first_success(ok_provider: AsyncMock, timeout_provider: AsyncMock) -> None:
    chain = FallbackChain([(ok_provider, "gpt-4o-mini"), (timeout_provider, "backup")])
    await chain.run("hello")
    timeout_provider.complete.assert_not_called()


def test_empty_slots_raises() -> None:
    with pytest.raises(ValueError):
        FallbackChain([])
