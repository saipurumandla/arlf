from unittest.mock import AsyncMock, MagicMock, patch

from alrf.providers.gemini import GeminiProvider
from alrf.providers.base import ProviderResponse


def _mock_gemini_response(text: str = "Gemini answer.") -> MagicMock:
    candidate = MagicMock()
    response = MagicMock()
    response.text = text
    response.candidates = [candidate]
    usage = MagicMock()
    usage.prompt_token_count = 10
    usage.candidates_token_count = 8
    response.usage_metadata = usage
    return response


async def test_gemini_returns_provider_response() -> None:
    mock_resp = _mock_gemini_response("The answer is 42.")
    with patch("google.generativeai.GenerativeModel") as MockModel:
        instance = MockModel.return_value
        instance.generate_content_async = AsyncMock(return_value=mock_resp)
        provider = GeminiProvider()
        result = await provider.complete("What is the answer?", "gemini-1.5-flash")

    assert isinstance(result, ProviderResponse)
    assert result.provider == "gemini"
    assert result.text == "The answer is 42."
    assert result.stop_reason == "stop"


async def test_gemini_records_token_counts() -> None:
    mock_resp = _mock_gemini_response()
    with patch("google.generativeai.GenerativeModel") as MockModel:
        instance = MockModel.return_value
        instance.generate_content_async = AsyncMock(return_value=mock_resp)
        result = await GeminiProvider().complete("hello", "gemini-1.5-flash")

    assert result.input_tokens == 10
    assert result.output_tokens == 8


async def test_gemini_empty_candidates_gives_length_stop() -> None:
    mock_resp = _mock_gemini_response()
    mock_resp.candidates = []
    with patch("google.generativeai.GenerativeModel") as MockModel:
        instance = MockModel.return_value
        instance.generate_content_async = AsyncMock(return_value=mock_resp)
        result = await GeminiProvider().complete("hello", "gemini-1.5-flash")

    assert result.stop_reason == "length"


def test_router_result_schema() -> None:
    from alrf.models.response import RouterResult
    r = RouterResult(
        answer="yes",
        route="fast",
        provider="openai",
        model="gpt-4o-mini",
        cost_usd=0.001,
        latency_ms=420,
        confidence=0.85,
        escalated=False,
        rag_used=False,
        decision_trace=[],
    )
    assert r.answer == "yes"
    assert r.cost_usd == 0.001


def test_router_result_nullable_cost() -> None:
    from alrf.models.response import RouterResult
    r = RouterResult(
        answer="ok",
        route="local",
        provider="ollama",
        model="llama3.2",
        cost_usd=None,
        latency_ms=80,
        confidence=0.9,
        escalated=False,
        rag_used=False,
        decision_trace=[],
    )
    assert r.cost_usd is None
