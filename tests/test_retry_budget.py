from unittest.mock import AsyncMock, patch

import pytest

from alrf import Router, RouterConfig
from alrf.models.response import RouterResult
from alrf.providers.base import ProviderResponse


def _resp(text: str = "ok", stop_reason: str = "stop") -> ProviderResponse:
    return ProviderResponse(
        text=text, model="gpt-4o-mini", provider="openai",
        stop_reason=stop_reason, input_tokens=5, output_tokens=3,
    )


async def test_router_returns_router_result(openai_mock: object) -> None:
    # quality_first routes simple → fast (OpenAI), avoids needing ollama_mock
    config = RouterConfig(policy="quality_first")
    result = await Router(config=config).run("What is Redis?")
    assert isinstance(result, RouterResult)
    assert result.answer
    assert result.provider == "openai"


async def test_result_has_all_fields(openai_mock: object) -> None:
    config = RouterConfig(policy="quality_first")
    result = await Router(config=config).run("What is Redis?")
    assert result.latency_ms >= 0
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.decision_trace, list)
    assert len(result.decision_trace) >= 1


async def test_escalated_flag_set_when_confidence_low(openai_mock: object) -> None:
    hedge = _resp("I'm not sure, I don't know, unclear, I cannot say, it depends on context.")

    async def always_hedge(prompt: str, model: str) -> ProviderResponse:
        return hedge

    config = RouterConfig(policy="quality_first", escalation_threshold=0.99, retry_budget=3)
    with patch("alrf.router.OpenAIProvider") as MockOpenAI, \
         patch("alrf.router.AnthropicProvider") as MockAnthropic:
        MockOpenAI.return_value.complete = always_hedge
        MockAnthropic.return_value.complete = always_hedge
        result = await Router(config=config).run("What is Redis?")

    assert result.escalated is True


async def test_retry_budget_caps_attempts() -> None:
    call_count = 0

    async def counting(prompt: str, model: str) -> ProviderResponse:
        nonlocal call_count
        call_count += 1
        return _resp("I'm not sure, I don't know, unclear, I cannot, it depends.")

    config = RouterConfig(policy="quality_first", escalation_threshold=0.99, retry_budget=2)
    with patch("alrf.router.OpenAIProvider") as MockOpenAI, \
         patch("alrf.router.AnthropicProvider") as MockAnthropic, \
         patch("alrf.router.OllamaProvider") as MockOllama:
        MockOpenAI.return_value.complete = counting
        MockAnthropic.return_value.complete = counting
        MockOllama.return_value.complete = counting
        await Router(config=config).run("What is Redis?")

    assert call_count <= config.retry_budget


async def test_no_escalation_when_confidence_above_threshold(openai_mock: object) -> None:
    config = RouterConfig(policy="quality_first", escalation_threshold=0.0)
    result = await Router(config=config).run("What is Redis?")
    assert result.escalated is False


async def test_cost_aware_routes_simple_to_local(ollama_mock: object) -> None:
    config = RouterConfig(policy="cost_aware")
    result = await Router(config=config).run("What is Redis?")
    assert result.provider == "ollama"
    assert result.route == "local"
