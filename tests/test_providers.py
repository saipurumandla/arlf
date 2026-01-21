import pytest
import respx

from alrf.providers.ollama import OllamaProvider
from alrf.providers.openai import OpenAIProvider


async def test_ollama_returns_provider_response(ollama_mock: respx.MockRouter) -> None:
    provider = OllamaProvider()
    resp = await provider.complete("What is Redis?", "llama3.2")
    assert resp.provider == "ollama"
    assert resp.text == "The default pool size is 10."
    assert resp.stop_reason == "stop"


async def test_ollama_makes_one_request(ollama_mock: respx.MockRouter) -> None:
    await OllamaProvider().complete("hello", "llama3.2")
    assert ollama_mock.calls.call_count == 1


async def test_ollama_custom_base_url(ollama_mock: respx.MockRouter) -> None:
    # OllamaProvider is configurable — useful when Ollama runs on a non-default port
    provider = OllamaProvider(base_url="http://localhost:11434")
    resp = await provider.complete("ping", "llama3.2")
    assert resp.model == "llama3.2"


async def test_openai_compat_uses_base_url(openai_mock: respx.MockRouter) -> None:
    # Passing base_url points the adapter at a local OpenAI-compatible server
    # (LM Studio, vLLM, llama.cpp) without any code changes
    provider = OpenAIProvider(base_url="https://api.openai.com/v1", api_key="local")
    resp = await provider.complete("hello", "gpt-4o-mini")
    assert resp.provider == "openai"
    assert resp.text == "The default pool size is 10."


async def test_openai_default_has_no_base_url(openai_mock: respx.MockRouter) -> None:
    provider = OpenAIProvider()
    resp = await provider.complete("hello", "gpt-4o-mini")
    assert resp.stop_reason == "stop"
