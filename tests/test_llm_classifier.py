import json
import httpx
import pytest
import respx

from alrf.classifier.base import QueryComplexity, QueryIntent
from alrf.classifier.llm import LLMClassifier


def _ollama_reply(payload: dict) -> httpx.Response:
    body = {
        "model": "llama3.2",
        "message": {"role": "assistant", "content": json.dumps(payload)},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 80,
        "eval_count": 30,
    }
    return httpx.Response(200, json=body)


@pytest.fixture
def clf() -> LLMClassifier:
    return LLMClassifier()


async def test_simple_factual_query(clf: LLMClassifier) -> None:
    payload = {"complexity": "simple", "intent": "qa", "needs_retrieval": False, "signals": ["short_factual"]}
    with respx.mock(base_url="http://localhost:11434", assert_all_called=False) as mock:
        mock.post("/api/chat").mock(return_value=_ollama_reply(payload))
        result = await clf.classify("What is Redis?")
    assert result.complexity == QueryComplexity.simple
    assert result.intent == QueryIntent.qa
    assert result.needs_retrieval is False


async def test_complex_debug_query(clf: LLMClassifier) -> None:
    payload = {"complexity": "complex", "intent": "debug", "needs_retrieval": True, "signals": ["troubleshooting", "causal_question"]}
    with respx.mock(base_url="http://localhost:11434", assert_all_called=False) as mock:
        mock.post("/api/chat").mock(return_value=_ollama_reply(payload))
        result = await clf.classify("Why does my connection pool exhaust under load?")
    assert result.complexity == QueryComplexity.complex
    assert result.intent == QueryIntent.debug
    assert result.needs_retrieval is True


async def test_code_request(clf: LLMClassifier) -> None:
    payload = {"complexity": "complex", "intent": "code", "needs_retrieval": False, "signals": ["implementation_request"]}
    with respx.mock(base_url="http://localhost:11434", assert_all_called=False) as mock:
        mock.post("/api/chat").mock(return_value=_ollama_reply(payload))
        result = await clf.classify("Implement a retry decorator with exponential backoff")
    assert result.intent == QueryIntent.code
    assert result.complexity == QueryComplexity.complex


async def test_signals_carried_through(clf: LLMClassifier) -> None:
    payload = {"complexity": "moderate", "intent": "explanation", "needs_retrieval": False, "signals": ["conceptual", "how_does"]}
    with respx.mock(base_url="http://localhost:11434", assert_all_called=False) as mock:
        mock.post("/api/chat").mock(return_value=_ollama_reply(payload))
        result = await clf.classify("How does async/await work?")
    assert "conceptual" in result.signals
    assert "how_does" in result.signals


async def test_token_estimate_always_set(clf: LLMClassifier) -> None:
    payload = {"complexity": "simple", "intent": "qa", "needs_retrieval": False, "signals": []}
    with respx.mock(base_url="http://localhost:11434", assert_all_called=False) as mock:
        mock.post("/api/chat").mock(return_value=_ollama_reply(payload))
        result = await clf.classify("ping")
    assert result.token_estimate > 0


async def test_llm_classifier_confidence_higher_than_heuristic(clf: LLMClassifier) -> None:
    payload = {"complexity": "moderate", "intent": "qa", "needs_retrieval": False, "signals": []}
    with respx.mock(base_url="http://localhost:11434", assert_all_called=False) as mock:
        mock.post("/api/chat").mock(return_value=_ollama_reply(payload))
        result = await clf.classify("What is the default timeout?")
    assert result.confidence == 0.85


async def test_empty_query_raises(clf: LLMClassifier) -> None:
    with pytest.raises(ValueError):
        await clf.classify("")


async def test_too_long_query_raises(clf: LLMClassifier) -> None:
    with pytest.raises(ValueError):
        await clf.classify("x" * 2001)
