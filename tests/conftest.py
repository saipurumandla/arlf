import httpx
import pytest
import respx

from alrf.providers.base import ProviderResponse

MOCK_OPENAI_JSON = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "The default pool size is 10."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 10, "total_tokens": 22},
}

MOCK_ANTHROPIC_JSON = {
    "id": "msg-test",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5-20251001",
    "content": [{"type": "text", "text": "The default pool size is 10."}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 12, "output_tokens": 10},
}


@pytest.fixture
def mock_openai_response() -> ProviderResponse:
    return ProviderResponse(
        text="The default pool size is 10.",
        model="gpt-4o-mini",
        provider="openai",
        stop_reason="stop",
        input_tokens=12,
        output_tokens=10,
    )


@pytest.fixture
def mock_anthropic_response() -> ProviderResponse:
    return ProviderResponse(
        text="The default pool size is 10.",
        model="claude-haiku-4-5-20251001",
        provider="anthropic",
        stop_reason="stop",
        input_tokens=12,
        output_tokens=10,
    )


@pytest.fixture
def openai_mock() -> respx.MockRouter:
    with respx.mock(base_url="https://api.openai.com", assert_all_called=False) as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=MOCK_OPENAI_JSON)
        )
        yield mock  # type: ignore[misc]


@pytest.fixture
def anthropic_mock() -> respx.MockRouter:
    with respx.mock(base_url="https://api.anthropic.com", assert_all_called=False) as mock:
        mock.post("/v1/messages").mock(
            return_value=httpx.Response(200, json=MOCK_ANTHROPIC_JSON)
        )
        yield mock  # type: ignore[misc]


MOCK_OLLAMA_JSON = {
    "model": "llama3.2",
    "message": {"role": "assistant", "content": "The default pool size is 10."},
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 12,
    "eval_count": 10,
}


@pytest.fixture
def mock_ollama_response() -> ProviderResponse:
    return ProviderResponse(
        text="The default pool size is 10.",
        model="llama3.2",
        provider="ollama",
        stop_reason="stop",
        input_tokens=12,
        output_tokens=10,
    )


@pytest.fixture
def ollama_mock() -> respx.MockRouter:
    with respx.mock(base_url="http://localhost:11434", assert_all_called=False) as mock:
        mock.post("/api/chat").mock(
            return_value=httpx.Response(200, json=MOCK_OLLAMA_JSON)
        )
        yield mock  # type: ignore[misc]
