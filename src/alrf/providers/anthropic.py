import os
from typing import override

from anthropic import AsyncAnthropic

from alrf.providers.base import BaseProvider, ProviderResponse

_STOP_REASON_MAP = {"end_turn": "stop", "max_tokens": "length"}


class AnthropicProvider(BaseProvider):
    def __init__(self) -> None:
        self._client = AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            timeout=30.0,
        )

    @override
    async def complete(self, prompt: str, model: str) -> ProviderResponse:
        response = await self._client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        text = next((b.text for b in response.content if hasattr(b, "text")), "")
        stop_reason = _STOP_REASON_MAP.get(response.stop_reason or "", "stop")
        return ProviderResponse(
            text=text,
            model=response.model,
            provider="anthropic",
            stop_reason=stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
