import os
from typing import override

from openai import AsyncOpenAI

from alrf.providers.base import BaseProvider, ProviderResponse


class OpenAIProvider(BaseProvider):
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            timeout=30.0,
        )

    @override
    async def complete(self, prompt: str, model: str) -> ProviderResponse:
        response = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        choice = response.choices[0]
        return ProviderResponse(
            text=choice.message.content or "",
            model=response.model,
            provider="openai",
            stop_reason=choice.finish_reason or "stop",
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )
