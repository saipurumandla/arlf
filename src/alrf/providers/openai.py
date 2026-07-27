import os
from collections.abc import AsyncIterator
from typing import override

from openai import AsyncOpenAI

from alrf.providers.base import BaseProvider, ProviderResponse


class OpenAIProvider(BaseProvider):
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        # base_url lets this adapter talk to any OpenAI-compatible local server
        # (LM Studio, vLLM, llama.cpp) — leave None for the real OpenAI API
        self._client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            base_url=base_url,
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

    @override
    async def stream(self, prompt: str, model: str) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
