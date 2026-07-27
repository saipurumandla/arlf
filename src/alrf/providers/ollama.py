import json
from collections.abc import AsyncIterator
from typing import override

import httpx

from alrf.providers.base import BaseProvider, ProviderResponse


class OllamaProvider(BaseProvider):
    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self._base_url = base_url.rstrip("/")

    @override
    async def complete(self, prompt: str, model: str) -> ProviderResponse:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
        data = resp.json()
        return ProviderResponse(
            text=data["message"]["content"],
            model=data["model"],
            provider="ollama",
            stop_reason="stop" if data.get("done_reason") == "stop" else "length",
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )

    @override
    async def stream(self, prompt: str, model: str) -> AsyncIterator[str]:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST", f"{self._base_url}/api/chat", json=payload, timeout=30.0
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line).get("message", {}).get("content", "")
                    if chunk:
                        yield chunk
