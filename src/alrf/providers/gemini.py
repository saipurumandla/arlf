import os
from typing import override

import google.generativeai as genai

from alrf.providers.base import BaseProvider, ProviderResponse


class GeminiProvider(BaseProvider):
    def __init__(self) -> None:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))

    @override
    async def complete(self, prompt: str, model: str) -> ProviderResponse:
        gen_model = genai.GenerativeModel(model)
        response = await gen_model.generate_content_async(prompt)
        text = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        return ProviderResponse(
            text=text,
            model=model,
            provider="gemini",
            stop_reason="stop" if response.candidates else "length",
            input_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
        )
