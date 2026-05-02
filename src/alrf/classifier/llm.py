import json
import re

import httpx
import tiktoken

from alrf.classifier.base import ClassifierResult, QueryComplexity, QueryIntent

_enc = tiktoken.get_encoding("cl100k_base")

_PROMPT = """\
You are a query classifier for an LLM routing system. Classify the user query.

Return ONLY a JSON object — no explanation, no markdown, just JSON:
{{
  "complexity": "simple" | "moderate" | "complex",
  "intent": "qa" | "debug" | "explanation" | "code",
  "needs_retrieval": true | false,
  "signals": ["short list of reason strings"]
}}

Complexity guide:
  simple   - short factual lookup, one clear answer
  moderate - requires context or some reasoning
  complex  - multi-step, deep debugging, or detailed implementation

needs_retrieval: true when query asks why/how about something specific and is detailed enough to benefit from retrieved context.

Examples:
Query: What is Redis?
{{"complexity":"simple","intent":"qa","needs_retrieval":false,"signals":["short_factual"]}}

Query: Why does my connection pool exhaust under load and how do I fix it?
{{"complexity":"complex","intent":"debug","needs_retrieval":true,"signals":["troubleshooting","causal_question"]}}

Query: Implement a retry decorator with exponential backoff and jitter
{{"complexity":"complex","intent":"code","needs_retrieval":false,"signals":["implementation_request"]}}

Query: How does async/await work in Python?
{{"complexity":"moderate","intent":"explanation","needs_retrieval":false,"signals":["conceptual_question"]}}

Query: {query}
"""


def _parse(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON in response: {text[:120]}")
    return json.loads(match.group())


class LLMClassifier:
    """Classifies queries using a local Ollama model instead of keyword heuristics."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    async def classify(self, query: str) -> ClassifierResult:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if len(query) > 2000:
            raise ValueError("query exceeds 2000 characters")

        prompt = _PROMPT.format(query=query)

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                timeout=30.0,
            )
            resp.raise_for_status()

        data = _parse(resp.json()["message"]["content"])

        return ClassifierResult(
            complexity=QueryComplexity(data.get("complexity", "moderate")),
            intent=QueryIntent(data.get("intent", "qa")),
            token_estimate=len(_enc.encode(query)),
            confidence=0.85,
            needs_retrieval=bool(data.get("needs_retrieval", False)),
            signals=data.get("signals", []),
        )
