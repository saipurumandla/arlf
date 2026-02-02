import os

import structlog
from dotenv import load_dotenv

from alrf.classifier.heuristic import HeuristicClassifier
from alrf.fallback.chain import FallbackChain
from alrf.providers.anthropic import AnthropicProvider
from alrf.providers.openai import OpenAIProvider
from alrf.rag.hook import RAGHook
from alrf.routing.engine import RoutingEngine

load_dotenv()

_log_format = os.getenv("LOG_FORMAT", "console")
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        (
            structlog.dev.ConsoleRenderer()
            if _log_format == "console"
            else structlog.processors.JSONRenderer()
        ),
    ]
)

logger = structlog.get_logger()


class Router:
    def __init__(self, rag_hook: RAGHook | None = None) -> None:
        self._classifier = HeuristicClassifier()
        self._engine = RoutingEngine()
        self._rag_hook = rag_hook
        self._fast_chain = FallbackChain([
            (OpenAIProvider(), "gpt-4o-mini"),
            (AnthropicProvider(), "claude-haiku-4-5-20251001"),
        ])
        self._reasoning_chain = FallbackChain([
            (AnthropicProvider(), "claude-sonnet-4-6"),
            (OpenAIProvider(), "gpt-4o"),
        ])

    async def run(self, query: str) -> str:
        clf_result = self._classifier.classify(query)
        decision = self._engine.decide(clf_result)

        prompt = query
        rag_used = False
        if decision.use_rag and self._rag_hook is not None:
            try:
                chunks = await self._rag_hook.retrieve(query)
                prompt = "[CONTEXT]\n" + "\n".join(chunks) + "\n\n" + query
                rag_used = True
            except Exception as exc:
                await logger.awarning("rag_hook_failed", error=str(exc))

        chain = self._reasoning_chain if decision.tier == "reasoning" else self._fast_chain
        response = await chain.run(prompt)

        await logger.ainfo(
            "routing_decision",
            tier=decision.tier,
            provider=response.provider,
            model=response.model,
            complexity=clf_result.complexity.value,
            rag_used=rag_used,
        )
        return response.text
